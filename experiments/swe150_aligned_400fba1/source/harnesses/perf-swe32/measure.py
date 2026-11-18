"""Per-request and batch measurements of one multi-session stage-2 server run (stdlib only; the GPU job and the verifier).

Stage 2 decodes several sessions' requests in one vLLM step (max_num_seqs = N).
Admissions stay one at a time on the engine thread and stall every running
decode while they last. Records are per request (session, per-session turn),
plus the server's decode batch over time. Sources, all archived, so the
verifier recomputes every record exactly:
- /metrics chain (ChainProbe): one scrape when the driver starts and one after
  every response, under one lock. Requests overlap, so the window between two
  chain scrapes may hold several finished requests or none. The k-th response
  is matched to the window that holds the k-th finish (cumulative finished
  count since the first scrape): its per-request histogram values count only
  if that window holds exactly this one finish with this request's generated
  tokens (`window_own`); otherwise TPOT is the window mean (`tpot_source`
  'window_mean') and the other per-request values are left out (`metrics_note`).
- /metrics samples (the same probe): one scrape every `sample_interval` seconds
  while the sessions run, filtered to the batch families (running and waiting
  requests, KV usage, generated tokens, preemptions, inter-token latency,
  finished requests). They give the decode batch size over time, aggregate
  decode tokens/s and inter-token latency per batch size, and the preemption
  counter (smoke_checks: no-preemption).
- The runtime's native terminal receipt and runtime row: phase-1 time,
  worker-operation times, the request's own H2D bytes and active KV bytes,
  page counts, the engine stall of its admission (`admission_engine_ms`), its
  CAPACITY_WAIT retries (`capacity_waits`) and its queue wait.
- The server log's content-free admission timing lines (REMOTE_RA_REGISTERED_ADD)
  with their log positions: admission blocks, their acks, and the engine's
  CAPACITY_WAIT refusals.
- The client timeline: send and receive times of every request on one clock.
Missing metric names are recorded (`metrics_error`) and never fail a request.

ms2a (runtime 9b18421, goal_async): on a server whose sessions decode together,
a goal_query admission parks at its Goal. Its engine-thread work is then two
pieces in the server log: `engine_dispatch` (up to the park) and, later,
`engine_resume` (the rest of the admission), inside which its
`engine_ack_queued` (matched by `call_id`) is written; between them the Goal is
built on a background thread while the other requests keep decoding.
`engine_pieces` reads that structure; every record carries its Goal timing
(`goal_async_ms` parked to resumed, `goal_construct_ms` on the Goal thread,
from the native record), the Goal decision fields, and its parked interval on
the client clock (`park_start_s`, `park_end_s`: the admission start plus the
dispatch piece, then `goal_async_ms`), and `parking` summarizes the decode
that went on while Goals were parked.

perf-adm (adm-e's (b), in adm-f): with `--phase1-in-step` on a server whose sessions decode together, an admission parks at its phase
1 as at a Goal (dispatch up to the park, a later resume holding its ack; a goal_query admission may park at its Goal
and then again at its phase 1: the resume between them acknowledges nothing, `reparks`); every record carries the
native `phase1_mode` (mixed, probe_only, fallback), whether it parked (`phase1_parked`, with `phase1_park_ms`), a mixed
step's `phase1_step_ms` and `phase1_step_rows`, and with `--gdn-device-handoff` the restore's `d2d_bytes`. `phase1`
summarizes them per server: the modes, the share of continuing turns that ran mixed, the step that carried a probe
against the steady inter-token latency of a plain decode step of the same batch without it (rows - 1), and the stall
share once the probe steps' excess and the parked fallbacks are counted too (`stall_with_steps_fraction`).

(f) (adm-f, `--admission-overlap K`): up to K admissions in flight, so admission blocks overlap and several admissions
may be parked at once. K is read from the blocks' pinned ACK wait (`log_permits`: 300 + 120 (K - 1) s); with K > 1
`engine_pieces` allows several parked calls (a resume that acknowledges nothing while several are parked goes to
`unattributed`), and `admissions` pairs each block with its ADD by elimination (the ADD's wait equals the block's, a
Cancel waits 60 s; each ADD belongs to one block), then FIFO; what stays ambiguous is `add_ambiguous`. Every record
carries `lease_wait_ms` (parked for the one phase-1 lease, (f) with (b)); `overlap` summarizes per server the queue
wait (mean, p95, share of the request time), Goal wait and construction, lease waits, engine-thread admission time,
CAPACITY_WAITs, turns/min and the rewriter's busy share.
"""
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

PREFIX = 'vllm:'
# smoke-dp1: run labels (A-full, C-goal_query, ...) differ from their policies; only the warm-up server is the warm-up.
WARMUP_LABEL = 'warmup'
METRICS_PATH = '/metrics'
METRICS_DIR = 'metrics'
SCRAPE_TIMEOUT = 10
SETTLE_SECONDS, SETTLE_INTERVAL = 3.0, 0.2
# perf-int (scg2 r3): 0.25 s (perf-ms16: 1.0). A batch_profile cell at b needs two consecutive samples that both count b
# running requests; with CUDA graphs 16 sessions decode ~80-token replies in ~2-3 s per turn, so 1-s samples may leave
# no steady interval at b = 16 (scg2 job 3338893's b-C-pw had no b = 8 cell at 1 s). A sample file is ~6 KB.
SAMPLE_INTERVAL = 0.25
# Per-request histograms: a window's count delta is the number of requests that finished in it.
PER_REQUEST = (('e2e_s', 'vllm:e2e_request_latency_seconds'), ('ttft_s', 'vllm:time_to_first_token_seconds'),
               ('queue_s', 'vllm:request_queue_time_seconds'),
               ('prefill_phase2_s', 'vllm:request_prefill_time_seconds'),
               ('decode_s', 'vllm:request_decode_time_seconds'),
               ('inference_s', 'vllm:request_inference_time_seconds'))
FINISHED = 'vllm:e2e_request_latency_seconds'
TPOT_PER_REQUEST = 'vllm:request_time_per_output_token_seconds'
TPOT_PER_TOKEN = ('vllm:inter_token_latency_seconds', 'vllm:time_per_output_token_seconds')
GENERATION_PER_REQUEST = 'vllm:request_generation_tokens'
COUNTERS = (('generation_tokens_metric', 'vllm:generation_tokens_total'),
            ('prompt_tokens_metric', 'vllm:prompt_tokens_total'),
            ('requests_finished_metric', 'vllm:request_success_total'),
            ('preemptions_metric', 'vllm:num_preemptions_total'),
            ('prefix_cache_queries_metric', 'vllm:prefix_cache_queries_total'),
            ('prefix_cache_hits_metric', 'vllm:prefix_cache_hits_total'))
REQUIRED_COUNTERS = ('vllm:generation_tokens_total',)
KV_USAGE = ('vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc')
RUNNING, WAITING = 'vllm:num_requests_running', 'vllm:num_requests_waiting'
PREEMPTIONS = 'vllm:num_preemptions_total'
# The families kept in each batch sample (`_created` series dropped).
SAMPLE_FAMILIES = (RUNNING, WAITING) + KV_USAGE + ('vllm:generation_tokens_total', PREEMPTIONS,
                                                   'vllm:request_success_total', FINISHED) + TPOT_PER_TOKEN
# An inter-token latency above this is a decode step stalled by an admission (phase 1, Goal, selection):
# measured steps take 65-150 ms at batch sizes 1-12 (emulation job 3291778), phase 1 about 0.8-1 s.
# (Kept for the `steady` cells and `stall_intervals`, so old tables stay reproducible; a plain step is read from the
# `clean` cells below since 2026-11-10.)
STALL_SECONDS = 0.5
# 2026-11-10 (perf review FINDINGS 1.2): today's admissions stall decode for 0.2-0.4 s, below STALL_SECONDS, so the
# `steady` cells hold them (obs262 of job 3461703: steady ITL +35-55 % at b = 12-15). A stable interval at batch size b
# is `clean` when it holds no inter-token latency above clean_bound(b) = the smallest ITL-histogram bound >=
# CLEAN_FACTOR x the median of the mean ITL of b's stable intervals, and at most CLEAN_CAP_SECONDS. Relative to the
# batch's own step, so it holds across batch sizes and models: clean steps 20-28 ms on Qwen at b = 1-16 (job 3461703
# obs262) and 26-42 ms on Gemma (job 3462014 obs4k) give bounds of 0.05-0.075 s and 0.075-0.1 s; an admission stall
# (0.2-0.4 s) and a step that carries an admission's ~60-ms block are above them. The median stays a plain step while
# fewer than half of b's stable intervals hold a stall (obs262: at most 28 %); the cap keeps a stall out of a small cell
# whose median is not. Only the histogram's bounds can be tested (vLLM's buckets: ... 0.05, 0.075, 0.1, 0.15 ...).
CLEAN_FACTOR = 2.0
CLEAN_CAP_SECONDS = 0.15
SAMPLE_KEYS = ('running', 'waiting', 'kv_usage', 'generated', 'preemptions', 'finished', 'itl_sum', 'itl_count',
               'itl_slow', 'itl_buckets')
# The capacity gate's authenticated refusal (remote_ra_live_v2.capacity: 'CapacityWait: ' + CAPACITY_WAIT).
CAPACITY_WAIT = 'CapacityWait: CAPACITY_WAIT'
# goal_async (9b18421): the gate's refusal of an add while another admission is parked at its Goal.
PARKED_REFUSAL = CAPACITY_WAIT + ': another admission waits for its Goal'
# The engine thread's timed admission pieces (registered_add_timing.timed): a registered-add's dispatch and, for an
# admission parked at its Goal, its resume. `engine_decode` runs on the input thread and is not one of them.
ENGINE_PIECES = ('engine_dispatch', 'engine_resume')
# Goal decision fields of a query turn's native record (query.body.goal), kept per request.
GOAL_FIELDS = ('gate_decision', 'applied_decision', 'slm_attempted', 'slm_succeeded', 'fallback_reason')
TURN_CHECKS = ('runtime', 'io', 'http', 'token-stream', 'stage-b', 'selection')
WORKER_OPS = ('prepare', 'phase1', 'phase1_drain', 'reserve', 'allocation_update', 'complete')
# perf-ctx: up to the 131,072-token context limit of align-ctx.
PROMPT_BUCKETS = ((0, 16384, '<=16K'), (16384, 32768, '16-32K'), (32768, 49152, '32-48K'), (49152, 65536, '48-64K'),
                  (65536, 81920, '64-80K'), (81920, 98304, '80-96K'), (98304, 114688, '96-112K'),
                  (114688, 131072, '112-128K'))
RECEIPT_KEYS = ('index', 'session', 'turn', 'file', 'status', 'error', 'seconds', 'bytes', 'sha256')
MB = 1e6


def check_name(session, turn, kind):
    """Per-request check name of smoke_checks: s<k>-turn-<t>-<kind>."""
    return 's%d-turn-%d-%s' % (session, turn, kind)


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=1)
        stream.write('\n')


def save_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def finite(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


# ---------------------------------------------------------------- Prometheus text format (0.0.4)

def parse_labels(text):
    """`a="x",b="y\\"z"` (inside the braces) -> dict; unescapes \\\\, \\" and \\n."""
    labels, i, n = {}, 0, len(text)
    while i < n:
        while i < n and text[i] in ' ,':
            i += 1
        if i >= n:
            break
        equals = text.index('=', i)
        name = text[i:equals].strip()
        if not name or text[equals + 1] != '"':
            raise ValueError('malformed label set: ' + text)
        i, value = equals + 2, []
        while True:
            char = text[i]
            if char == '\\':
                following = text[i + 1]
                value.append({'n': '\n', '\\': '\\', '"': '"'}.get(following, '\\' + following)); i += 2
            elif char == '"':
                i += 1; break
            else:
                value.append(char); i += 1
        labels[name] = ''.join(value)
    return labels


def parse_sample(line):
    """`name{labels} value [timestamp]` -> (name, labels, float value)."""
    brace = line.find('{')
    blanks = [i for i in (line.find(' '), line.find('\t')) if i >= 0]
    space = min(blanks) if blanks else len(line)
    if 0 <= brace < space:
        i, quoted = brace + 1, False
        while True:
            char = line[i]
            if quoted:
                if char == '\\':
                    i += 2; continue
                quoted = char != '"'
            elif char == '"':
                quoted = True
            elif char == '}':
                break
            i += 1
        name, labels, rest = line[:brace], parse_labels(line[brace + 1:i]), line[i + 1:].split()
    else:
        name, labels, rest = line[:space], {}, line[space:].split()
    return name, labels, float(rest[0])


def parse(text):
    """-> (types {family: type}, samples [(name, labels, value)], malformed line count)."""
    types, samples, malformed = {}, [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            parts = line.split(None, 3)
            if len(parts) == 4 and parts[1] == 'TYPE':
                types[parts[2]] = parts[3].strip()
            continue
        try:
            samples.append(parse_sample(line))
        except (ValueError, IndexError):
            malformed += 1
    return types, samples, malformed


def family_of(name, types):
    if name in types:
        return name
    for suffix in ('_bucket', '_count', '_sum', '_total'):
        if name.endswith(suffix) and name[:-len(suffix)] in types:
            return name[:-len(suffix)]
    return name


def series_key(labels, drop=()):
    return ','.join('%s=%s' % (k, json.dumps(labels[k])) for k in sorted(labels) if k not in drop)


def snapshot(text):
    """{family: dict(type, series)} of the vllm:* families; `_created` timestamps are dropped."""
    types, samples, _ = parse(text)
    out = {}
    for name, labels, value in samples:
        if not name.startswith(PREFIX) or name.endswith('_created'):
            continue
        family = family_of(name, types)
        kind = types.get(family, 'untyped')
        series = out.setdefault(family, dict(type=kind, series={}))['series']
        if kind == 'histogram':
            entry = series.setdefault(series_key(labels, ('le',)), dict(sum=None, count=None, buckets={}))
            if name.endswith('_bucket'):
                entry['buckets'][labels.get('le', '')] = value
            elif name.endswith('_sum'):
                entry['sum'] = value
            elif name.endswith('_count'):
                entry['count'] = value
        else:
            series[series_key(labels)] = value
    return out


def minus(after, before):
    after, before = finite(after), finite(before if before is not None else 0.0)
    return None if after is None or before is None else after - before


def deltas(before_text, after_text):
    """Change between two scrapes of one server."""
    before, after = snapshot(before_text), snapshot(after_text)
    out = dict(counters={}, histograms={}, gauges={}, errors=[])
    for family in sorted(after):
        info, prior = after[family], before.get(family, dict(series={}))['series']
        if info['type'] == 'counter':
            out['counters'][family] = {k: minus(v, prior.get(k)) for k, v in sorted(info['series'].items())}
            values = out['counters'][family].values()
        elif info['type'] == 'histogram':
            series = {}
            for key, entry in sorted(info['series'].items()):
                old = prior.get(key) or dict(sum=0.0, count=0.0, buckets={})
                buckets = {le: minus(v, old['buckets'].get(le)) for le, v in entry['buckets'].items()}
                series[key] = dict(sum=minus(entry['sum'], old['sum']), count=minus(entry['count'], old['count']),
                                   buckets={le: d for le, d in sorted(buckets.items()) if d})
            out['histograms'][family] = series
            values = [s['count'] for s in series.values()] + [s['sum'] for s in series.values()]
        else:
            out['gauges'][family] = {k: dict(before=finite(prior.get(k)), after=finite(v))
                                     for k, v in sorted(info['series'].items())}
            continue
        if any(v is not None and v < 0 for v in values):
            out['errors'].append('negative delta: ' + family)
    return out


def histogram_total(delta, family):
    series = delta['histograms'].get(family)
    if series is None:
        return None
    return (sum(s['sum'] or 0.0 for s in series.values()), sum(s['count'] or 0.0 for s in series.values()))


def counter_total(delta, family):
    series = delta['counters'].get(family)
    return None if series is None else sum(v or 0.0 for v in series.values())


def finished_total(text):
    """Requests finished so far by one scrape: the e2e histogram's count, else the success counter; None without both."""
    snap = snapshot(text)
    family = snap.get(FINISHED)
    if family is not None and family['type'] == 'histogram':
        counts = [finite(s['count']) for s in family['series'].values()]
        if counts and None not in counts:
            return sum(counts)
    family = snap.get('vllm:request_success_total')
    if family is not None and family['type'] == 'counter':
        values = [finite(v) for v in family['series'].values()]
        if values and None not in values:
            return sum(values)
    return None


def settle_state(first_text, after_text, responses):
    """'settled' once a scrape counts every response so far as finished since the first chain scrape, 'pending' to
    retry, 'unavailable' if no finished-request signal can show it (vLLM records request stats in its output loop)."""
    if not first_text:
        return 'unavailable'
    if not after_text:
        return 'pending'
    after = finished_total(after_text)
    if after is None:
        return 'unavailable'
    first = finished_total(first_text) or 0.0  # a family not yet exported before any finish counts 0
    return 'settled' if after - first >= responses else 'pending'


def vllm_text(raw):
    """Only the vllm:* samples and their HELP/TYPE lines are archived."""
    text = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
    keep = [line for line in text.splitlines()
            if line.startswith(PREFIX) or line.startswith(('# HELP ' + PREFIX, '# TYPE ' + PREFIX))]
    return '\n'.join(keep) + '\n' if keep else ''


def sample_family(name):
    for suffix in ('_bucket', '_count', '_sum'):
        if name.endswith(suffix) and name[:-len(suffix)] in SAMPLE_FAMILIES:
            return name[:-len(suffix)]
    return name


def sample_text(raw):
    """A batch sample: the SAMPLE_FAMILIES lines of one scrape (HELP/TYPE and samples; `_created` series dropped)."""
    keep = []
    for line in vllm_text(raw).splitlines():
        if line.startswith('# '):
            parts = line.split(None, 3)
            name = parts[2] if len(parts) > 2 else ''
        else:
            name = line.split('{', 1)[0].split(None, 1)[0]
        if sample_family(name) in SAMPLE_FAMILIES:
            keep.append(line)
    return '\n'.join(keep) + '\n' if keep else ''


def http_get(url, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def metrics_record(session, turn, before, after, state, attempts, seconds, responses):
    """s<k>/metrics-<t>.json: the two chain scrapes around the response, the settle state and the window deltas."""
    errors = []
    if before is None:
        errors.append('before scrape: no chain scrape before the request')
    for phase, scrape in (('before', before), ('after', after)):
        if scrape is not None and scrape['error']:
            errors.append('%s scrape: %s' % (phase, scrape['error']))
    if state == 'pending':
        errors.append('%d responses but fewer finished requests in /metrics after %.1f s' % (responses, seconds))
    delta = None
    if before is not None and before['text'] and after['text']:
        delta = deltas(before['text'], after['text'])
        errors += delta['errors']
    else:
        errors.append('no vllm:* samples in the %s scrape: no window delta'
                      % ('before' if before is None or not before['text'] else 'after'))
    receipts = {phase: None if scrape is None else {k: scrape[k] for k in RECEIPT_KEYS}
                for phase, scrape in (('before', before), ('after', after))}
    return dict(session=session, turn=turn, path=METRICS_PATH, settle=state, attempts=attempts, settle_seconds=seconds,
                responses=responses, deltas=delta, errors=errors, **receipts)


class ChainProbe:
    """drive_sessions hooks: the /metrics chain (start, after every response) and the batch sampler.

    Chain: one scrape at the start and one after every response, serialized by
    a lock and outside the timed requests. After the k-th response the probe
    re-scrapes every 0.2 s for at most 3 s until /metrics counts k finished
    requests since the first scrape (vLLM records request stats in its output
    loop); under batched decode that scrape may already count later finishes.
    Sampler: from `start` until `finish`, a background thread scrapes every
    `sample_interval` seconds (and once more at the end) into
    metrics/sample-<i>.prom with its receipt line in metrics/samples.jsonl
    (`t_s`: seconds since the driver's epoch). It never raises into the driver
    (session._hook keeps its errors).
    """
    def __init__(self, base_url, output, *, get=http_get, clock=time.monotonic, sleep=time.sleep,
                 settle=SETTLE_SECONDS, interval=SETTLE_INTERVAL, sample_interval=SAMPLE_INTERVAL):
        self.url = base_url.rstrip('/') + METRICS_PATH
        self.output, self.get, self.clock, self.sleep = Path(output), get, clock, sleep
        self.settle, self.interval, self.sample_interval = settle, interval, sample_interval
        self.lock, self.last, self.first, self.count, self.responses = threading.Lock(), None, None, 0, 0
        self.samples, self.stopped, self.thread, self.epoch = 0, threading.Event(), None, None

    def scrape(self, select=vllm_text):
        started = self.clock()
        try:
            status, raw = self.get(self.url, SCRAPE_TIMEOUT)
            text, error = (select(raw), None) if status == 200 else ('', 'HTTP %s' % status)
        except Exception as exc:
            status, text, error = None, '', '%s: %s' % (type(exc).__name__, exc)
        return dict(status=status, error=error, seconds=round(self.clock() - started, 3), text=text)

    def keep(self, scrape, session=None, turn=None):
        """Archive one chain scrape: metrics/scrape-<i>.prom and its receipt line in metrics/chain.jsonl."""
        index, self.count = self.count, self.count + 1
        name = '%s/scrape-%d.prom' % (METRICS_DIR, index)
        raw = scrape['text'].encode()
        (self.output / name).write_bytes(raw)
        receipt = dict(index=index, session=session, turn=turn, file=name, status=scrape['status'], error=scrape['error'],
                       seconds=scrape['seconds'], bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        with (self.output / METRICS_DIR / 'chain.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(receipt, sort_keys=True, separators=(',', ':')) + '\n')
        self.last = dict(receipt, text=scrape['text'])
        return self.last

    def start(self, epoch=None):
        self.epoch = self.clock() if epoch is None else epoch
        with self.lock:
            (self.output / METRICS_DIR).mkdir(exist_ok=True)
            self.first = self.keep(self.scrape())
        if self.sample_interval:
            self.thread = threading.Thread(target=self._sample_loop, name='metrics-sampler', daemon=True)
            self.thread.start()

    def _sample_loop(self):
        while True:
            self.sample()
            if self.stopped.wait(self.sample_interval):
                break
        self.sample()  # once more after every session ended

    def sample(self):
        """One batch sample: metrics/sample-<i>.prom (SAMPLE_FAMILIES only) and its line in metrics/samples.jsonl."""
        started = self.clock()
        scrape = self.scrape(sample_text)
        index, self.samples = self.samples, self.samples + 1
        name = '%s/sample-%d.prom' % (METRICS_DIR, index)
        raw = scrape['text'].encode()
        (self.output / name).write_bytes(raw)
        receipt = dict(index=index, t_s=round(started - self.epoch, 3), file=name, status=scrape['status'],
                       error=scrape['error'], seconds=scrape['seconds'], bytes=len(raw),
                       sha256=hashlib.sha256(raw).hexdigest())
        with (self.output / METRICS_DIR / 'samples.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(receipt, sort_keys=True, separators=(',', ':')) + '\n')

    def after(self, session, turn, entry):
        with self.lock:
            self.responses += 1
            responses, before = self.responses, self.last
            first = self.first['text'] if self.first else ''
            started, attempts = self.clock(), 0
            while True:
                attempts += 1
                after = self.scrape()
                state = settle_state(first, after['text'], responses)
                if state != 'pending' or self.clock() - started >= self.settle:
                    break
                self.sleep(self.interval)
            seconds = round(self.clock() - started, 3)
            kept = self.keep(after, session, turn)
            save_json(self.output / ('s%d' % session) / ('metrics-%d.json' % turn),
                      metrics_record(session, turn, before, kept, state, attempts, seconds, responses))

    def finish(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=3 * SCRAPE_TIMEOUT)


def load_chain(out):
    """metrics/chain.jsonl, checked: consecutive indices, each scrape file matching its receipt."""
    out = Path(out)
    chain = load_jsonl(out / METRICS_DIR / 'chain.jsonl')
    for position, receipt in enumerate(chain):
        if receipt['index'] != position or receipt['file'] != '%s/scrape-%d.prom' % (METRICS_DIR, position):
            raise ValueError('metrics chain index %s at position %d' % (receipt['index'], position))
        raw = (out / receipt['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt['sha256'] or len(raw) != receipt['bytes']:
            raise ValueError('chain scrape %d differs from its receipt' % position)
        if (position == 0) != (receipt['session'] is None):
            raise ValueError('chain scrape %d: only the first scrape precedes every request' % position)
    return chain


def chain_texts(out, chain):
    return [(Path(out) / receipt['file']).read_text(encoding='utf-8') for receipt in chain]


def recheck_metrics(out, session, turn, completion_tokens, chain):
    """Offline: the archived chain scrapes reproduce s<k>/metrics-<t>.json (receipts, deltas, settle state)."""
    out = Path(out)
    value = json.loads((out / ('s%d' % session) / ('metrics-%d.json' % turn)).read_bytes())
    after, before = value['after'], value['before']
    if after is None or after['session'] != session or after['turn'] != turn or chain[after['index']] != after:
        raise ValueError('s%d metrics-%d after scrape is not its chain receipt' % (session, turn))
    if before is not None and (before['index'] != after['index'] - 1 or chain[before['index']] != before):
        raise ValueError('s%d metrics-%d before scrape is not the previous chain scrape' % (session, turn))
    if value['responses'] != after['index']:
        raise ValueError('s%d metrics-%d counts %s responses at chain scrape %d' % (session, turn, value['responses'],
                                                                                   after['index']))
    texts = {phase: (out / value[phase]['file']).read_text(encoding='utf-8') if value[phase] else ''
             for phase in ('before', 'after')}
    delta = deltas(texts['before'], texts['after']) if texts['before'] and texts['after'] else None
    if delta != value['deltas']:
        raise ValueError('s%d metrics-%d deltas differ from the archived scrapes' % (session, turn))
    first = (out / chain[0]['file']).read_text(encoding='utf-8')
    state = settle_state(first, texts['after'], after['index'])
    if state != value['settle']:
        raise ValueError('s%d metrics-%d settle state %r, archived scrapes give %r' % (session, turn, value['settle'], state))
    return value


def metric_fields(delta, completion_tokens, finished=None):
    """Per-request values of the window holding a request's finish -> (fields, errors, notes).

    `finished`: the requests that finished in the window (default: its e2e
    histogram count). Per-request histograms count only at a finish, so with
    exactly one finish whose generated-token histogram is this request's own
    tokens (`window_own`), they are this request's values. Otherwise TPOT is
    the window mean of the per-request TPOT histogram (`window_mean`) and the
    other per-request values are left out. Counters and the per-token TPOT
    advance per step: they are used only when no other request generated in
    the window (`window_exclusive`), rare under batched decode. A missing
    family is an error, never an exception; a shared window is only a note.
    """
    fields = {name: None for name, _ in PER_REQUEST}
    fields.update({name: None for name, _ in COUNTERS}, tpot_ms=None, tpot_source=None, kv_cache_usage_after=None,
                  window_exclusive=None, window_own=None, window_requests=None)
    if delta is None:
        return fields, [], []
    errors, notes = [], []
    total = histogram_total(delta, FINISHED)
    if finished is None and total is not None:
        finished = total[1]
    fields['window_requests'] = finished
    own_tokens = histogram_total(delta, GENERATION_PER_REQUEST)
    own = bool(finished == 1 and (own_tokens is None or (own_tokens[1] == 1 and own_tokens[0] == completion_tokens)))
    fields['window_own'] = own
    for name, family in PER_REQUEST:
        value = histogram_total(delta, family)
        if value is None:
            errors.append('missing ' + family)
        elif own and value[1] == 1:
            fields[name] = round(value[0], 9)
    if not own:
        notes.append('window holds %s finished requests%s: per-request values left out' % (
            finished, '' if finished != 1 else ' (generated tokens of another request)'))
    generated = counter_total(delta, 'vllm:generation_tokens_total')
    exclusive = bool(own and generated is not None and type(completion_tokens) is int and generated == completion_tokens)
    fields['window_exclusive'] = exclusive
    value = histogram_total(delta, TPOT_PER_REQUEST)
    if value is not None and value[1] >= 1:
        fields.update(tpot_ms=round(value[0] / value[1] * 1000, 6),
                      tpot_source=TPOT_PER_REQUEST if own else 'window_mean')
    elif exclusive:
        for family in TPOT_PER_TOKEN:
            value = histogram_total(delta, family)
            if value is not None and value[1] > 0:
                fields.update(tpot_ms=round(value[0] / value[1] * 1000, 6), tpot_source=family)
                break
    if fields['tpot_ms'] is None and fields['decode_s'] is not None and type(completion_tokens) is int \
            and completion_tokens > 1:
        fields.update(tpot_ms=round(fields['decode_s'] / (completion_tokens - 1) * 1000, 6),
                      tpot_source='decode_s/(gen_tokens-1)')
    for name, family in COUNTERS:
        value = counter_total(delta, family)
        if value is None and family in REQUIRED_COUNTERS:
            errors.append('missing ' + family)
        fields[name] = None if value is None or not exclusive else round(value, 6)
    if generated is not None and own and not exclusive:
        notes.append('counters shared with other running requests (generated-token delta %s, own %s)'
                     % (generated, completion_tokens))
    for family in KV_USAGE:
        series = delta['gauges'].get(family)
        if series:
            values = [s['after'] for s in series.values() if s['after'] is not None]
            fields['kv_cache_usage_after'] = round(max(values), 6) if values else None
            break
    return fields, errors, notes


def chain_finishes(texts):
    """Per chain scrape: requests finished since the first scrape (None where a scrape cannot show it)."""
    if not texts or not texts[0]:
        return [None] * len(texts)
    first = finished_total(texts[0]) or 0.0
    out = []
    for text in texts:
        value = finished_total(text) if text else None
        out.append(None if value is None else value - first)
    return out


def finish_window(finishes, index):
    """The chain window (j - 1, j) holding the `index`-th finish (cumulative count), or None."""
    for j in range(1, len(finishes)):
        if finishes[j] is not None and finishes[j] >= index:
            return j if finishes[j - 1] is not None else None
    return None


# ---------------------------------------------------------------- batch samples

def load_samples(out):
    """metrics/samples.jsonl, checked: consecutive indices, files matching receipts, nondecreasing times; [] if absent."""
    out = Path(out)
    path = out / METRICS_DIR / 'samples.jsonl'
    if not path.exists():
        return []
    samples = load_jsonl(path)
    for position, receipt in enumerate(samples):
        if receipt['index'] != position or receipt['file'] != '%s/sample-%d.prom' % (METRICS_DIR, position):
            raise ValueError('metrics sample index %s at position %d' % (receipt['index'], position))
        raw = (out / receipt['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt['sha256'] or len(raw) != receipt['bytes']:
            raise ValueError('metrics sample %d differs from its receipt' % position)
        if position and receipt['t_s'] < samples[position - 1]['t_s']:
            raise ValueError('metrics sample %d goes back in time' % position)
    return samples


def stall_bound(buckets):
    """The largest histogram bound <= STALL_SECONDS, as its label, or None."""
    bounds = []
    for le in buckets:
        try:
            value = float(le)
        except ValueError:
            continue
        if value <= STALL_SECONDS:
            bounds.append((value, le))
    return max(bounds)[1] if bounds else None


def sample_values(text):
    """Batch values of one sample (None where a family is absent)."""
    snap = snapshot(text)

    def plain(names, combine=sum):
        for name in names:
            family = snap.get(name)
            if family is not None and family['type'] != 'histogram':
                values = [finite(v) for v in family['series'].values()]
                if values and None not in values:
                    return combine(values)
        return None
    values = dict(running=plain((RUNNING,)), waiting=plain((WAITING,)), kv_usage=plain(KV_USAGE, max),
                  generated=plain(('vllm:generation_tokens_total',)), preemptions=plain((PREEMPTIONS,)),
                  finished=finished_total(text), itl_sum=None, itl_count=None, itl_slow=None, itl_buckets=None)
    for name in TPOT_PER_TOKEN:
        family = snap.get(name)
        if family is None or family['type'] != 'histogram' or not family['series']:
            continue
        series = list(family['series'].values())
        values.update(itl_sum=sum(s['sum'] or 0.0 for s in series), itl_count=sum(s['count'] or 0.0 for s in series))
        slow = 0.0
        for entry in series:
            bound = stall_bound(entry['buckets'])
            if bound is None or entry['count'] is None:
                slow = None
                break
            slow += entry['count'] - entry['buckets'][bound]
        values['itl_slow'] = slow
        values['itl_buckets'] = itl_buckets(series)
        break
    return values


def itl_buckets(series):
    """{finite bound label: cumulative count summed over the series} of the ITL histogram (the clean rule), or None
    when the series do not share their finite bounds or a count is missing."""
    labels = None
    for entry in series:
        finite_labels = set()
        for le, count in entry['buckets'].items():
            try:
                value = float(le)
            except ValueError:
                continue
            if math.isfinite(value):
                if finite(count) is None:
                    return None
                finite_labels.add(le)
        if labels is not None and finite_labels != labels:
            return None
        labels = finite_labels
    if not labels:
        return None
    return {le: sum(entry['buckets'][le] for entry in series) for le in sorted(labels, key=float)}


def batch_timeline(out):
    """Every batch sample of one server in time order: receipt time and its values (None where unreadable)."""
    out, rows = Path(out), []
    for receipt in load_samples(out):
        text = (out / receipt['file']).read_text(encoding='utf-8')
        values = sample_values(text) if receipt['status'] == 200 and text else None
        rows.append(dict(index=receipt['index'], t_s=receipt['t_s'], ok=values is not None,
                         **(values or dict.fromkeys(SAMPLE_KEYS))))
    return rows


def clean_bound(labels, reference_ms):
    """The clean rule's bound (a histogram bound label) for a batch size whose median interval ITL is `reference_ms`:
    the smallest bound >= CLEAN_FACTOR x the median, at most the largest bound <= CLEAN_CAP_SECONDS; None without one."""
    bounds = sorted((float(le), le) for le in labels)
    capped = [b for b in bounds if b[0] <= CLEAN_CAP_SECONDS]
    above = [b for b in capped if b[0] >= CLEAN_FACTOR * reference_ms / 1000]
    return (above[0] if above else capped[-1] if capped else (None, None))[1]


def median(values):
    values = sorted(values)
    n = len(values)
    return None if not n else values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def plain_step(clean, running):
    """The clean ITL (s) at `running` requests (>= 1): linear between the nearest batch sizes with a clean cell, the
    nearest one beyond them; None without any clean cell."""
    sizes = sorted(clean)
    if not sizes:
        return None
    r = max(1.0, running)
    lo = max([s for s in sizes if s <= r], default=sizes[0])
    hi = min([s for s in sizes if s >= r], default=sizes[-1])
    if lo == hi:
        return clean[lo] / 1000
    return (clean[lo] + (clean[hi] - clean[lo]) * (r - lo) / (hi - lo)) / 1000


def decode_lost(pairs, clean):
    """The decode time the running requests lost, from vLLM's ITL histogram (FINDINGS 1.1): over every interval of
    the sampler window, the ITL sum minus its observations x the clean step at the interval's batch size, as request-
    seconds and as a share of the ITL sum (each decode step adds one observation per running request, so the ITL sum
    is the requests' decode time). Point: each interval at the mean of its two running counts. Bracket (the review's):
    low = every observation at the largest clean step of any batch size; high = each interval at the faster of its two
    end counts' steps. Not clamped per interval: a clean interval adds exactly zero at its own b. None without a clean
    cell or without ITL samples."""
    pairs = [(a, b, ds, dn) for a, b, ds, dn in pairs if dn and dn > 0]
    if not clean or not pairs:
        return None
    S, N = sum(p[2] for p in pairs), sum(p[3] for p in pairs)
    if S <= 0:
        return None
    point = sum(dn * plain_step(clean, (a + b) / 2) for a, b, _, dn in pairs)
    fast = sum(dn * min(plain_step(clean, a), plain_step(clean, b)) for a, b, _, dn in pairs)
    slowest = max(clean.values()) / 1000
    lost, high, low = S - point, S - fast, S - N * slowest
    return dict(itl_sum_s=round(S, 3), observations=round(N), mean_itl_ms=round(S / N * 1000, 3),
                lost_s=round(lost, 3), lost_low_s=round(low, 3), lost_high_s=round(high, 3),
                lost_share=round(lost / S, 4), lost_share_low=round(low / S, 4), lost_share_high=round(high / S, 4),
                largest_clean_itl_ms=round(slowest * 1000, 3), clean_sizes=sorted(clean))


def batch_profile(timeline):
    """The decode batch over time and, per batch size b, aggregate decode tokens/s and inter-token latency.

    An interval joins two consecutive readable samples. It is `stable` at b
    when both samples count b running requests (b >= 1), and `steady` when in
    addition none of its inter-token latencies exceeds STALL_SECONDS (kept as
    it was: today's 0.2-0.4-s admissions stay in it), and `clean` (2026-11-10)
    when none exceeds clean_bound(b) (CLEAN_FACTOR x the median interval ITL at
    b; see CLEAN_FACTOR): the cells a "plain decode step" is read from.
    Tokens/s: the generated-token counter's delta over the interval; ITL: the
    inter-token latency histogram's mean over the interval (each decode step
    adds one observation per running request, so this is the TPOT a request
    sees at batch size b). `decode_lost`: the decode time lost to everything
    that is not a plain step (decode_lost()).
    """
    points = [p for p in timeline if p['ok']]
    out = dict(samples=len(timeline), readable=len(points), span_s=None, running_peak=None, running_mean=None,
               waiting_peak=None, kv_usage_peak=None, preemptions_max=None, stall_intervals=0, batches=[],
               clean_rule=dict(factor=CLEAN_FACTOR, cap_s=CLEAN_CAP_SECONDS), decode_lost=None)
    if not points:
        return out
    present = lambda key: [p[key] for p in points if p[key] is not None]
    for key, name, combine in (('running', 'running_peak', max), ('waiting', 'waiting_peak', max),
                               ('kv_usage', 'kv_usage_peak', max), ('preemptions', 'preemptions_max', max)):
        values = present(key)
        out[name] = combine(values) if values else None
    groups, weighted, total, pairs = {}, 0.0, 0.0, []
    for a, b in zip(points, points[1:]):
        dt = b['t_s'] - a['t_s']
        if dt <= 0 or a['running'] is None or b['running'] is None:
            continue
        weighted += (a['running'] + b['running']) / 2 * dt
        total += dt
        slow = minus(b['itl_slow'], a['itl_slow']) if a['itl_slow'] is not None and b['itl_slow'] is not None else None
        if slow:
            out['stall_intervals'] += 1
        itl = (minus(b['itl_sum'], a['itl_sum']), minus(b['itl_count'], a['itl_count']))
        if None not in itl:
            pairs.append((a['running'], b['running'], itl[0], itl[1]))
        if a['running'] != b['running'] or a['running'] < 1:
            continue
        group = groups.setdefault(int(a['running']), dict(stable_s=0.0, stable_tokens=0.0, intervals=0, steady_s=0.0,
                                                          steady_tokens=0.0, steady_intervals=0, itl_sum=0.0,
                                                          itl_count=0.0, candidates=[]))
        tokens = minus(b['generated'], a['generated'])
        group['intervals'] += 1
        group['stable_s'] += dt
        group['stable_tokens'] += tokens or 0.0
        if slow == 0 and itl[1]:
            group['steady_intervals'] += 1
            group['steady_s'] += dt
            group['steady_tokens'] += tokens or 0.0
            group['itl_sum'] += itl[0]
            group['itl_count'] += itl[1]
        # the clean rule's candidates: every stable interval with ITL observations and the bucket counts at both ends
        ba, bb = a.get('itl_buckets'), b.get('itl_buckets')
        if itl[0] is not None and itl[1] and ba and bb and sorted(ba) == sorted(bb):
            group['candidates'].append(dict(dt=dt, tokens=tokens or 0.0, ds=itl[0], dn=itl[1],
                                            below={le: bb[le] - ba[le] for le in bb}))
    out.update(span_s=round(points[-1]['t_s'] - points[0]['t_s'], 3),
               running_mean=round(weighted / total, 3) if total > 0 else None)
    clean = {}
    for size in sorted(groups):
        g = groups[size]
        cells = g['candidates']
        reference = median([1000 * c['ds'] / c['dn'] for c in cells])
        bound = clean_bound(cells[0]['below'], reference) if cells else None
        kept = [c for c in cells if bound is not None and c['dn'] - c['below'][bound] <= 0]
        clean_s, clean_n = sum(c['dt'] for c in kept), sum(c['dn'] for c in kept)
        clean_itl = round(sum(c['ds'] for c in kept) / clean_n * 1000, 3) if clean_n else None
        if clean_itl is not None:
            clean[size] = clean_itl
        out['batches'].append(dict(
            batch=size, intervals=g['intervals'], stable_s=round(g['stable_s'], 3),
            stable_tok_s=round(g['stable_tokens'] / g['stable_s'], 3) if g['stable_s'] > 0 else None,
            steady_intervals=g['steady_intervals'], steady_s=round(g['steady_s'], 3),
            steady_tok_s=round(g['steady_tokens'] / g['steady_s'], 3) if g['steady_s'] > 0 else None,
            steady_itl_ms=round(g['itl_sum'] / g['itl_count'] * 1000, 3) if g['itl_count'] else None,
            median_interval_itl_ms=round(reference, 3) if reference is not None else None,
            clean_bound_s=float(bound) if bound is not None else None, clean_intervals=len(kept),
            clean_s=round(clean_s, 3),
            clean_tok_s=round(sum(c['tokens'] for c in kept) / clean_s, 3) if clean_s > 0 else None,
            clean_itl_ms=clean_itl))
    out['decode_lost'] = decode_lost(pairs, clean)
    return out


def running_between(timeline, start, end):
    """Mean of the sampled running-request count over [start, end] (seconds since the driver's epoch), or None."""
    values = [p['running'] for p in timeline if p['ok'] and p['running'] is not None and start <= p['t_s'] <= end]
    return round(sum(values) / len(values), 3) if values else None


def generated_at(timeline, t):
    """vLLM's generated-token counter at time t, linear between the two samples around it; None outside the samples."""
    points = [(p['t_s'], p['generated']) for p in timeline if p['ok'] and p['generated'] is not None]
    for (a, x), (b, y) in zip(points, points[1:]):
        if a <= t <= b:
            return x if b == a else x + (y - x) * (t - a) / (b - a)
    return None


# ---------------------------------------------------------------- server log: admissions

def admission_events(text):
    """REMOTE_RA_REGISTERED_ADD JSON bodies in log order, and the count of unreadable ones."""
    events, unreadable = [], 0
    for line in text.splitlines():
        if 'REMOTE_RA_REGISTERED_ADD ' not in line:
            continue
        try:
            value = json.loads(line.split('REMOTE_RA_REGISTERED_ADD ', 1)[1])
        except ValueError:
            unreadable += 1
            continue
        if isinstance(value, dict):
            events.append(value)
    return events, unreadable


# A completion's Cancel (and a session close) waits 60 s for its ACK (registered_add_protocol); an admission ADD waits
# ADMISSION_WAIT_SECONDS or more (goal_async; 60 s only before it, when no admission parked).
COMMAND_WAIT_SECONDS = 60


def engine_pieces(events, several=False, admission_timeouts=()):
    """The engine thread's registered-add pieces of one server log, and where each command was acknowledged.

    The engine thread writes, per registered-add, `engine_dispatch_start/_end` (call_id), and right after it
    `engine_ack_queued` of that call; on a goal_async server an admission parked at its Goal ends its dispatch
    without an ack, and a later `engine_resume_start/_end` (no call_id) holds its one `engine_ack_queued`
    (core._remote_ra_resume -> GoalAdmissions.resume -> _remote_ra_ack). A parked admission cancelled while
    parked is acknowledged inside the Cancel's dispatch (`aborted`). One thread writes these lines, so the
    pieces never overlap: a piece starting inside another (`overlaps`) means admissions ran concurrently on the
    engine. `errors` collects acknowledgement faults: a command acknowledged twice or never, an ack outside its
    place, a resume without exactly one parked call, two admissions parked at once. Returns dict(pieces, calls,
    parked, overlaps, errors, unattributed); `parked` maps each parked call (in park order) to its dispatch, resume or
    abort, and ack position.

    `several` (adm-f's (f), --admission-overlap K > 1): up to K admissions are in flight and may be parked at once (at
    their Goals, an in-step phase 1, a lease wait); a resume continues the oldest done one, so a resume that
    acknowledges nothing (the admission parks again) while several are parked belongs to one of them that the log does
    not name: such pieces go to `unattributed` (engine time not credited to a call), not to `errors`.

    `held` (--deferred-completion-io, completion-offload ad66aca / 120bc3c): a completion's Cancel (a 60-s command, the
    frontend's `frontend_command_sent` timeout; never an admission ADD, whose wait is one of `admission_timeouts`)
    whose request's files are still being written is acknowledged later, once the writer wrote them
    (dispatcher.handle_steps returns Held; lifecycle.tick sends it on the engine loop): its dispatch ends without an ack
    and its one `engine_ack_queued` follows outside its dispatch (job 3428631: after another admission's resume
    piece). Such a call is `held`, never parked: {call: dict(dispatch, ack_pos)}; the check decides whether the server
    may hold acks (check_goal_async: only with --deferred-completion-io). A held call never acknowledged is an
    error ("never acknowledged")."""
    thread = [(p, e) for p, e in enumerate(events) if str(e.get('phase', '')).startswith('engine_')
              and not str(e.get('phase', '')).startswith('engine_decode')]
    pieces, calls, parked, overlaps, errors, acks, unattributed = [], {}, {}, [], [], {}, []
    held = {}
    waits = {}
    for event in events:  # the frontend's ACK wait per command: a 60-s command is a Cancel / close, not an ADD
        if event.get('phase') == 'frontend_command_sent' and event.get('call_id') is not None:
            waits.setdefault(event['call_id'], event.get('timeout_seconds'))
    command = lambda call: (waits.get(call) == COMMAND_WAIT_SECONDS
                            and COMMAND_WAIT_SECONDS not in set(admission_timeouts or ()))
    current = None
    for i, (position, event) in enumerate(thread):
        phase, call = event.get('phase'), event.get('call_id')
        if phase in ('engine_dispatch_start', 'engine_resume_start'):
            if current is not None:
                overlaps.append('%s at event %d inside the %s piece started at event %d'
                                % (phase, position, current['kind'], current['start_pos']))
            current = dict(kind=phase[:-len('_start')], call_id=call if phase == 'engine_dispatch_start' else None,
                           start_pos=position, end_pos=None, ms=None, outcome=None, acks=[], rejected=[], ack=None)
        elif phase in ('engine_dispatch_end', 'engine_resume_end'):
            kind = phase[:-len('_end')]
            if current is None or current['kind'] != kind or (kind == 'engine_dispatch' and current['call_id'] != call):
                overlaps.append('%s at event %d does not end the open piece (%s)' % (
                    phase, position, 'none' if current is None else '%s from event %d' % (current['kind'],
                                                                                          current['start_pos'])))
                current = None
                continue
            current.update(end_pos=position, ms=event.get('elapsed_ms'), outcome=event.get('outcome'))
            pieces.append(current)
            for at, acked in current['acks']:  # the parked admissions this piece resumed (or aborted)
                entry = parked.get(acked)
                if entry is None or entry['resume'] or entry['abort']:
                    errors.append('%s at event %d acknowledges call %s, which was not parked before it'
                                  % (kind, current['start_pos'], acked))
                elif kind == 'engine_resume':
                    entry.update(resume=current, ack_pos=at)
                else:
                    entry.update(abort=current, ack_pos=at)
            if kind == 'engine_resume' and not current['acks']:
                # perf-adm: a goal_query admission resumed after its Goal parks again at its in-step phase 1; one
                # admission is in flight at a time, so the resume belongs to the one parked call.
                waiting = [v for v in parked.values() if v['resume'] is None and v['abort'] is None]
                if len(waiting) == 1:
                    waiting[0].setdefault('reparks', []).append(current)
                elif several and waiting:
                    unattributed.append(current)
                else:
                    errors.append('resume at event %d acknowledges nothing while %d admissions are parked'
                                  % (current['start_pos'], len(waiting)))
            elif kind == 'engine_resume' and len(current['acks']) != 1:
                errors.append('resume at event %d acknowledges %d commands, not one parked admission'
                              % (current['start_pos'], len(current['acks'])))
            if kind == 'engine_dispatch':
                calls.setdefault(call, []).append(current)
                following = thread[i + 1][1] if i + 1 < len(thread) else {}
                if following.get('phase') == 'engine_ack_queued' and following.get('call_id') == call:
                    current['ack'] = 'immediate'
                elif command(call):
                    # --deferred-completion-io: the Cancel's ack is held until its request's files are written
                    current['ack'] = 'held'
                    if call in held:
                        errors.append('call %s held twice' % call)
                    else:
                        held[call] = dict(dispatch=current, ack_pos=None)
                else:
                    current['ack'] = 'parked'
                    if not several and any(v['resume'] is None and v['abort'] is None for v in parked.values()):
                        errors.append('call %s parked at event %d while another admission was still parked'
                                      % (call, position))
                    if call in parked:
                        errors.append('call %s parked twice' % call)
                    else:
                        parked[call] = dict(dispatch=current, resume=None, abort=None, ack_pos=None)
            current = None
        elif phase == 'engine_ack_queued':
            acks.setdefault(call, []).append(position)
            if call in held and held[call]['ack_pos'] is None:
                held[call]['ack_pos'] = position  # the held Cancel's ack, sent by the engine loop (not a resume's)
                continue
            if current is not None:
                current['acks'].append((position, call))
                continue
            previous = thread[i - 1][1] if i else {}
            if not (previous.get('phase') == 'engine_dispatch_end' and previous.get('call_id') == call):
                errors.append('ack of call %s at event %d is neither right after its dispatch nor inside a piece'
                              % (call, position))
        elif phase == 'engine_rejected':
            if current is None:
                errors.append('engine rejection at event %d outside any engine piece' % position)
            else:
                current['rejected'].append(event)
    if current is not None:
        overlaps.append('the %s piece started at event %d never ended' % (current['kind'], current['start_pos']))
    for call, positions in acks.items():
        if len(positions) != 1:
            errors.append('call %s acknowledged %d times' % (call, len(positions)))
    for call, values in calls.items():
        if len(values) != 1:
            errors.append('call %s dispatched %d times' % (call, len(values)))
        if call not in acks:
            errors.append('call %s never acknowledged' % call)
    for call, entry in parked.items():
        if entry['resume'] is None and entry['abort'] is None:
            errors.append('parked call %s was never resumed' % call)
    return dict(pieces=pieces, calls=calls, parked=parked, overlaps=overlaps, errors=errors, unattributed=unattributed,
                held=held)


# The frontend's wait for an admission's ACK (registered_add_protocol.ADMISSION_SECONDS) and, with (f)
# --admission-overlap K, the pinned 300 + 120 (K - 1) s (admission_flags.timeouts): the admission block's and its ADD's
# `timeout_seconds`; a completion's Cancel waits 60 s.
ADMISSION_WAIT_SECONDS, REWRITE_WAIT_SECONDS = 300, 120


def log_permits(events):
    """K of a server log: from the frontend_admission_start timeouts (all at most 300 s: 1; 300 + 120 (K - 1): K). Returns
    (K or None when they disagree or fit no K, the sorted distinct timeouts)."""
    timeouts = sorted({e.get('timeout_seconds') for e in events if e.get('phase') == 'frontend_admission_start'}
                      - {None})
    if all(type(t) in (int, float) and t <= ADMISSION_WAIT_SECONDS for t in timeouts):
        return 1, timeouts  # one admission in flight (300 s since goal_async, 60 s before)
    if len(timeouts) != 1 or type(timeouts[0]) is not int or timeouts[0] < ADMISSION_WAIT_SECONDS \
            or (timeouts[0] - ADMISSION_WAIT_SECONDS) % REWRITE_WAIT_SECONDS:
        return None, timeouts
    return 1 + (timeouts[0] - ADMISSION_WAIT_SECONDS) // REWRITE_WAIT_SECONDS, timeouts


def admissions(text):
    """{request_sha256: [admission]} of one stage-2 server log, with event positions.

    frontend_admission_start/end (APIServer, registered_add_bridge) carry
    sha256(request_id) and bound one admission. Stage 2 serializes admissions
    (the lock passes on at engine acceptance), but the completions' commands
    (Cancel, sent as registered-add frames with the same 60-s timeout)
    interleave with them. The admission's ADD is the command that frontend pid
    sent inside the block whose ack also arrived inside it; when several
    commands qualify it stays unnamed (`add_ambiguous`) and its engine time is
    left out. An engine_rejected event of that request inside the block whose
    error is the capacity gate's marks a CAPACITY_WAIT refusal
    (`capacity_wait`); any other rejection is kept in `rejected`.

    goal_async: an admission parked at its Goal keeps its block (the frontend's
    lock passes on at the ack, after the resume), and the completions' Cancels
    sent and acknowledged during the park fall inside it; its ADD is then the
    one candidate whose dispatch parked. `parked` (its ack came in a resume
    piece), `dispatch_ms` (the dispatch piece: up to the park), `resume_ms`, and
    `engine_ms` (the engine-thread time: dispatch plus resume) come from
    `engine_pieces`, which the result keeps under `engine`.
    """
    events, unreadable = admission_events(text)
    permits, timeouts = log_permits(events)
    engine = engine_pieces(events, several=(permits or 1) > 1, admission_timeouts=timeouts)
    dispatch, sent, acked, adds = {}, {}, {}, {}
    for position, event in enumerate(events):
        phase, call = event.get('phase'), event.get('call_id')
        if call is None:
            continue
        if phase == 'engine_dispatch_end':
            dispatch.setdefault(call, event)
        elif phase == 'frontend_command_sent':
            if call not in sent and event.get('timeout_seconds') in timeouts:
                adds.setdefault(event.get('pid'), []).append(call)  # an ADD (its timeout is the admission wait)
            sent.setdefault(call, (position, event.get('pid'), event.get('timeout_seconds')))
        elif phase == 'frontend_ack_received':
            acked.setdefault(call, position)
    found, current, starts = {}, {}, {}
    for position, event in enumerate(events):
        phase, key = event.get('phase'), event.get('request_sha256')
        if phase == 'frontend_admission_start':
            starts.setdefault(event.get('pid'), []).append(position)
            value = dict(pid=event.get('pid'), timeout=event.get('timeout_seconds'), start_pos=position, end_pos=None,
                         frontend_ms=None, outcome=None, call_id=None, ack_pos=None, add_ambiguous=False,
                         engine_ms=None, engine_outcome=None, capacity_wait=False, rejected=None, parked=None,
                         dispatch_ms=None, resume_ms=None, parked_refusal=False, reparks=0)
            found.setdefault(key, []).append(value)
            current[key] = value
        elif phase == 'frontend_admission_end' and key in current:
            current.pop(key).update(frontend_ms=event.get('elapsed_ms'), outcome=event.get('outcome'), end_pos=position)
    rejections = [(p, e) for p, e in enumerate(events) if e.get('phase') == 'engine_rejected'
                  and e.get('kind') == 'RegisteredAdd']
    def candidates_of(value):
        start, end = value['start_pos'], value['end_pos']
        candidates = [call for call, (position, pid, _) in sent.items()
                      if start < position < end and pid == value['pid'] and start < acked.get(call, -1) < end]
        own = [call for call in candidates if sent[call][2] == value['timeout']]
        if len(candidates) > 1 and own:
            candidates = own  # the completions' Cancels (60 s) are not the ADD (the admission wait)
        parked = [call for call in candidates if call in engine['parked']]
        if len(candidates) > 1 and len(parked) == 1:
            candidates = parked  # the completions' Cancels acknowledged during the park are not the ADD
        return candidates

    blocks = [value for values in found.values() for value in values if value['end_pos'] is not None]
    chosen = {id(value): candidates_of(value) for value in blocks}
    if (permits or 1) > 1:
        # (f): up to K admissions in flight, so their blocks overlap and one block may hold several ADDs. Each ADD
        # belongs to exactly one block: a block left with one ADD that no other block has taken gets it, until
        # nothing changes; a block still left with several takes the ADD of its FIFO rank (the frontend sends each
        # ADD after its own block opened, permits in arrival order) when no other block took it; else it stays
        # unnamed (`add_ambiguous`).
        taken, changed = {}, True
        while changed:
            changed = False
            for value in blocks:
                if id(value) in taken:
                    continue
                live = [c for c in chosen[id(value)] if c not in taken.values()]
                if len(live) == 1:
                    taken[id(value)] = live[0]
                    changed = True
        for value in sorted(blocks, key=lambda v: v['start_pos']):
            live = [c for c in chosen[id(value)] if c not in taken.values()]
            if id(value) not in taken and len(live) > 1:
                rank = starts.get(value['pid'], []).index(value['start_pos'])
                fifo = (adds.get(value['pid']) or [])[rank:rank + 1]
                if fifo and fifo[0] in live:
                    taken[id(value)] = fifo[0]
        unnamed = {key for key, calls in chosen.items() if key not in taken and calls}
        chosen = {key: [taken[key]] if key in taken else calls for key, calls in chosen.items()}
    else:
        unnamed = {key for key, calls in chosen.items() if len(calls) > 1}
    for key, values in found.items():
        for value in values:
            start, end = value['start_pos'], value['end_pos']
            if end is None:
                continue
            candidates = chosen[id(value)]
            if id(value) in unnamed:
                value.update(add_ambiguous=True, ack_pos=min(acked[call] for call in candidates))
            elif len(candidates) == 1:
                call = candidates[0]
                first = dispatch.get(call) or {}
                entry = engine['parked'].get(call)
                resume = (entry or {}).get('resume')
                value.update(call_id=call, ack_pos=acked[call], engine_ms=first.get('elapsed_ms'),
                             engine_outcome=first.get('outcome'), parked=entry is not None,
                             dispatch_ms=first.get('elapsed_ms'))
                if entry is not None:
                    value['resume_ms'] = resume['ms'] if resume else None
                    pieces = (value['dispatch_ms'], value['resume_ms']) + tuple(r['ms'] for r in entry.get('reparks') or ())
                    value['engine_ms'] = round(sum(pieces), 3) if None not in pieces else None
                    value['reparks'] = len(entry.get('reparks') or ())
            elif candidates:
                value.update(add_ambiguous=True, ack_pos=min(acked[call] for call in candidates))
            for position, event in rejections:
                if start < position < end and event.get('request_sha256') == key:
                    error = event.get('error') or ''
                    if error.startswith(CAPACITY_WAIT):
                        value['capacity_wait'] = True
                        value['parked_refusal'] = error.startswith(PARKED_REFUSAL)
                    else:
                        value['rejected'] = error
    return dict(requests=found, unreadable=unreadable, events=len(events), engine=engine, permits=permits,
                admission_timeouts=timeouts)


def admission_blocks(found):
    """[(start_pos, end_pos, request_sha256)] of every admission in log order; an open block ends at its start."""
    blocks = []
    for key, values in found['requests'].items():
        for value in values:
            blocks.append((value['start_pos'], value['end_pos'] if value['end_pos'] is not None else value['start_pos'],
                           key))
    return sorted(blocks)


def accepted_admissions(values):
    """The admissions of one request that the engine did not refuse at the capacity gate."""
    return [v for v in values if not v['capacity_wait']]


# ---------------------------------------------------------------- receipts

def receipt_fields(row, native):
    """Phase-1 time, worker-operation times, the request's own H2D bytes, active KV, pages and engine stall."""
    frames = native['worker_frames']
    ops = {}
    for frame in frames:
        entry = ops.setdefault(frame['operation'], dict(ms=0.0, count=0))
        entry['ms'] += frame['http_measurement']['elapsed_ns'] / 1e6
        entry['count'] += 1
    for entry in ops.values():
        entry['ms'] = round(entry['ms'], 3)
    issued = lambda frame, side: frame['http_measurement'][side]['h2d_issued']
    total = issued(frames[-1], 'after') - issued(frames[0], 'before')
    phase1 = sum(issued(f, 'after') - issued(f, 'before') for f in frames if f['operation'] == 'phase1')
    stage = row['stage_b']
    first = stage['phase1']
    span = native.get('observation_span')
    stall = finite(native.get('admission_engine_ms'))
    goal = (((native.get('query') or {}).get('body') or {}).get('goal')) or {}
    timing = {k: None if finite(native.get(k)) is None else round(native[k], 3) for k in ('goal_async_ms', 'goal_construct_ms')}
    # perf-adm: how phase 1 ran (--phase1-in-step) and the device-to-device GDN bytes (--gdn-device-handoff).
    step = finite(native.get('phase1_step_ns'))
    timing.update(phase1_mode=native.get('phase1_mode'), phase1_parked='phase1_park_ms' in native,
                  phase1_park_ms=None if finite(native.get('phase1_park_ms')) is None else round(native['phase1_park_ms'], 3),
                  phase1_step_ms=None if step is None else round(step / 1e6, 3),
                  phase1_step_rows=native.get('phase1_step_rows') if type(native.get('phase1_step_rows')) is int else None,
                  d2d_bytes=(row.get('restore') or {}).get('d2d_bytes'),
                  # (f) --admission-overlap: parked waiting for the one phase-1 lease (not in admission_engine_ms)
                  lease_wait_ms=None if finite(native.get('lease_wait_ms')) is None else round(native['lease_wait_ms'], 3),
                  lease_waited='lease_wait_ms' in native)
    # goal-rollback-1116: a rollback whose Goal carried (only where the record has a carry: else as before)
    carry = {'goal_carried': goal['carry'].get('carried') is True} if isinstance(goal.get('carry'), dict) else {}
    return dict(timing, **carry, **{k: goal.get(k) for k in GOAL_FIELDS},
                prompt_tokens=len(native['prompt_token_ids']), observation_tokens=span[1] - span[0] if span else 0,
                phase1_tokens=first['forward_tokens'] if first else 0,
                phase1_ms=round(first['phase1_ns'] / 1e6, 3) if first else None,
                goal_forward=bool(first and first.get('goal_forward')),
                worker_op_ms=ops, h2d_phase1_bytes=phase1, h2d_other_bytes=total - phase1, h2d_total_bytes=total,
                active_kv_bytes=row['active_kv_bytes'],
                pages=dict(candidates=row['candidate_page_count'], active=len(native['plan']['selected']),
                           retained=len(stage['retain']), restored=len(stage['restore']),
                           released=len(stage['release']), run=len(stage['run_pages']),
                           # 2026-11-12 (--release-idle-pages only): the Pages this turn readmitted from the arena
                           **({'readmitted': len((row.get('idle_release') or {}).get('readmitted_pages') or ())}
                              if isinstance(row.get('idle_release'), dict) else {})),
                server_latency_s=round(row['request_latency_ns'] / 1e9, 3),
                admission_engine_ms=None if stall is None else round(stall, 3))


RECORD_KEYS = ('session_id', 'request_index', 'prompt_tokens', 'observation_tokens', 'phase1_tokens', 'phase1_ms',
               'goal_forward', 'worker_op_ms', 'h2d_phase1_bytes', 'h2d_other_bytes', 'h2d_total_bytes', 'active_kv_bytes',
               'pages', 'server_latency_s', 'queue_wait_s', 'capacity_waits', 'gen_tokens', 'client_e2e_s', 't_send_s',
               't_recv_s', 'service_s', 'admission_ms', 'engine_admission_ms', 'admission_engine_ms', 'selection_ms',
               'phase1_ms_per_1k', 'batch_mean', 'concurrency', 'metrics_note', 'goal_async_ms', 'goal_construct_ms',
               'parked', 'park_dispatch_ms', 'resume_ms', 'park_start_s', 'park_end_s', 'parked_running_mean',
               'phase1_mode', 'phase1_parked', 'phase1_park_ms', 'phase1_step_ms', 'phase1_step_rows', 'd2d_bytes',
               'reparks', 'lease_wait_ms', 'lease_waited') + GOAL_FIELDS


def consistency(record, row):
    """Cross-source warnings (never failures): the sources should agree where they overlap."""
    warnings = []
    gen, metric = record.get('gen_tokens'), record.get('generation_tokens_metric')
    if metric is not None and gen is not None and metric != gen:
        warnings.append('vllm:generation_tokens_total delta %s != usage.completion_tokens %s' % (metric, gen))
    client = record.get('client_e2e_s')
    if client is not None and record.get('e2e_s') is not None and record['e2e_s'] > client + 0.5:
        warnings.append('vLLM e2e %.3f s exceeds the client latency %.3f s' % (record['e2e_s'], client))
    if client is not None and record.get('admission_ms') is not None and record['admission_ms'] > client * 1000 + 500:
        warnings.append('admission %.1f ms exceeds the client latency %.3f s' % (record['admission_ms'], client))
    stall, admission = record.get('admission_engine_ms'), record.get('admission_ms')
    if stall is not None and admission is not None and stall > admission + 500:
        warnings.append('engine admission %.1f ms exceeds the frontend admission %.1f ms' % (stall, admission))
    try:
        if record.get('h2d_phase1_bytes') is not None and record['h2d_phase1_bytes'] != row['prepare']['h2d_bytes']:
            warnings.append('phase-1 frame H2D %s != runtime prepare.h2d_bytes %s'
                            % (record['h2d_phase1_bytes'], row['prepare']['h2d_bytes']))
        if record.get('h2d_other_bytes') is not None and record['h2d_other_bytes'] != row['restore']['h2d_bytes']:
            warnings.append('other H2D %s != runtime restore.h2d_bytes %s'
                            % (record['h2d_other_bytes'], row['restore']['h2d_bytes']))
    except (KeyError, TypeError):
        warnings.append('runtime row lacks prepare/restore H2D bytes')
    return warnings


class ChainView:
    """The /metrics chain of one server for the measurement records (scrape texts and cumulative finishes)."""
    def __init__(self, out):
        self.error = None
        try:
            self.chain = load_chain(out)
            self.texts = chain_texts(out, self.chain)
        except Exception as exc:  # absent or unreadable chain: every request records the gap
            self.chain, self.texts = [], []
            self.error = '%s: %s' % (type(exc).__name__, exc)
        self.finishes = chain_finishes(self.texts)

    def window(self, after_index):
        """(j, delta, finished) of the window holding the after_index-th finish, or None."""
        if after_index is None:
            return None
        j = finish_window(self.finishes, after_index)
        if j is None or not self.texts[j - 1] or not self.texts[j]:
            return None
        return j, deltas(self.texts[j - 1], self.texts[j]), self.finishes[j] - self.finishes[j - 1]


def turn_measurement(out, session, turn, position, row, client, statuses, found, log_error, policy, label, chain,
                     timeline):
    """One request: session k's turn t, joined to the runtime row at `position` of runtime.jsonl."""
    failed = [name for name in ['ready'] + [check_name(session, turn, k) for k in TURN_CHECKS]
              if statuses.get(name) != 'PASS']
    record = dict({k: None for k in RECORD_KEYS}, policy=policy, label=label, warmup=label == WARMUP_LABEL, session=session,
                  turn=turn, valid=not failed, failed_checks=failed, errors=[], admission_error=None)
    request_id = row.get('request_id') if isinstance(row, dict) else None
    record.update(session_id=row.get('session_id') if isinstance(row, dict) else None,
                  request_index=row.get('request_index') if isinstance(row, dict) else None)
    try:
        key = hashlib.sha256(request_id.encode()).hexdigest()
        native = json.loads((out / ('native-' + key + '.json')).read_bytes())
        record.update(receipt_fields(row, native))
    except Exception as exc:
        key = None
        record['errors'].append('receipt: %s: %s' % (type(exc).__name__, exc))
    if isinstance(row, dict) and finite(row.get('queue_wait_s')) is not None:
        record['queue_wait_s'] = round(row['queue_wait_s'], 6)
    elif isinstance(row, dict):
        record['errors'].append('runtime row lacks queue_wait_s')
    if isinstance(row, dict) and type(row.get('capacity_waits')) is int:
        record['capacity_waits'] = row['capacity_waits']
    elif isinstance(row, dict):
        record['errors'].append('runtime row lacks capacity_waits')
    sdir = out / ('s%d' % session)
    try:
        response = json.loads((sdir / ('http-response-%d.json' % turn)).read_bytes())
        record['gen_tokens'] = (response.get('usage') or {}).get('completion_tokens')
    except Exception as exc:
        record['errors'].append('response: %s: %s' % (type(exc).__name__, exc))
    entries = client.get('turns') or []
    entry = entries[turn] if turn < len(entries) else {}
    record.update(client_e2e_s=entry.get('latency_s'), t_send_s=entry.get('t_send_s'), t_recv_s=entry.get('t_recv_s'))
    if None not in (record['t_send_s'], record['t_recv_s'], record['queue_wait_s']):
        record['service_s'] = round(record['t_recv_s'] - record['t_send_s'] - record['queue_wait_s'], 3)
        record['batch_mean'] = running_between(timeline, record['t_send_s'] + record['queue_wait_s'], record['t_recv_s'])
    if log_error:
        record['admission_error'] = log_error
    else:
        values = accepted_admissions(found['requests'].get(key, []) if key else [])
        if len(values) != 1:
            record['admission_error'] = '%d accepted admission records for this request' % len(values)
        else:
            value = values[0]
            record.update(admission_ms=value['frontend_ms'], engine_admission_ms=value['engine_ms'],
                          parked=value['parked'], park_dispatch_ms=value['dispatch_ms'] if value['parked'] else None,
                          resume_ms=value['resume_ms'], reparks=value.get('reparks') or 0)
            if value['outcome'] != 'ok' or value['frontend_ms'] is None:
                record['admission_error'] = 'admission outcome %r' % value['outcome']
    if record['goal_async_ms'] is not None and None not in (record['t_send_s'], record['queue_wait_s']):
        # Parked at its Goal: from the admission start (the queue wait ends there) past its dispatch piece, for
        # goal_async_ms; the client clock, so frame transfer (milliseconds) is left in.
        start = record['t_send_s'] + record['queue_wait_s'] + (record['park_dispatch_ms'] or 0.0) / 1000
        record.update(park_start_s=round(start, 3), park_end_s=round(start + record['goal_async_ms'] / 1000, 3))
        record['parked_running_mean'] = running_between(timeline, record['park_start_s'], record['park_end_s'])
    if record['phase1_ms'] is not None and record['phase1_tokens']:
        record['phase1_ms_per_1k'] = round(record['phase1_ms'] / (record['phase1_tokens'] / 1000), 3)
    if record['admission_ms'] is not None:
        # Render, Goal, selection and reserve: the admission minus its phase-1 forward (ms1's derivation).
        record['selection_ms'] = round(record['admission_ms'] - (record['phase1_ms'] or 0.0), 3)
    path = sdir / ('metrics-%d.json' % turn)
    metrics_errors, notes = [], []
    try:
        metrics = json.loads(path.read_bytes())
        metrics_errors = list(metrics.get('errors') or [])
        window = chain.window((metrics.get('after') or {}).get('index'))
        if window is None:
            fields, _, _ = metric_fields(None, None)
            metrics_errors.append('no chain window holds this request\'s finish')
        else:
            j, delta, finished = window
            fields, errors, notes = metric_fields(delta, record['gen_tokens'], finished)
            metrics_errors += errors + delta['errors']
            if j != metrics['after']['index']:
                notes.append('finish counted in chain window %d, before this response\'s scrape %d'
                             % (j, metrics['after']['index']))
    except FileNotFoundError:
        fields, _, _ = metric_fields(None, None)
        metrics_errors = ['s%d/%s missing' % (session, path.name)]
    except Exception as exc:
        fields, _, _ = metric_fields(None, None)
        metrics_errors = ['s%d/%s unreadable: %s: %s' % (session, path.name, type(exc).__name__, exc)]
    if chain.error:
        metrics_errors.append('metrics chain: ' + chain.error)
    record.update(fields, metrics_error=metrics_errors or None, metrics_note='; '.join(notes) or None)
    record['warnings'] = consistency(record, row) if isinstance(row, dict) else []
    return record


def runtime_rows(out):
    path = Path(out) / 'runtime.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line] if path.exists() else []


def load_timeline(out):
    """(batch timeline, error) of one server: samples absent or unreadable give ([], reason)."""
    try:
        return batch_timeline(out), None
    except Exception as exc:
        return [], '%s: %s' % (type(exc).__name__, exc)


def policy_measurements(out, server_log, result, *, policy, label=None):
    """One record per completed request, in (session, turn) order (measurements.jsonl).

    `valid` only if READY and every check of that request passed. The runtime
    row is joined by the response id of the session's own HTTP response.
    """
    out, label = Path(out), label or policy
    rows = runtime_rows(out)
    by_response = {}
    for position, row in enumerate(rows):
        by_response.setdefault(row.get('response_id'), (position, row))
    statuses = {c['check']: c['status'] for c in result.get('checks') or ()}
    try:
        found, log_error = admissions(Path(server_log).read_text(encoding='utf-8', errors='replace')), None
    except OSError as exc:
        found, log_error = dict(requests={}, unreadable=0, events=0), 'server log unreadable: %s' % type(exc).__name__
    chain = ChainView(out)
    timeline, timeline_error = load_timeline(out)
    records = []
    for session, completed in enumerate(result.get('session_turns') or ()):
        sdir = out / ('s%d' % session)
        client = json.loads((sdir / 'session.json').read_bytes()) if (sdir / 'session.json').exists() else {}
        for turn in range(completed):
            try:
                response_id = json.loads((sdir / ('http-response-%d.json' % turn)).read_bytes()).get('id')
            except Exception:
                response_id = None
            position, row = by_response.get(response_id, (None, None))
            record = turn_measurement(out, session, turn, position, row, client, statuses, found, log_error, policy,
                                      label, chain, timeline)
            if timeline_error:
                record['metrics_error'] = (record['metrics_error'] or []) + ['batch samples: ' + timeline_error]
            records.append(record)
    concurrency(records)
    return records


def service_interval(record):
    if None in (record.get('t_send_s'), record.get('t_recv_s')):
        return None
    return record['t_send_s'] + (record.get('queue_wait_s') or 0.0), record['t_recv_s']


def concurrency(records):
    """Each request's mean number of requests in service (itself included) over its own service interval."""
    spans = [(r, service_interval(r)) for r in records]
    for record, span in spans:
        if span is None or span[1] <= span[0]:
            continue
        overlap = sum(max(0.0, min(span[1], other[1]) - max(span[0], other[0]))
                      for r, other in spans if r is not record and other is not None)
        record['concurrency'] = round(1 + overlap / (span[1] - span[0]), 3)
    return records


# ---------------------------------------------------------------- aggregates

def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def percentile(values, q):
    """Nearest-rank percentile (no interpolation): the value at rank ceil(q/100 x n)."""
    values = sorted(v for v in values if v is not None)
    return values[max(0, math.ceil(q / 100 * len(values)) - 1)] if values else None


def stats(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), mean=mean(values), p50=percentile(values, 50), p95=percentile(values, 95),
                max=max(values) if values else None)


def merged(intervals):
    out = []
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def covered(start, end, union):
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in union)


def bucket_of(prompt_tokens):
    for low, high, name in PROMPT_BUCKETS:
        if low < prompt_tokens <= high:
            return name
    return None


def buckets(records):
    """Per prompt-length bucket, valid measured (non-warm-up) requests only."""
    out = []
    for low, high, name in PROMPT_BUCKETS:
        rows = [r for r in records if r['valid'] and not r['warmup'] and r['prompt_tokens'] is not None
                and low < r['prompt_tokens'] <= high]
        out.append(dict(bucket=name, turns=len(rows), mean_prompt_tokens=mean(r['prompt_tokens'] for r in rows),
                        mean_tpot_ms=mean(r['tpot_ms'] for r in rows),
                        mean_phase1_ms_per_1k=mean(r['phase1_ms_per_1k'] for r in rows),
                        mean_admission_ms=mean(r['admission_ms'] for r in rows),
                        mean_admission_engine_ms=mean(r['admission_engine_ms'] for r in rows),
                        mean_queue_wait_s=mean(r['queue_wait_s'] for r in rows),
                        mean_e2e_s=mean(r['e2e_s'] for r in rows), mean_client_e2e_s=mean(r['client_e2e_s'] for r in rows),
                        mean_batch=mean(r['batch_mean'] for r in rows),
                        mean_active_kv_mb=mean(r['active_kv_bytes'] / MB if r['active_kv_bytes'] is not None else None
                                               for r in rows)))
    return out


def aggregate(records):
    """Server-wide throughput, latency, admission stall and capacity waits of the valid requests of one server.

    Span: first send to last receive. Busy: union of the service intervals
    [send + queue wait, receive]; concurrency: their summed length over the
    span (the mean number of requests in service). Tool time: per session, the
    gaps between a response and the session's next request; the overlap is the
    part of those gaps during which another session's request was in service.
    Admission stall: the engine-thread time of every admission
    (`admission_engine_ms`), during which no running request decodes.
    """
    timed = [r for r in records if r['t_send_s'] is not None and r['t_recv_s'] is not None]
    if not timed:
        return dict(requests=len(records), span_s=None)
    start, end = min(r['t_send_s'] for r in timed), max(r['t_recv_s'] for r in timed)
    span = round(end - start, 3)
    service = [(r['session'],) + service_interval(r) for r in timed]
    union = merged((a, b) for _, a, b in service)
    tool, overlap = 0.0, 0.0
    for session in sorted({r['session'] for r in timed}):
        own = sorted((r['turn'], r) for r in timed if r['session'] == session)
        others = merged((a, b) for s, a, b in service if s != session)
        for (turn, previous), (following, current) in zip(own, own[1:]):
            if following == turn + 1 and current['t_send_s'] > previous['t_recv_s']:
                tool += current['t_send_s'] - previous['t_recv_s']
                overlap += covered(previous['t_recv_s'], current['t_send_s'], others)
    gen = sum(r['gen_tokens'] or 0 for r in timed)
    busy = sum(b - a for a, b in union)
    in_service = sum(max(0.0, b - a) for _, a, b in service)
    stall = sum(r['admission_engine_ms'] or 0.0 for r in timed) / 1000
    per_minute = lambda count: round(count / span * 60, 3) if span > 0 else None
    return dict(requests=len(timed), span_s=span, turns_per_min=per_minute(len(timed)), gen_tokens=gen,
                gen_tokens_per_s=round(gen / span, 3) if span > 0 else None, busy_s=round(busy, 3),
                busy_fraction=round(busy / span, 3) if span > 0 else None,
                concurrency_mean=round(in_service / span, 3) if span > 0 else None, tool_s=round(tool, 3),
                tool_overlap_s=round(overlap, 3), tool_overlap_fraction=round(overlap / tool, 3) if tool > 0 else None,
                admission_stall_s=round(stall, 3), admission_stall_fraction=round(stall / span, 4) if span > 0 else None,
                capacity_waits=sum(r['capacity_waits'] or 0 for r in timed),
                capacity_wait_requests=sum(1 for r in timed if r['capacity_waits']),
                capacity_waits_max=max([r['capacity_waits'] or 0 for r in timed]),
                client_e2e_s=stats([r['client_e2e_s'] for r in timed]),
                service_s=stats([r['service_s'] for r in timed]),
                server_latency_s=stats([r['server_latency_s'] for r in timed]),
                queue_wait_s=stats([r['queue_wait_s'] for r in timed]),
                admission_ms=stats([r['admission_ms'] for r in timed]),
                admission_engine_ms=stats([r['admission_engine_ms'] for r in timed]),
                selection_ms=stats([r['selection_ms'] for r in timed]), tpot_ms=stats([r['tpot_ms'] for r in timed]),
                ttft_s=stats([r['ttft_s'] for r in timed]), e2e_s=stats([r['e2e_s'] for r in timed]),
                batch_mean=stats([r['batch_mean'] for r in timed]), concurrency=stats([r['concurrency'] for r in timed]))


def per_session(records):
    """Per session: latencies, queue waits, capacity waits (retries go to the back of the FIFO and can be overtaken
    again: a session's total and its worst request show starvation), and the active KV it holds after each turn."""
    out = []
    for session in sorted({r['session'] for r in records}):
        rows = sorted((r for r in records if r['session'] == session), key=lambda r: r['turn'])
        out.append(dict(session=session, session_id=rows[0]['session_id'], turns=len(rows),
                        client_e2e_s=stats([r['client_e2e_s'] for r in rows]),
                        queue_wait_s=stats([r['queue_wait_s'] for r in rows]),
                        tpot_ms=stats([r['tpot_ms'] for r in rows]),
                        capacity_waits=sum(r['capacity_waits'] or 0 for r in rows),
                        capacity_waits_max=max(r['capacity_waits'] or 0 for r in rows),
                        capacity_wait_s=round(sum(r['queue_wait_s'] or 0.0 for r in rows if r['capacity_waits']), 3),
                        prompt_tokens=[r['prompt_tokens'] for r in rows],
                        active_kv_bytes=[r['active_kv_bytes'] for r in rows],
                        gen_tokens=sum(r['gen_tokens'] or 0 for r in rows)))
    return out


def parking(records, timeline=None):
    """Goals built off the engine thread (goal_async) and the decode that went on meanwhile, over valid requests.

    Parked: the requests whose native record carries goal_async_ms. Their parked intervals (client clock,
    `park_start_s`..`park_end_s`) never overlap (one admission at a time). While they last: the sampled running
    requests (vLLM's batch; the parked admission is not in it yet), the decode rate (the generated-token counter,
    linear between samples) and the other requests in service on the client clock. `stall_if_sync_s`: the
    admission stall plus the Goal construction time, i.e. what the engine thread would have spent with the
    Goal built in place (the synchronous path of 6e52a94).
    """
    timeline = timeline or []
    timed = [r for r in records if r['t_send_s'] is not None and r['t_recv_s'] is not None]
    rows = [r for r in timed if r.get('goal_async_ms') is not None]
    stall = round(sum(r['admission_engine_ms'] or 0.0 for r in timed) / 1000, 3)
    construct = round(sum(r.get('goal_construct_ms') or 0.0 for r in rows) / 1000, 3)
    out = dict(parked_requests=len(rows), goal_async_s=round(sum(r['goal_async_ms'] for r in rows) / 1000, 3),
               goal_construct_s=round(construct, 3), goal_async_ms=stats([r['goal_async_ms'] for r in rows]),
               goal_construct_ms=stats([r.get('goal_construct_ms') for r in rows]),
               admission_stall_s=round(stall, 3), stall_if_sync_s=round(stall + construct, 3), parked_s=None,
               parked_samples=0, parked_running_mean=None, parked_decode_tokens=None, parked_decode_tok_s=None,
               parked_in_service_mean=None, goal_fallbacks=sum(1 for r in timed if r.get('fallback_reason')
                                                                 not in (None, 'rewriter_disabled')))
    intervals = [(r['park_start_s'], r['park_end_s'], r) for r in rows if r.get('park_start_s') is not None
                 and r['park_end_s'] > r['park_start_s']]
    if not intervals:
        return out
    union = merged((a, b) for a, b, _ in intervals)
    length = sum(b - a for a, b in union)
    samples = [p['running'] for p in timeline if p['ok'] and p['running'] is not None
               and any(a <= p['t_s'] <= b for a, b in union)]
    tokens, seconds = 0.0, 0.0
    for a, b in union:
        x, y = generated_at(timeline, a), generated_at(timeline, b)
        if x is not None and y is not None:
            tokens += y - x
            seconds += b - a
    others = 0.0
    for a, b, record in intervals:
        others += sum(max(0.0, min(b, s[1]) - max(a, s[0])) for s in (service_interval(r) for r in timed if r is not record)
                      if s is not None)
    out.update(parked_s=round(length, 3), parked_samples=len(samples),
               parked_running_mean=round(sum(samples) / len(samples), 3) if samples else None,
               parked_decode_tokens=round(tokens, 1) if seconds > 0 else None,
               parked_decode_tok_s=round(tokens / seconds, 3) if seconds > 0 else None,
               parked_in_service_mean=round(others / sum(b - a for a, b, _ in intervals), 3))
    return out


def goal_turns(records):
    """Per turn over the valid requests that carry a Goal: decisions, rewriter use, fallbacks, Goal and stall times
    and client latency (mean and max over the sessions)."""
    out = []
    rows = [r for r in records if r.get('gate_decision') is not None or r.get('goal_async_ms') is not None]
    for turn in sorted({r['turn'] for r in rows}):
        now = [r for r in rows if r['turn'] == turn]
        top = lambda key: max([r[key] for r in now if r.get(key) is not None], default=None)
        gates = {}
        for r in now:
            gates[str(r.get('gate_decision'))] = gates.get(str(r.get('gate_decision')), 0) + 1
        out.append(dict(turn=turn, requests=len(now), gates=dict(sorted(gates.items())),
                        slm_attempted=sum(1 for r in now if r.get('slm_attempted')),
                        slm_succeeded=sum(1 for r in now if r.get('slm_succeeded')),
                        fallbacks=sorted({r['fallback_reason'] for r in now if r.get('fallback_reason') is not None}),
                        parked=sum(1 for r in now if r.get('goal_async_ms') is not None),
                        goal_async_ms=mean(r.get('goal_async_ms') for r in now), goal_async_ms_max=top('goal_async_ms'),
                        goal_construct_ms=mean(r.get('goal_construct_ms') for r in now),
                        goal_construct_ms_max=top('goal_construct_ms'),
                        admission_engine_ms=mean(r.get('admission_engine_ms') for r in now),
                        admission_engine_ms_max=top('admission_engine_ms'),
                        client_e2e_s=mean(r.get('client_e2e_s') for r in now), client_e2e_s_max=top('client_e2e_s'),
                        parked_running_mean=mean(r.get('parked_running_mean') for r in now)))
    return out


def phase1_summary(records, batch=None, totals=None):
    """perf-adm: how phase 1 ran on one server's valid continuing turns. Modes (`none`: no phase1_mode, a flag-off or
    one-session server), the share that ran mixed, the parked ones, the step that carried a probe (phase1_step_ms,
    rows = the probe plus the decodes) against the clean ITL (2026-11-10; was the steady one) of a plain decode step at
    rows - 1 running requests (batch_profile; `step_excess_ms` = step - that ITL, where the profile has the batch
    size), the phase 1 of parked
    fallbacks (a separate pass while decodes ran: a stall), the restore H2D and D2D bytes per continuing turn and, with
    the aggregate (`totals`), the stall share counting the probe steps' excess and the parked fallbacks too."""
    turns = [r for r in records if r['turn'] and r.get('phase1_tokens')]
    modes = {}
    for r in turns:
        modes[str(r.get('phase1_mode') or 'none')] = modes.get(str(r.get('phase1_mode') or 'none'), 0) + 1
    mixed = [r for r in turns if r.get('phase1_mode') == 'mixed']
    # 2026-11-10: the plain step is the clean cell (batch_profile), no longer the steady one (FINDINGS 1.2)
    itl = {b['batch']: b['clean_itl_ms'] for b in (batch or {}).get('batches') or () if b.get('clean_itl_ms') is not None}
    plain = [itl.get((r.get('phase1_step_rows') or 0) - 1) for r in mixed]
    excess = [round(r['phase1_step_ms'] - p, 3) for r, p in zip(mixed, plain)
              if p is not None and r.get('phase1_step_ms') is not None]
    fallback = [r['phase1_ms'] for r in turns if r.get('phase1_parked') and r.get('phase1_mode') == 'fallback'
                and r.get('phase1_ms') is not None]
    out = dict(continuing_turns=len(turns), modes=dict(sorted(modes.items())), mixed=len(mixed),
               mixed_share=round(len(mixed) / len(turns), 4) if turns else None,
               parked=sum(1 for r in turns if r.get('phase1_parked')),
               reparked=sum(1 for r in turns if r.get('reparks')),
               park_ms=stats([r.get('phase1_park_ms') for r in turns]),
               step_ms=stats([r.get('phase1_step_ms') for r in mixed]),
               step_rows=stats([r.get('phase1_step_rows') for r in mixed]),
               plain_itl_ms_at_rows=stats(plain), step_excess_ms=stats(excess),
               step_excess_s=round(sum(max(0.0, e) for e in excess) / 1000, 3), steps_without_plain_itl=len(mixed) - len(excess),
               phase1_ms=stats([r.get('phase1_ms') for r in turns]),
               fallback_phase1_s=round(sum(fallback) / 1000, 3),
               restore_h2d_bytes=stats([r.get('h2d_other_bytes') for r in turns]),
               d2d_bytes=stats([r.get('d2d_bytes') for r in turns]), stall_with_steps_s=None,
               stall_with_steps_fraction=None)
    span, stall = (totals or {}).get('span_s'), (totals or {}).get('admission_stall_s')
    if span and stall is not None:
        extra = out['step_excess_s'] + out['fallback_phase1_s']
        out.update(stall_with_steps_s=round(stall + extra, 3), stall_with_steps_fraction=round((stall + extra) / span, 4))
    return out


def summarize(records, timeline=None):
    """The summary section of one server: per-request flags, buckets, aggregate, sessions, the batch profile, the
    Goals built off the engine thread (`parking`, `goal_turns`) and (perf-adm) how phase 1 ran (`phase1`)."""
    valid = [r for r in records if r['valid']]
    key = lambda r: [r['session'], r['turn']]
    totals, batch = aggregate(valid), batch_profile(timeline or [])
    return dict(requests=len(records), valid_requests=len(valid), invalid_requests=[key(r) for r in records if not r['valid']],
                receipt_errors=[key(r) for r in valid if r['errors']],
                metrics_error_requests=[key(r) for r in records if r['metrics_error']],
                admission_error_requests=[key(r) for r in records if r['admission_error']],
                warning_requests=[key(r) for r in records if r['warnings']], buckets=buckets(records),
                aggregate=totals, sessions=per_session(valid), batch=batch,
                parking=parking(valid, timeline), goal_turns=goal_turns(valid),
                phase1=phase1_summary(valid, batch, totals), overlap=overlap_summary(valid, totals))


def overlap_summary(records, totals=None):
    """(f) --admission-overlap: what several admissions in flight change, per server (any server: off is the baseline).
    The queue wait (the FIFO / permit wait plus every CAPACITY_WAIT retry: mean, p95 and its share of the request time,
    the sum of client end-to-end times), the parked Goal's wait and construction (goal_async_ms, goal_construct_ms),
    the lease wait (lease_wait_ms, (f) with (b)), the engine-thread admission time, CAPACITY_WAITs, turns/min and the
    rewriter's busy share: the construction time of the turns whose Goal called the rewriter (slm_attempted) over the
    span (an upper bound: a construction also runs the deterministic Goal parts)."""
    totals = totals or {}
    queue = [r['queue_wait_s'] for r in records if r.get('queue_wait_s') is not None]
    request = [r['client_e2e_s'] if r.get('client_e2e_s') is not None else
               (r['t_recv_s'] - r['t_send_s'] if None not in (r.get('t_send_s'), r.get('t_recv_s')) else None)
               for r in records]
    request_s = sum(x for x in request if x is not None)
    rewrites = [r for r in records if r.get('slm_attempted') is True]
    busy = sum(r['goal_construct_ms'] or 0.0 for r in rewrites) / 1000
    span = totals.get('span_s')
    return dict(requests=len(records), queue_wait_s=stats(queue),
                queue_wait_share=round(sum(queue) / request_s, 4) if request_s > 0 and queue else None,
                goal_async_ms=stats([r.get('goal_async_ms') for r in records]),
                goal_construct_ms=stats([r.get('goal_construct_ms') for r in records]),
                lease_waits=sum(1 for r in records if r.get('lease_waited')),
                lease_wait_ms=stats([r.get('lease_wait_ms') for r in records if r.get('lease_waited')]),
                admission_engine_ms=stats([r.get('admission_engine_ms') for r in records]),
                capacity_waits=totals.get('capacity_waits'), turns_per_min=totals.get('turns_per_min'),
                rewriter_calls=len(rewrites), rewriter_busy_s=round(busy, 3),
                rewriter_busy_share=round(busy / span, 4) if span else None,
                decisions={d: sum(1 for r in records if r.get('applied_decision') == d)
                           for d in sorted({r.get('applied_decision') for r in records} - {None})})
