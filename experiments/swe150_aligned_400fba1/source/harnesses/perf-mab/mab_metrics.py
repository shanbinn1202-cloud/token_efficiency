"""perf-mab /metrics samples of one server (stdlib only; delivered; the verifier recomputes the summary).

`MetricsSampler` scrapes GET /metrics every `interval` seconds from READY to the stop (metrics/samples.jsonl: one row
per scrape with the vllm:* families perf-swe16 samples - running / waiting gauges, KV usage, generation counter,
preemptions, finished requests, the inter-token-latency histogram - and the text's SHA-256). `summarize` derives,
from the archived samples only: the preemption counter's maximum (0 on our runtime; a stock server's count is a
finding, mab_checks.check_metrics), the peak running count, and the inter-token latency by running count over clean
intervals (two consecutive readable samples with the same running
count b >= 1, nothing waiting, no inter-token latency above measure.STALL_SECONDS, at least one observation):
ITL(b) = the histogram's sum / count over those intervals - the TPOT a request sees while b decode together, for both
server kinds alike (G2: "ITL with 16 running").
"""
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

import measure

SAMPLES = 'samples.jsonl'
SCRAPE_TIMEOUT = 10
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _family(name):
    for suffix in ('_bucket', '_count', '_sum'):
        if name.endswith(suffix) and name[:-len(suffix)] in measure.SAMPLE_FAMILIES:
            return name[:-len(suffix)]
    return name


def filter_text(raw):
    keep = []
    for line in measure.vllm_text(raw).splitlines():
        if line.startswith('# '):
            parts = line.split(None, 3)
            name = parts[2] if len(parts) > 2 else ''
        else:
            name = line.split('{', 1)[0].split(None, 1)[0]
        if not name.endswith('_created') and _family(name) in measure.SAMPLE_FAMILIES:
            keep.append(line)
    return '\n'.join(keep) + '\n' if keep else ''


def http_get(url, timeout):
    try:
        with _OPENER.open(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class MetricsSampler:
    """One /metrics scrape every `interval` seconds from start to finish, on the server's epoch clock."""

    def __init__(self, base_url, out, *, interval, epoch, clock=time.monotonic, get=None):
        self.url = base_url.rstrip('/') + measure.METRICS_PATH
        self.dir = Path(out) / 'metrics'
        self.interval, self.epoch, self.clock, self.get = interval, epoch, clock, get or http_get
        self.stopped, self.thread, self.stream, self.index = threading.Event(), None, None, 0

    def start(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stream = (self.dir / SAMPLES).open('x', encoding='utf-8', newline='\n')
        self.thread = threading.Thread(target=self._loop, name='metrics-sampler', daemon=True)
        self.thread.start()
        return self

    def sample(self):
        started = self.clock()
        status, text, error = None, '', None
        try:
            status, raw = self.get(self.url, SCRAPE_TIMEOUT)
            text = filter_text(raw) if status == 200 else ''
        except Exception as exc:  # recorded per sample, never raised into the job
            error = '%s: %s' % (type(exc).__name__, exc)
        raw = text.encode()
        row = dict(index=self.index, t_s=round(started - self.epoch, 4), seconds=round(self.clock() - started, 4),
                   mono_ns=time.monotonic_ns(), status=status, error=error, bytes=len(raw),
                   sha256=hashlib.sha256(raw).hexdigest(), text=text)
        self.stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
        self.stream.flush()
        self.index += 1

    def _loop(self):
        next_at = self.clock()
        while not self.stopped.is_set():
            self.sample()
            next_at += self.interval
            delay = next_at - self.clock()
            if delay < 0:
                next_at, delay = self.clock(), 0
            if self.stopped.wait(delay):
                break

    def finish(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=SCRAPE_TIMEOUT + 5)
        if self.stream is not None:
            self.sample()
            self.stream.close()
        return self.index


def load_samples(out):
    path = Path(out) / 'metrics' / SAMPLES
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    for position, row in enumerate(rows):
        if row['index'] != position:
            raise ValueError('metrics sample index %s at position %d' % (row['index'], position))
        if hashlib.sha256(row['text'].encode()).hexdigest() != row['sha256']:
            raise ValueError('metrics sample %d differs from its digest' % position)
    return rows


def timeline(out):
    rows = []
    for sample in load_samples(out):
        values = measure.sample_values(sample['text']) if sample['status'] == 200 and sample['text'] else None
        rows.append(dict(index=sample['index'], t_s=sample['t_s'], ok=values is not None,
                         **(values or dict.fromkeys(measure.SAMPLE_KEYS))))
    return rows


def _delta(b, a, key):
    return measure.minus(b[key], a[key]) if a[key] is not None and b[key] is not None else None


def summarize(out):
    """The server's metrics summary (see the module docstring)."""
    points = timeline(out)
    readable = [p for p in points if p['ok']]
    cells = {}
    for a, c in zip(readable, readable[1:]):
        if a['running'] is None or a['running'] != c['running'] or a['running'] < 1:
            continue
        if (a['waiting'] or 0) or (c['waiting'] or 0):
            continue
        slow, count = _delta(c, a, 'itl_slow'), _delta(c, a, 'itl_count')
        if slow is None or slow != 0 or not count:
            continue
        cell = cells.setdefault(int(a['running']), dict(intervals=0, seconds=0.0, itl_sum=0.0, itl_count=0.0))
        cell['intervals'] += 1
        cell['seconds'] += c['t_s'] - a['t_s']
        cell['itl_sum'] += _delta(c, a, 'itl_sum') or 0.0
        cell['itl_count'] += count
    itl = {str(b): dict(intervals=v['intervals'], seconds=round(v['seconds'], 2),
                        itl_ms=round(1000 * v['itl_sum'] / v['itl_count'], 2) if v['itl_count'] else None,
                        observations=int(v['itl_count'])) for b, v in sorted(cells.items())}
    preemptions = [p['preemptions'] for p in readable if p['preemptions'] is not None]
    running = [p['running'] for p in readable if p['running'] is not None]
    return dict(samples=len(points), readable=len(readable), preemption_samples=len(preemptions),
                preemptions_max=max(preemptions) if preemptions else None,
                running_max=max(running) if running else None, itl_by_running=itl,
                generated_total=readable[-1]['generated'] if readable else None)
