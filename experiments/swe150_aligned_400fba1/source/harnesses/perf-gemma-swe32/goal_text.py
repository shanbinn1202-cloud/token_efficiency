"""The per-turn Goal-text evidence of the selective-prefix runtime (runtime branch goal-text), stdlib only: which files
hold it, how to read them, and the hash verification. Delivered with the payload (swe_checks: presence and parse on
the GPU node); used offline by postrun/verify_evidence.py (the verification) and postrun/pack_gswe32.py (the texts per
instance for Xuanxue).

What the runtime records (branch goal-text, 227e32c "Record goal_query Goal, query and rewriter texts in goals.jsonl"):
one row per terminal goal_query request (completed or cancelled) in `goals.jsonl` of the server's output directory,
schema remote-ra-goal-text-v1, the canonical JSON bytes of
  {request_id, session_id, session_index, epoch, turn, local_turn, terminal_state, schema,
   goal: {sha256, text},            text = canonical JSON of the GoalRecord selector payload (schema_version,
                                    root_goal, retrieval_goal, policy_hint); sha256 = runtime.jsonl query.goal_hash,
   query: {variant, sha256, text},  text = the <retrieval_query> text; sha256 = query.goal.payload_sha256,
   decision: {gate_decision, applied_decision, triggers, changed_fields, slm_attempted, slm_succeeded, fallback_reason},
   rewrites: [{request, response, error}, ...]}   one per rewriter call of the turn's Goal construction;
and the native file and the runtime row name their row: `goal_text: {file, sha256 of the row's bytes}` (native) and
`query.goal_text` (runtime row, a copy). Commit 00e173d and earlier record the hashes only (NOT_AVAILABLE).

The reader stays tolerant of another record layout (it was written before that commit existed):
- files: the names the sealed source declares (pins runtime_features.goal_text.files, controller.goal_text_feature) plus
  any *.json / *.jsonl in the output directory whose name holds 'goal' or 'rewriter' (no such file exists at 00e173d);
- records: JSON lines (each line's SHA-256 kept for the row references), or one JSON list, or one object holding a list;
- join with runtime.jsonl: the row's query.goal_text reference (file + SHA-256 of the line), else request_id, else
  response_id, else (session_index or session, turn); records without any key are matched by their hashes alone;
- texts of a record: every string in it, the canonical JSON of every object / list in it, and the selector-payload
  projection of every object holding schema_version, root_goal and retrieval_goal.
verify_rows: every goal_query turn (a runtime row with query.goal) has records whose texts hash to its goal_hash and to
its payload_sha256; every reference names an existing row of the named file; every {sha256, text} pair and every
`<name>_sha256` next to a `<name>` value hash right (string, canonical JSON or projection); a record's own top-level
goal_hash / payload_sha256 (also under `goal`) equal its row's.

Rewriter timing (runtime 2867301, README section 8; `timing` = pins runtime_features.goal_rewriter_timing): every
rewrites[] entry holds wait_ms, start_ms, call_ms (ms, the server's time.perf_counter, rounded to 3 decimals) and
input_tokens, output_tokens, generation_latency_ms (the child's answer line), each a non-negative number or null; with
--goal-rewriter-batch N (runtime 5b15b2f; `batch` = N) also batch_id and batch_size (1 <= size <= N; the calls of one
batch agree on its size and are at most that many), without it neither; and the request's native file, when its Goal
was built on a Goal thread (goal_construct_ms) and called the rewriter, holds goal_rewrite_wait_ms and
goal_rewrite_call_ms equal to the runtime's goal_text.totals of the row's entries (the sum in order, rounded to 3
decimals, null when an entry's time is null), and a native file without rewriter calls holds neither. Null times are a
finding (GOAL_REWRITE_TIMING_NULL), not a failure.
"""
import hashlib
import json
from pathlib import Path
import re

NAME = re.compile(r'(goal|rewriter)', re.I)
SUFFIXES = ('.json', '.jsonl')
# Evidence files of the runtime's output directory that are never Goal-text files (00e173d's names).
KNOWN = frozenset({'startup.json', 'ready.json', 'launch.json', 'runtime.jsonl', 'sessions.jsonl', 'failures.jsonl',
                   'measurements.jsonl', 'worker-attached.json', 'engine-cleanup.json', 'parent-cleanup.json'})
PROJECTION_KEYS = ('schema_version', 'root_goal', 'retrieval_goal')
MAX_RECORDS = 200000
ROW = 'row_sha256'


class GoalTextError(ValueError):
    pass


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def canonical(value):
    """GoalRecord._stable_hash's serialization (= remote_ra_live_v2.goal_text.canonical_text)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def projection(value):
    """The selector payload of a GoalRecord-like object (its version_hash input), or None."""
    if isinstance(value, dict) and all(k in value for k in PROJECTION_KEYS):
        return dict(schema_version=value['schema_version'], root_goal=value['root_goal'],
                    retrieval_goal=value['retrieval_goal'], policy_hint=value.get('policy_hint'))
    return None


def candidate_files(out, declared=()):
    """The Goal-text files of one server output directory: the declared names that exist, then every other
    *.json / *.jsonl whose name holds 'goal' or 'rewriter' (sorted). Native receipts and the known files never count."""
    out = Path(out)
    found = [out / name for name in declared if (out / name).is_file()]
    if out.is_dir():
        for path in sorted(out.iterdir()):
            if (path.is_file() and path.suffix in SUFFIXES and NAME.search(path.name) and path.name not in KNOWN
                    and not path.name.startswith('native-') and path not in found):
                found.append(path)
    return found


def read_rows(path):
    """[(record, SHA-256 of its line or None)] of one file: JSON lines (.jsonl, or a .json with one object per line),
    one JSON list, or one object holding exactly one list of objects. Every record must be an object."""
    raw = Path(path).read_bytes()
    rows, value = None, None

    def lines():
        out = []
        for line in raw.split(b'\n'):
            line = line[:-1] if line.endswith(b'\r') else line
            if line.strip():
                out.append((json.loads(line.decode('utf-8')), hashlib.sha256(line).hexdigest()))
        return out
    if Path(path).suffix == '.jsonl':
        rows = lines()
    else:
        try:
            value = json.loads(raw.decode('utf-8'))
        except ValueError:
            rows = lines()
    if rows is None:
        if isinstance(value, list):
            records = value
        elif isinstance(value, dict):
            lists = [v for v in value.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
            records = lists[0] if len(lists) == 1 else [value]
        else:
            raise GoalTextError('%s holds no JSON object' % Path(path).name)
        rows = [(r, None) for r in records]
    if not all(isinstance(r, dict) for r, _ in rows):
        raise GoalTextError('%s: every record must be a JSON object' % Path(path).name)
    if len(rows) > MAX_RECORDS:
        raise GoalTextError('%s: more than %d records' % (Path(path).name, MAX_RECORDS))
    return rows


def read_records(path):
    return [r for r, _ in read_rows(path)]


def walk(value, path=()):
    """(path, node) for every node of a JSON value, the value first."""
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, path + (str(index),))


def text_hashes(record):
    """{sha256: path} of every text a record holds: strings, canonical JSON of objects and lists, projections."""
    out = {}
    for path, node in walk(record):
        if isinstance(node, str):
            out.setdefault(sha(node), '/'.join(path) or '.')
        elif isinstance(node, (dict, list)) and path:
            out.setdefault(sha(canonical(node)), '/'.join(path) + '(json)')
            projected = projection(node)
            if projected is not None:
                out.setdefault(sha(canonical(projected)), '/'.join(path) + '(selector payload)')
    projected = projection(record)
    if projected is not None:
        out.setdefault(sha(canonical(projected)), '.(selector payload)')
    return out


def self_pairs(record):
    """Every {sha256, text} object (goals.jsonl's goal / query) and every `<name>_sha256` field next to a non-null
    `<name>` field: (path, ok). The value's hash may be of the string, of its canonical JSON or of its projection."""
    pairs = []

    def wanted(value):
        out = {sha(value)} if isinstance(value, str) else {sha(canonical(value))}
        if projection(value) is not None:
            out.add(sha(canonical(projection(value))))
        return out
    for path, node in walk(record):
        if not isinstance(node, dict):
            continue
        if isinstance(node.get('sha256'), str) and node.get('text') is not None:
            pairs.append(('/'.join(path + ('sha256',)), node['sha256'].lower() in wanted(node['text'])))
        for key, digest in node.items():
            if not (isinstance(key, str) and key.endswith('_sha256') and isinstance(digest, str)):
                continue
            base = key[:-len('_sha256')]
            if base not in node or node[base] is None:
                continue
            pairs.append(('/'.join(path + (key,)), digest.lower() in wanted(node[base])))
    return pairs


def keys_of(record):
    """The record's own join keys with runtime.jsonl, strongest first."""
    keys = []
    for name in ('request_id', 'response_id'):
        if isinstance(record.get(name), str) and record[name]:
            keys.append((name, record[name]))
    session = record.get('session_index', record.get('session'))
    if type(session) is int and type(record.get('turn')) is int:
        keys.append(('session_turn', (session, record['turn'])))
    return keys


def reference_of(row):
    """The row's reference to its Goal-text row ({file, sha256}, goal-text runtime), or None."""
    query = row.get('query')
    value = query.get('goal_text') if isinstance(query, dict) else None
    return value if isinstance(value, dict) and isinstance(value.get('sha256'), str) else None


def row_keys(row):
    keys = []
    ref = reference_of(row)
    if ref is not None:
        keys.append((ROW, (ref.get('file'), ref['sha256'])))
    return keys + [('request_id', row.get('request_id')), ('response_id', row.get('response_id')),
                   ('session_turn', (row.get('session_index'), row.get('turn')))]


def goal_of(row):
    """(goal_hash, payload_sha256) of a runtime row with a Goal, else None."""
    query = row.get('query')
    if not isinstance(query, dict) or not isinstance(query.get('goal'), dict):
        return None
    return query.get('goal_hash'), query['goal'].get('payload_sha256')


def load_rows(files):
    """{file name: [(record, line SHA-256 or None)]} (raises GoalTextError on an unreadable file)."""
    out = {}
    for path in files:
        try:
            out[Path(path).name] = read_rows(path)
        except (OSError, ValueError) as exc:
            raise GoalTextError('%s: %s: %s' % (Path(path).name, type(exc).__name__, exc)) from None
    return out


def load_files(files):
    """{file name: [records]} (raises GoalTextError on an unreadable file)."""
    return {name: [r for r, _ in rows] for name, rows in load_rows(files).items()}


def as_rows(records_by_file):
    """Accept {name: [records]} or {name: [(record, line hash)]}."""
    return {name: [v if isinstance(v, tuple) else (v, None) for v in values]
            for name, values in records_by_file.items()}


def presence(out, policy, declared=()):
    """The GPU node's light check (swe_checks): which files exist, their sizes, SHA-256 and record counts. A goal_query
    server of a runtime that declares the feature must have written at least one readable, non-empty file."""
    files = candidate_files(out, declared)
    value = dict(declared=list(declared), files={}, records=0)
    loaded = load_rows(files)
    for path in files:
        raw = Path(path).read_bytes()
        value['files'][path.name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                         records=len(loaded[path.name]))
        value['records'] += len(loaded[path.name])
    value['undeclared'] = sorted(set(value['files']) - set(declared))
    if declared and policy == 'goal_query' and not value['records']:
        raise GoalTextError('the runtime declares Goal-text files %s but the goal_query server wrote no record (%s)'
                            % (list(declared), sorted(value['files']) or 'no file'))
    return value


def index_of(rows_by_file):
    """{join key: [(file, position, record)]} over every record's own keys and its line's reference key."""
    index = {}
    for name, values in sorted(rows_by_file.items()):
        for position, (record, digest) in enumerate(values):
            keys = keys_of(record) + ([(ROW, (name, digest))] if digest else [])
            for key in keys:
                index.setdefault(key, []).append((name, position, record))
    return index


TIMING_KEYS = ('wait_ms', 'start_ms', 'call_ms', 'input_tokens', 'output_tokens', 'generation_latency_ms')
TOTAL_KEYS = (('goal_rewrite_wait_ms', 'wait_ms'), ('goal_rewrite_call_ms', 'call_ms'))
BATCH_KEYS = ('batch_id', 'batch_size')


def native_of(out, request_id):
    """The native terminal file of a request (smoke_checks.native_path), or None."""
    path = Path(out) / ('native-' + hashlib.sha256(request_id.encode()).hexdigest() + '.json')
    return json.loads(path.read_bytes()) if path.is_file() else None


def rewrite_totals(rewrites):
    """The runtime's goal_text.totals: per native key the sum of the entries' times in order, rounded to 3 decimals,
    null when an entry's time is null; {} without entries."""
    if not rewrites:
        return {}
    out = {}
    for key, name in TOTAL_KEYS:
        values = [exchange.get(name) for exchange in rewrites]
        out[key] = None if any(v is None for v in values) else round(sum(values), 3)
    return out


def _quantile(values, q):
    xs = sorted(values)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def verify_timing(records, pairs, timing, batch=None, natives=None):
    """The per-call timing / batch fields of every record's rewrites[] and the native totals of the joined rows
    (`pairs`: [(row, [records])]; `natives(request_id)` -> the native file or None). Returns (summary, reasons,
    findings)."""
    reasons, findings, nulls = [], [], 0
    bad_entries, bad_batch, bad_totals = [], [], []
    waits, calls, generations, sizes, batches = [], [], [], {}, {}
    for name, position, record in records:
        rewrites = record.get('rewrites') if isinstance(record.get('rewrites'), list) else []
        for i, exchange in enumerate(rewrites):
            where = '%s[%d].rewrites[%d]' % (name, position, i)
            if not isinstance(exchange, dict):
                bad_entries.append(where + ': not an object')
                continue
            missing = [k for k in TIMING_KEYS if k not in exchange]
            wrong = [k for k in TIMING_KEYS if k in exchange and exchange[k] is not None and (
                type(exchange[k]) not in (int, float) or exchange[k] < 0)]
            if missing or wrong:
                bad_entries.append('%s: missing %s, not a non-negative number %s' % (where, missing, wrong))
                continue
            nulls += sum(1 for k in ('wait_ms', 'call_ms') if exchange[k] is None)
            waits += [exchange['wait_ms']] if exchange['wait_ms'] is not None else []
            calls += [exchange['call_ms']] if exchange['call_ms'] is not None else []
            generations += [exchange['generation_latency_ms']] if exchange['generation_latency_ms'] is not None else []
            present = [k for k in BATCH_KEYS if k in exchange]
            if batch is None:
                if present:
                    bad_batch.append('%s: %s on a server without --goal-rewriter-batch' % (where, present))
                continue
            ident, size = exchange.get('batch_id'), exchange.get('batch_size')
            if not (type(ident) is int and ident >= 0 and type(size) is int and 1 <= size <= batch):
                bad_batch.append('%s: batch_id %r batch_size %r (N = %d)' % (where, ident, size, batch))
                continue
            sizes[size] = sizes.get(size, 0) + 1
            batches.setdefault(ident, []).append(size)
    for ident, values in batches.items():
        if len(set(values)) != 1 or len(values) > values[0]:
            bad_batch.append('batch %s: sizes %s over %d calls' % (ident, sorted(set(values)), len(values)))
    checked = skipped = 0
    for row, matched in pairs:
        native = natives(row['request_id']) if natives is not None and isinstance(row.get('request_id'), str) else None
        if native is None:
            skipped += 1
            continue
        rewrites = [x for _, _, r in matched for x in (r.get('rewrites') if isinstance(r.get('rewrites'), list)
                                                       else []) if isinstance(x, dict)]
        has = {k: k in native for k, _ in TOTAL_KEYS}
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        checked += 1
        if not rewrites:
            if any(has.values()):
                bad_totals.append('%s: native totals without a rewriter call' % where)
            continue
        if 'goal_construct_ms' not in native and not any(has.values()):
            continue  # the Goal was not built on a Goal thread: no totals (runtime goal_async)
        wanted = rewrite_totals(rewrites)
        got = {k: native.get(k, 'absent') for k, _ in TOTAL_KEYS}
        if got != wanted:
            bad_totals.append('%s: native %s, the entries give %s' % (where, got, wanted))
    if bad_entries:
        reasons.append('%d rewriter entries without the timing fields: %s' % (len(bad_entries), bad_entries[:5]))
    if bad_batch:
        reasons.append('%d rewriter batch fields wrong: %s' % (len(bad_batch), bad_batch[:5]))
    if bad_totals:
        reasons.append('%d native rewrite totals differ: %s' % (len(bad_totals), bad_totals[:5]))
    if nulls:
        findings.append(dict(finding='GOAL_REWRITE_TIMING_NULL', times=nulls))
    summary = dict(calls=len(calls), nulls=nulls, natives_checked=checked, natives_absent=skipped,
                   wait_ms=dict(p50=_quantile(waits, 0.5), p90=_quantile(waits, 0.9), max=max(waits, default=None),
                                total=round(sum(waits), 3)),
                   call_ms=dict(p50=_quantile(calls, 0.5), max=max(calls, default=None), total=round(sum(calls), 3)),
                   generation_latency_ms_p50=_quantile(generations, 0.5), batch=batch,
                   batch_sizes={str(k): v for k, v in sorted(sizes.items())}, batches=len(batches),
                   bad_entries=bad_entries[:20], bad_batch=bad_batch[:20], bad_totals=bad_totals[:20])
    return summary, reasons, findings


def verify_rows(records_by_file, rows, policy, declared=(), timing=None, batch=None, natives=None):
    """The verification of one server (verify_evidence). `records_by_file`: {name: [records]} or {name: [(record, line
    SHA-256)]} (load_rows). With `timing` (the runtime's goal_rewriter_timing feature) also the rewriter's per-call
    timing, `batch` (N of --goal-rewriter-batch) its batch fields, `natives(request_id)` the native totals (verify_timing).
    Returns dict(status, ...): VERIFIED, NOT_AVAILABLE (no file and nothing declared) or FAIL (reasons)."""
    rows_by_file = as_rows(records_by_file)
    reasons, findings = [], []
    goal_rows = [r for r in rows if goal_of(r) is not None]
    records = [(name, position, record) for name, values in sorted(rows_by_file.items())
               for position, (record, _) in enumerate(values)]
    value = dict(policy=policy, declared=list(declared), files=sorted(rows_by_file), records=len(records),
                 goal_turns=len(goal_rows), matched=0, matched_goal_hash=0, matched_payload=0, missing=[],
                 self_pairs=0, self_pair_failures=[], copy_mismatches=[], bad_references=[], joined_by=dict(),
                 unjoined_records=0, records_with_error=0, rewriter_calls=0, rewriter_errors=0,
                 terminal_states={}, findings=findings)
    references = [r for r in rows if reference_of(r) is not None]
    if not records:
        if references or (declared and policy == 'goal_query' and goal_rows):
            value.update(status='FAIL', reasons=['no Goal-text record although %s' % (
                '%d runtime rows reference one' % len(references) if references else
                'the runtime declares %s' % list(declared))])
        else:
            value.update(status='NOT_AVAILABLE', reasons=[])
        return value
    undeclared = sorted(set(rows_by_file) - set(declared))
    if undeclared:
        findings.append(dict(finding='GOAL_TEXT_UNDECLARED', files=undeclared))
    index = index_of(rows_by_file)
    for name, position, record in records:
        value['records_with_error'] += 1 if any(k.endswith('error') and record.get(k) for k in record) else 0
        for path, ok in self_pairs(record):
            value['self_pairs'] += 1
            if not ok:
                value['self_pair_failures'].append('%s[%d] %s' % (name, position, path))
        rewrites = record.get('rewrites')
        if isinstance(rewrites, list):
            value['rewriter_calls'] += len(rewrites)
            value['rewriter_errors'] += sum(1 for x in rewrites if isinstance(x, dict) and x.get('error'))
        state = record.get('terminal_state')
        if isinstance(state, str):
            value['terminal_states'][state] = value['terminal_states'].get(state, 0) + 1
    keyed = any(key[0] != ROW for key in index)
    hashes = {(name, position): text_hashes(record) for name, position, record in records}
    used = set()
    pairs = []
    for row in references:
        ref = reference_of(row)
        if (ROW, (ref.get('file'), ref['sha256'])) not in index:
            value['bad_references'].append('s%s turn %s: %s %s' % (row.get('session_index'), row.get('turn'),
                                                                   ref.get('file'), ref['sha256'][:12]))
    for row in goal_rows:
        goal_hash, payload = goal_of(row)
        matched, how = [], None
        for key in row_keys(row):
            if key in index:
                matched, how = index[key], key[0]
                break
        if not matched and not keyed:
            matched = [(n, p, r) for n, p, r in records
                       if goal_hash in hashes[(n, p)] and payload in hashes[(n, p)]]
            how = 'hash' if matched else None
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        if not matched:
            value['missing'].append(where)
            continue
        value['joined_by'][how] = value['joined_by'].get(how, 0) + 1
        pairs.append((row, matched))
        found = {}
        for name, position, record in matched:
            used.add((name, position))
            found.update(hashes[(name, position)])
            for holder in (record, record.get('goal') if isinstance(record.get('goal'), dict) else {}):
                for field, wanted in (('goal_hash', goal_hash), ('payload_sha256', payload)):
                    if isinstance(holder.get(field), str) and holder[field] != wanted:
                        value['copy_mismatches'].append('%s: %s %s != %s' % (where, field, holder[field][:12],
                                                                             str(wanted)[:12]))
            for part, wanted in (('goal', goal_hash), ('query', payload)):  # goals.jsonl's {sha256, text}
                holder = record.get(part)
                if isinstance(holder, dict) and isinstance(holder.get('sha256'), str) and holder['sha256'] != wanted:
                    value['copy_mismatches'].append('%s: %s.sha256 %s != %s' % (where, part, holder['sha256'][:12],
                                                                                str(wanted)[:12]))
        ok_goal, ok_payload = goal_hash in found, payload in found
        value['matched_goal_hash'] += ok_goal
        value['matched_payload'] += ok_payload
        value['matched'] += ok_goal and ok_payload
        if not (ok_goal and ok_payload):
            reasons.append('%s: no text hashes to its %s' % (where, ' and '.join(
                n for n, ok in (('goal_hash', ok_goal), ('payload_sha256', ok_payload)) if not ok)))
    value['unjoined_records'] = len(records) - len(used)
    if 'hash' in value['joined_by']:
        findings.append(dict(finding='JOINED_BY_HASH', turns=value['joined_by']['hash']))
    if policy == 'goal_query' and value['missing']:
        reasons.append('%d goal_query turns have no Goal-text record: %s' % (len(value['missing']), value['missing'][:5]))
    if value['bad_references']:
        reasons.append('%d runtime rows reference no Goal-text row: %s' % (len(value['bad_references']),
                                                                         value['bad_references'][:5]))
    if value['self_pair_failures']:
        reasons.append('%d sha256 fields differ from their text: %s' % (len(value['self_pair_failures']),
                                                                       value['self_pair_failures'][:5]))
    if value['copy_mismatches']:
        reasons.append('%d records name another goal_hash / payload_sha256 than their turn: %s'
                       % (len(value['copy_mismatches']), value['copy_mismatches'][:5]))
    if timing:
        value['timing'], more, extra = verify_timing(records, pairs, timing, batch, natives)
        reasons += more
        findings += extra
    for key in ('missing', 'self_pair_failures', 'copy_mismatches', 'bad_references'):
        value[key] = value[key][:50]
    value.update(status='FAIL' if reasons else 'VERIFIED', reasons=reasons[:20])
    return value


def verify_server(out, rows, policy, declared=(), timing=None, batch=None):
    """verify_rows over the files of one server output directory (unreadable files fail); the native files of the
    joined rows give the rewrite totals (timing)."""
    try:
        loaded = load_rows(candidate_files(out, declared))
    except GoalTextError as exc:
        return dict(status='FAIL', reasons=[str(exc)], policy=policy, declared=list(declared), files=[], records=0)
    return verify_rows(loaded, rows, policy, declared, timing=timing, batch=batch,
                       natives=lambda request_id: native_of(out, request_id))


def per_turn(records_by_file, rows):
    """{(session_index, turn): [records]} for the rows (pack_gswe32: the texts of one instance in turn order), joined as
    verify_rows joins them."""
    rows_by_file = as_rows(records_by_file)
    index = index_of(rows_by_file)
    keyed = any(key[0] != ROW for key in index)
    out = {}
    for row in rows:
        found = []
        for key in row_keys(row):
            if key in index:
                found = index[key]
                break
        if not found and not keyed and goal_of(row) is not None:
            goal_hash, payload = goal_of(row)
            found = [(n, p, r) for n, values in sorted(rows_by_file.items()) for p, (r, _) in enumerate(values)
                     if goal_hash in text_hashes(r) and payload in text_hashes(r)]
        if found:
            out[(row.get('session_index'), row.get('turn'))] = [dict(r, _file=n, _line=p) for n, p, r in found]
    return out
