"""Selection parity of adm-vec on recorded GH200 evidence (local only, never delivered; adapted from
D:\\Remote-RA-temp\\scratch\\adm-vec\\replay.py, which replayed the v14 pilots 1,372/1,372).

For every selective turn of every session of each given server output directory: rebuild the Page catalog from the
worker frames (phase-1 and completion Page vectors of earlier turns plus this turn's phase-1 vectors), take the
observation vector from the phase-1 frame (observation_query; random_fill has no query), and run
QuerySelectionAuthority.plan of the given source tree twice, with the vectorized path (fast_selection) and with the
reference path (`fast = None`). Checks, per turn: fast == reference (ranking, count and evidence as JSON text, so
key order, float repr and the sign of zero count), and both == the recorded ranking, selected count and full
selection evidence of the native record (scores included). The GH200's float32 projection of the query can differ
from the local one in the last bits; when the locally compacted query does not reproduce the recorded query_hash, the
float32 neighbourhood (then, if needed, the recorded 8-decimal scores) is searched for the vector whose query_hash is
the recorded one (`query_recovered`); a turn whose query cannot be recovered is counted, never compared with the
record. Read-only on the evidence.

Usage: <python with torch and numpy> -B selection_parity.py --source <tree>/source-overlay --out <json> <server dir>...
Exit 0 iff on every selective turn fast == reference, and every turn with the recorded (or recovered) query equals the
record. The verifier (verify_evidence.py --parity-python) runs it on the sealed commit's source (a fresh clone of the
sealed bundle) over the archive's selective runs.
2026-11-12 (runtime policy-registry-1112): when the replayed source has source-overlay/policy_registry.py, the selective
policies and what each replays come from its describe() (selector_v2; query_vector observation: the phase-1
observation vector, None with kind random: no query; a Goal-vector, window or rule policy is not replayed) and each
server's report names the policy's evidence_key and counts the recorded selections that carry it
(`evidence_key_recorded`); before the registry, POLICIES and the two replayed names as before.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

POLICIES = ('observation_query', 'random_fill', 'goal_query')


def load_modules(source):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(source).resolve(strict=True)))
    import torch  # noqa: F401  (before numpy: one OpenMP runtime on Windows/Anaconda)
    from remote_ra_live_v2 import query_selection, selection_contract
    from remote_ra_live_v2 import fast_selection
    return query_selection, selection_contract, fast_selection


def load_registry(source):
    """describe() of the replayed source's policy registry by name, or None (a source before policy-registry-1112)."""
    if not (Path(source) / 'policy_registry.py').is_file():
        return None
    sys.path.insert(0, str(Path(source).resolve(strict=True)))
    import policy_registry
    return {row['name']: row for row in policy_registry.describe()}


def selective_policies(registry):
    """The Selector V2 policies (POLICIES before the registry)."""
    return POLICIES if registry is None else tuple(n for n, row in registry.items() if row['selector_v2'])


def replay_plan(registry, policy):
    """(query source, evidence key) of a replayed policy: 'observation' (the phase-1 observation vector) or None (no
    query: the seeded fill); a policy outside the replay (the Goal vector, window and rule fills, a non-selective one)
    raises."""
    if registry is None:
        if policy not in ('observation_query', 'random_fill'):
            raise ValueError('policy %s is not replayed (no rewriter here)' % policy)
        return ('observation' if policy == 'observation_query' else None), None
    row = registry.get(policy)
    if row is None or not row['selector_v2'] or not (
            (row['kind'] == 'query' and row['query_vector'] == 'observation')
            or (row['kind'] == 'random' and row['query_vector'] is None)):
        raise ValueError('policy %s is not replayed (the registry describes it as %s)' % (
            policy, None if row is None else {k: row[k] for k in ('kind', 'selector_v2', 'query_vector')}))
    return row['query_vector'], row['evidence_key']


class Transcript:
    initial_length = None


def authority(qs, policy, contract, fast, session_index):
    adapter = NS(page_tokens=16, sessions={'s': NS(policy=policy)},
                 authority=NS(source='x' * 64, sign=qs.digest, verify=lambda v, m: None))
    transcript = Transcript()
    auth = qs.QuerySelectionAuthority(adapter, task=dict(instance_id='i', problem_statement='p'), contract=contract,
                                      transcript=transcript)
    auth.session_index = {'s': session_index}
    if not fast:
        auth.fast = None
    return auth, transcript


def provenance(auth, body, space, projection_hash):
    return auth.provenance_type(body['model'], body['renderer'], max(0, space.get('decoder_layers', 1) - 1),
                                'last_token', projection_hash, 128, 'int8',
                                vector_source='vllm_pre_final_norm_hidden_states')


def query_hash(policy, prov, vector):
    return hashlib.sha256(('%s:%s:%s' % (policy, prov.provenance_hash, vector.content_hash)).encode()).hexdigest()


def recover_query(auth, values, policy, body, space):
    """The GH200's compact query from the float32 neighbourhood of the exact (float64) projection (replay.py)."""
    import numpy as np
    import torch
    matrix, projection_hash = auth.projection(len(values))
    x32 = torch.tensor(auth.query_vector(values), dtype=torch.float32)
    exact = (matrix.double() @ x32.double()).numpy()
    exact = exact / np.sqrt((exact * exact).sum())
    v32 = exact.astype(np.float32)
    peak = np.float32(np.abs(v32).max())
    prov = provenance(auth, body, space, projection_hash)

    def ulps(value, k):
        out, lo, hi = [np.float32(value)], np.float32(value), np.float32(value)
        for _ in range(k):
            lo, hi = np.nextafter(lo, np.float32(-np.inf)), np.nextafter(hi, np.float32(np.inf))
            out += [lo, hi]
        return out
    for p in ulps(peak, 4):
        scale = np.float32(p / np.float32(127.0))
        options = [sorted({int(np.clip(np.round(np.float32(c / scale)), -127, 127)) for c in ulps(v, 4)}) for v in v32]
        ambiguous = [i for i, o in enumerate(options) if len(o) > 1]
        if len(ambiguous) > 12:
            continue
        base = [o[0] for o in options]
        for combo in itertools.product(*[options[i] for i in ambiguous]):
            q = list(base)
            for i, c in zip(ambiguous, combo):
                q[i] = c
            vec = auth.compact_type(tuple(q), 'int8', float(scale))
            if query_hash(policy, prov, vec) == body['query_hash']:
                return vec
    return None


def recover_query_by_scores(auth, fast_selection, values, policy, body, space, entries):
    """Second attempt (replay.py --scores): GEMV error bounds give the ambiguous int8 elements, the recorded scores
    pick them, then the float32 scale is searched for the recorded query_hash. None, or 'scores-only'."""
    import numpy as np
    import torch
    scores = body['selection'].get('scores')
    if not scores:
        return None
    matrix, projection_hash = auth.projection(len(values))
    x = torch.tensor(auth.query_vector(values), dtype=torch.float32).double()
    m = matrix.double()
    raw = (m @ x).numpy()
    bound = (m.abs() @ x.abs()).numpy()
    norm = float(np.sqrt((raw * raw).sum()))
    v = raw / norm
    delta = 4e-6 * bound / norm + 4e-6 * np.abs(v)
    s0 = float(np.abs(v).max()) / 127.0
    lo = np.minimum((v - delta) / (s0 * (1 + 1e-5)), (v - delta) / (s0 * (1 - 1e-5)))
    hi = np.maximum((v + delta) / (s0 * (1 - 1e-5)), (v + delta) / (s0 * (1 + 1e-5)))
    qlo, qhi = np.clip(np.round(lo), -127, 127).astype(int), np.clip(np.round(hi), -127, 127).astype(int)
    options = [list(range(a, b + 1)) for a, b in zip(qlo.tolist(), qhi.tolist())]
    ambiguous = [i for i, o in enumerate(options) if len(o) > 1]
    if len(ambiguous) > 18:
        return None
    unit = auth.vectors.CompactVector.unit_values
    n = len(entries)
    columns = np.array([unit(entries[p]) for p in range(n)], dtype=np.float64).T.copy()
    want = np.array([scores[str(p)] for p in range(n)])
    probe = np.random.default_rng(0).choice(n, size=min(64, n), replace=False)
    sub = columns[:, probe].copy()
    base, matches = [o[0] for o in options], []
    for combo in itertools.product(*[options[i] for i in ambiguous]):
        q = list(base)
        for i, c in zip(ambiguous, combo):
            q[i] = c
        if not any(q):
            continue
        vec = auth.compact_type(tuple(q), 'int8', float(np.float32(s0)))
        got = fast_selection.cosines(np, sub, unit(vec))
        if all(round(a, 8) == b for a, b in zip(got.tolist(), want[probe].tolist())):
            full = fast_selection.cosines(np, columns, unit(vec))
            if all(round(a, 8) == b for a, b in zip(full.tolist(), want.tolist())):
                matches.append(q)
    prov = provenance(auth, body, space, projection_hash)
    start = np.float32(s0)
    for q in matches:
        lo_s = hi_s = start
        for k in range(4000):
            for s in ((start,) if k == 0 else (lo_s, hi_s)):
                vec = auth.compact_type(tuple(q), 'int8', float(s))
                if query_hash(policy, prov, vec) == body['query_hash']:
                    return vec
            lo_s, hi_s = np.nextafter(lo_s, np.float32(-np.inf)), np.nextafter(hi_s, np.float32(np.inf))
    return 'scores-only' if matches else None


def replay_server(modules, out, registry=None):
    """Every selective turn of one server output directory; returns (stats, examples)."""
    qs, sc, fast_selection = modules
    schema = qs.upstream()[1]
    out = Path(out)
    ready = json.loads((out / 'ready.json').read_bytes())
    policy = ready['policy']
    contract = dict(ready['selection_contract'])
    try:
        query_source, evidence_key = replay_plan(registry, policy)
    except ValueError as exc:
        raise ValueError('%s: %s' % (out.name, exc)) from None
    stats, examples = defaultdict(int), []
    stats['fast_path_available'] = int(fast_selection.available(qs.upstream()[0]) is not None)
    # perf-swe16: a real-agent pool server holds ~2,000 native receipts of up to ~1 MB each; index them by the runtime
    # rows (session, turn, request id) and load one session's receipts at a time, not all at once.
    index = defaultdict(list)
    with (out / 'runtime.jsonl').open('rb') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                index[row['session_id']].append((row['turn'], row['session_index'], row['request_id']))
                del row
    for sid in sorted(index):
        entries = sorted(index[sid])
        pair = [authority(qs, policy, contract, f, entries[0][1]) for f in (True, False)]
        catalog = {}
        for _, _, request_id in entries:
            v = json.loads((out / ('native-' + hashlib.sha256(request_id.encode()).hexdigest() + '.json')).read_bytes())
            body = v['query']['body'] if v.get('query') else None
            if v['terminal_state'] != 'complete' or body is None:
                stats['skipped_not_complete'] += 1
                continue
            local, n = v['local_turn'], body['candidate_pages']
            if local == 0:
                catalog = {}
                for auth, _ in pair:
                    auth.reset_session('s')
            frames = {f['operation']: f for f in v['worker_frames'] if f['operation'] in ('phase1', 'complete')}
            p1 = frames.get('phase1')
            sel = body['selection']
            for auth, transcript in pair:
                transcript.initial_length = sel['initial_prompt_end']
                auth.turn_ends.setdefault('s', [])
            fresh = {int(k): w for k, w in (p1['page_vectors'] if p1 else {}).items()}
            if sel['reason'] is None:
                query, recovered = (p1['observation_vector'] if query_source == 'observation' else None), None
                if query is not None:
                    auth0 = pair[0][0]
                    matrix, projection_hash = auth0.projection(len(query))
                    local_vec = sc.compact(auth0.query_vector(query), matrix, auth0.compact_type)
                    prov = provenance(auth0, body, p1['vector_space'], projection_hash)
                    if query_hash(policy, prov, local_vec) != body['query_hash']:
                        recovered = recover_query(auth0, query, policy, body, p1['vector_space'])
                        if recovered is None:
                            known = {p: auth0.page_vector(w) for p, w in catalog.items() if p < n}
                            known.update({p: auth0.page_vector(w) for p, w in fresh.items() if p < n})
                            recovered = recover_query_by_scores(auth0, fast_selection, query, policy, body,
                                                                p1['vector_space'], known)
                            if recovered == 'scores-only':
                                stats['query_int8_from_scores_scale_not_found'] += 1
                                recovered = None
                            elif recovered is not None:
                                stats['query_recovered_from_scores'] += 1
                        stats['query_recovered' if recovered is not None else 'query_not_recovered'] += 1
                results = []
                for auth, _ in pair:
                    entries = {p: auth.page_vector(w) for p, w in catalog.items() if p < n}
                    entries.update({p: auth.page_vector(w) for p, w in fresh.items() if p < n})
                    if set(entries) != set(range(n)):
                        raise ValueError('%s %s turn %s: %d Page vectors for %d candidates' % (out.name, sid, v['turn'],
                                                                                            len(entries), n))
                    b = NS(session='s', turn=body['turn'], model=body['model'], renderer=body['renderer'],
                           stable_end=v['stable_end'])
                    goal = schema.GoalRecord(session_id='s', turn_id=local + 1, constructor='replay', constructor_config={},
                                             text='replay')
                    context = schema.TurnContext('s', 'i', local + 1, 'objective', None, None, 'observation')
                    q = None if query_source is None else auth.query_vector(query)
                    space = None if query_source is None else p1['vector_space']
                    if recovered is not None:
                        q = auth.compact_type(recovered.values, 'int8', recovered.scale)
                        space = dict(space, hidden_size=len(query))
                    results.append(auth.plan(b, goal, context, entries, q, n, space, local))
                fast, ref = (json.dumps([list(r[0]), r[1], r[2]]) for r in results)
                stats['selective_turns'] += 1
                stats['max_pages'] = max(stats['max_pages'], n)
                if fast == ref:
                    stats['fast_equals_reference'] += 1
                else:
                    stats['fast_differs_from_reference'] += 1
                    examples.append(dict(kind='fast!=reference', session=sid, turn=v['turn']))
                ranking, count, evidence = results[0]
                rec, mine = json.loads(json.dumps(dict(sel))), json.loads(json.dumps(evidence))
                if evidence_key:
                    stats['evidence_key_recorded'] += rec.get(evidence_key) is not None
                same_query = mine.get('query_hash') == rec.get('query_hash')
                stats['query_hash_matches'] += same_query
                if list(ranking) == body['ranking'] and count == body['selected_count'] and mine == rec:
                    stats['equals_recorded'] += 1
                else:
                    stats['differs_from_recorded%s' % ('' if same_query else '_query_hash_differs')] += 1
                    if same_query and len(examples) < 10:
                        examples.append(dict(kind='recorded', session=sid, turn=v['turn'],
                                             keys=sorted(k for k in set(mine) | set(rec) if mine.get(k) != rec.get(k))))
            else:
                stats['inactive_turns'] += 1
            for auth, _ in pair:
                auth.turn_ends['s'].append(v['stable_end'])
                auth.previous_active['s'] = (frozenset(body['ranking'][:body['selected_count']]), n)
                auth.ever_selected.setdefault('s', set()).update(body['ranking'][:body['selected_count']])
            done = frames.get('complete')
            catalog.update(fresh)
            catalog.update({int(k): w for k, w in (done['page_vectors'] if done else {}).items()})
    return dict(stats, policy=policy, **({} if registry is None else dict(evidence_key=evidence_key))), examples


def verdict(stats):
    """fast == reference on every selective turn, and every turn whose query hash matched equals the record."""
    turns = stats.get('selective_turns', 0)
    return (turns > 0 and stats.get('fast_equals_reference', 0) == turns and not stats.get('differs_from_recorded', 0)
            and stats.get('equals_recorded', 0) + stats.get('differs_from_recorded_query_hash_differs', 0) == turns)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='source-overlay of the runtime tree to replay with')
    parser.add_argument('--out', required=True, help='new JSON report')
    parser.add_argument('servers', nargs='+', help='server output directories (job-output/<label>)')
    args = parser.parse_args(argv)
    modules = load_modules(args.source)
    registry = load_registry(args.source)
    report = dict(source=str(Path(args.source).resolve()), servers={})
    if registry is not None:  # (the report of a source before the registry is unchanged)
        report.update(policy_registry=True, selective_policies=list(selective_policies(registry)))
    ok = True
    for server in args.servers:
        stats, examples = replay_server(modules, server, registry)
        passed = verdict(stats)
        ok = ok and passed
        report['servers'][Path(server).name] = dict(stats=stats, examples=examples[:10], status='PASS' if passed else 'FAIL')
        print('PARITY %s %s %s' % (Path(server).name, 'PASS' if passed else 'FAIL', json.dumps(stats, sort_keys=True)))
    report['status'] = 'PASS' if ok else 'FAIL'
    with Path(args.out).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, sort_keys=True, indent=1)
        stream.write('\n')
    print('SELECTION_PARITY_%s' % report['status'])
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
