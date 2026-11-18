"""Per-step timing tables from the REMOTE_RA_STEP lines of server logs.

The servers ran with REMOTE_RA_STEP_PROFILE=1 (source-overlay/remote_ra_step_profile.py;
schema remote-ra-step-profile-v1). One server log = one label: ``server-<label>.log``
gives ``<label>``, any other file its stem. A log's ``total`` line (written at server
stop) is used when present; else its windows are merged (counts and sums exact,
p50/p95 the step-weighted mean of the windows' percentiles, marked ``~``).

For each label it prints, per batch size b (requests scheduled in the step), the
mean milliseconds per step of every component over all steps of the group, so
children add up to their parent; ``(other)`` under a parent is the parent minus
its named children (code that has no timer of its own); ``unaccounted`` is the
loop minus every named leaf. ``gpu.forward`` (CUDA events around the model
forward) and ``gc`` (Python collections, wherever they ran) are outside the tree.

``TREE`` is the profiler's own tree (remote_ra_step_profile.TREE; test_step_profile
checks they are equal). A component it does not name hangs under its longest dotted
prefix that is a component or an RPC (``rpc.<op>.worker.scan`` under
``rpc.<op>.worker`` under ``rpc.<op>``), else under ``loop``. The tree of the
earlier diagnostic profile (ms2-step-profile) used fewer names; its logs read the
same way.

The probe-only passes (phase 1 and the Goal forward, kept out of the steps) follow each
label's table, one column per (kind, graph mode, token bin), in milliseconds per pass
(PROBE_TREE; ``--probe-csv`` writes their long form).

The dispatch table follows them: one row per (phase, b, tag, mode), mode the CUDA-graph mode
vLLM's dispatch chose for the steps (FULL, PIECEWISE, NONE; '-' in a log that has none), with
the steps, the mean tokens scheduled and their padded size, and the step, the host forward and
the device forward in milliseconds per step, so the first step after an admission (tag admit)
reads against a plain decode step of the same b (``--dispatch-csv`` writes it). Tags: admit
(an admission), cancel and close (a Cancel / CloseSession pass; tagged admit by profiles
written before the split), finish, probe.

usage: python step_profile.py LOG_OR_DIR [LOG_OR_DIR ...] [--phase decode] [--tag ""]
                              [--csv FILE] [--markdown FILE] [--probe-csv FILE] [--dispatch-csv FILE]
"""
import argparse
import json
import math
from pathlib import Path
import sys

PREFIX = 'REMOTE_RA_STEP '
COLS = ('n', 'calls', 'sum', 'mean', 'p50', 'p95')

# (component, parent): the host-time tree of one engine step (remote_ra_step_profile.TREE).
TREE = (
    ('loop', None),
    ('input', 'loop'),
    ('input.poll', 'input'),
    ('input.tick', 'input'),
    ('input.settle', 'input'),
    ('input.resume', 'input'),
    ('input.registered', 'input'),
    ('step', 'loop'),
    ('step.schedule', 'step'),
    ('sched.allocate', 'step.schedule'),
    ('sched.allocate.validate', 'sched.allocate'),
    ('sched.allocate.grow', 'sched.allocate'),
    ('rpc.allocation_update', 'sched.allocate'),
    ('rpc.allocation_update.worker', 'rpc.allocation_update'),
    ('rpc.allocation_update.worker.scan', 'rpc.allocation_update.worker'),
    ('rpc.allocation_update.worker.command', 'rpc.allocation_update.worker'),
    ('sched.split', 'step.schedule'),
    ('rpc.allocation_batch', 'step.schedule'),
    ('rpc.allocation_batch.worker', 'rpc.allocation_batch'),
    ('rpc.allocation_batch.worker.scan', 'rpc.allocation_batch.worker'),
    ('rpc.allocation_batch.worker.command', 'rpc.allocation_batch.worker'),
    ('rpc.allocation_batch.rowless', 'rpc.allocation_batch'),
    ('step.phase1', 'step'),
    ('rpc.phase1_arm', 'step.phase1'),
    ('rpc.phase1_collect', 'step.phase1'),
    ('step.execute', 'step'),
    ('worker.execute', 'step.execute'),
    ('worker.guard_pre', 'worker.execute'),
    ('native.execute', 'worker.execute'),
    ('native.update_states', 'native.execute'),
    ('worker.rows', 'native.execute'),
    ('native.prepare_inputs', 'native.execute'),
    ('native.mamba', 'native.execute'),
    ('native.attn_metadata', 'native.execute'),
    ('native.preprocess', 'native.execute'),
    ('worker.before', 'native.execute'),
    ('worker.before.claims', 'worker.before'),
    ('worker.before.claims.observe', 'worker.before.claims'),
    ('worker.before.claims.increment', 'worker.before.claims'),
    ('worker.before.materialize', 'worker.before'),
    ('worker.before.positions', 'worker.before'),
    ('worker.before.qwen', 'worker.before'),
    ('worker.before.probe', 'worker.before'),
    ('worker.forward', 'native.execute'),
    ('worker.after', 'native.execute'),
    ('worker.after.finite', 'worker.after'),
    ('worker.after.probe', 'worker.after'),
    ('worker.after.qwen', 'worker.after'),
    ('worker.after.qwen.drain', 'worker.after.qwen'),
    ('worker.after.qwen.compact', 'worker.after.qwen'),
    ('worker.after.qwen.capture', 'worker.after.qwen'),
    ('worker.guard_post', 'worker.execute'),
    ('worker.finish', 'worker.guard_post'),
    ('worker.finish.finite', 'worker.finish'),
    ('worker.finish.qwen', 'worker.finish'),
    ('worker.finish.qwen.drain', 'worker.finish.qwen'),
    ('worker.finish.qwen.compact', 'worker.finish.qwen'),
    ('worker.finish.qwen.capture', 'worker.finish.qwen'),
    ('step.sample', 'step'),
    ('worker.sample', 'step.sample'),
    ('step.update', 'step'),
    ('sched.stop', 'step.update'),
    ('sched.complete', 'step.update'),
    ('sched.complete.drain', 'sched.complete'),
    ('rpc.complete', 'sched.complete.drain'),
    ('rpc.complete.copy', 'rpc.complete'),
    ('sched.complete.publish', 'sched.complete.drain'),
    ('sched.complete.after', 'sched.complete'),
    ('sched.complete.transcript', 'sched.complete.after'),
    ('sched.complete.terminal', 'sched.complete'),
    ('sched.complete.terminal.config', 'sched.complete.terminal'),
    ('sched.complete.terminal.copy', 'sched.complete.terminal'),
    ('sched.complete.terminal.goal', 'sched.complete.terminal'),
    ('sched.complete.terminal.save', 'sched.complete.terminal'),
    ('sched.complete.terminal.encode', 'sched.complete.terminal.save'),
    ('sched.complete.terminal.snapshot', 'sched.complete.terminal'),
    ('sched.free', 'step.update'),
    ('rpc.cancel', 'sched.free'),
    ('step.misc', 'step'),
    ('post', 'loop'),
)
PARENT = dict(TREE)
ORDER = [name for name, _ in TREE]
OUTSIDE = ('gpu.forward', 'gc')
# (component, parent) of one probe-only pass (remote_ra_step_profile.PROBE_TREE): each line's
# ``probes``, grouped by (kind, graph, bin); values per pass.
PROBE_TREE = (
    ('probe.rpc', None),
    ('probe.pass', 'probe.rpc'),
    ('probe.execute', 'probe.pass'),
    ('worker.execute', 'probe.execute'),
    ('worker.guard_pre', 'worker.execute'),
    ('native.execute', 'worker.execute'),
    ('native.update_states', 'native.execute'),
    ('worker.rows', 'native.execute'),
    ('native.prepare_inputs', 'native.execute'),
    ('probe.dispatch', 'native.execute'),
    ('native.mamba', 'native.execute'),
    ('native.attn_metadata', 'native.execute'),
    ('native.preprocess', 'native.execute'),
    ('worker.before', 'native.execute'),
    ('worker.before.positions', 'worker.before'),
    ('probe.restore', 'worker.before'),
    ('worker.forward', 'native.execute'),
    ('worker.after', 'native.execute'),
    ('probe.after', 'worker.after'),
    ('probe.capture', 'probe.after'),
    ('probe.tail', 'probe.after'),
    ('worker.guard_post', 'worker.execute'),
    ('probe.cleanup', 'probe.pass'),
    ('probe.recover', 'probe.pass'),
    ('probe.compact', 'probe.rpc'),
    ('probe.receipt', 'probe.rpc'),
    ('readd.update_states', None),
    ('readd.prepare_inputs', None),
)
PROBE_OUTSIDE = ('gpu.forward',)
PROBE_COUNTS = ('forwards', 'tokens', 'padded', 'graphed', 'rows', 'readds')
# Per-step values of a line's ``dispatch`` groups (remote_ra_step_profile.DISPATCH).
DISPATCH = ('tokens', 'padded', 'loop', 'step', 'step.execute', 'worker.forward', 'gpu.forward')
PROBE_NOTES = {
    'probe.rpc': 'the worker phase-1 RPC (model_queries.phase1; goal_forward for a separate Goal forward)',
    'probe.pass': 'the driver (NativeQueryForward.phase1 / goal_forward): forwards, cleanup, drain',
    'probe.execute': 'one forward: the runner pass with the guard (NativeQueryForward._execute)',
    'probe.dispatch': "vLLM's CUDA-graph dispatch and padding (the mode is the group's graph)",
    'probe.restore': 'the committed carry restored into the probe slot (backend.restore_carry)',
    'probe.after': 'the probe rows: closing rows, query row to the host, finiteness, state capture',
    'probe.capture': 'the phase-1 GDN state capture (backend.capture_phase1_state)',
    'probe.tail': 'the tail Page closed in phase 1 copied to the host (io.capture)',
    'probe.cleanup': 'the runner pass that removes the probe between forwards (outside the drain)',
    'probe.recover': 'the drain: cleanup pass, device drains, the drain receipt',
    'probe.compact': 'the Page vectors compacted (PageCompactor.compact)',
    'probe.receipt': 'the phase-1 receipt signed and verified',
    'readd.update_states': "the next step's _update_states: it re-adds the rows the pass took out",
    'readd.prepare_inputs': "the next step's _prepare_inputs",
    'gpu.forward': 'CUDA events around the forward (device time)',
}
NOTES = {
    'loop': 'one busy-loop iteration (from the end of the input pass if the engine was idle)',
    'input': 'input pass (_process_input_queue): poll, lifecycle tick, client requests',
    'input.poll': 'native authority poll',
    'input.tick': 'dynamic lifecycle tick (sweeps)',
    'input.settle': '--phase1-in-step: phase-1 works run in place (nested records dropped)',
    'input.resume': 'a goal_query admission continued after its Goal (nested records dropped)',
    'input.registered': 'registered command frames on the engine thread: an admission (tag admit), a Cancel '
                        '(cancel) or a CloseSession (close); nested records dropped',
    'step.schedule': 'scheduler.schedule',
    'sched.allocate': 'EngineCoreAdapter.allocate_slots, per scheduled request',
    'sched.allocate.validate': 'validate_native_allocation + validate_interval',
    'sched.allocate.grow': 'EngineRpcBoundary._grow: the decoupled row only grows',
    'rpc.allocation_update': 'EngineWorkerProxy.call allocation_update (UniProc: the worker inline)',
    'rpc.allocation_update.worker': 'evidence.measured_command on the worker',
    'rpc.allocation_update.worker.scan': 'counters to command: the scan over backend.owner.claims',
    'rpc.allocation_update.worker.command': 'the allocation owner command',
    'sched.split': 'EngineBridge.scheduler_split',
    'step.phase1': '--phase1-in-step: Phase1Steps.before/after (arm and collect RPCs)',
    'step.execute': 'model_executor.execute_model (UniProc: the worker runs inline)',
    'worker.execute': 'runner.execute_model incl. the generation guard',
    'worker.guard_pre': 'guard before the native call: enter_execute, connector check, device drain (sync)',
    'native.execute': "the runner's own execute_model",
    'native.update_states': 'runner._update_states',
    'worker.rows': 'worker_rows.install_before_prepare: each request row (incremental)',
    'native.prepare_inputs': 'runner._prepare_inputs',
    'native.mamba': 'mamba_utils.preprocess_mamba',
    'native.attn_metadata': 'runner._build_attention_metadata',
    'native.preprocess': 'runner._preprocess',
    'worker.before': 'hooks.before_forward (WorkerBridge.before_forward)',
    'worker.before.claims': 'owner.step: observe and claim every request of the step',
    'worker.before.claims.observe': 'owner._observe, per request',
    'worker.before.claims.increment': 'owner._claim_increment (incremental claims)',
    'worker.before.materialize': 'owner.materialize, per request (first-step zero/restore)',
    'worker.before.positions': 'positions.text_positions, per request (positions .tolist(): device read)',
    'worker.before.qwen': 'QwenAlignLifecycle.before_forward, per request',
    'worker.before.probe': "--phase1-in-step: the probe row's positions and carry restore (probe_before_forward)",
    'worker.forward': 'model forward, host launch wall (graph replay or eager launches)',
    'worker.after': 'hooks.after_forward (WorkerBridge.after_forward)',
    'worker.after.finite': 'isfinite over the step rows + .cpu(): the first device join after the forward',
    'worker.after.probe': "--phase1-in-step: the probe row's rows, vectors and state capture (probe_after_forward)",
    'worker.after.qwen': 'QwenAlignLifecycle.after_forward, per request',
    'worker.after.qwen.drain': 'CudaWorkerBackend.forward_complete: device drain (sync) unless the hand-off',
    'worker.after.qwen.compact': 'PageCompactor.compact: Page vectors where Pages close (decoupled)',
    'worker.after.qwen.capture': 'capture_latest_hybrid: latest GDN at an M-aligned end',
    'worker.guard_post': 'guard after the native call: leave_execute (device drain unless the hand-off)',
    'worker.finish': '--decode-fastpath2: WorkerBridge.finish_step, the after-forward work after the logits launch',
    'worker.finish.finite': 'wait for the step finiteness read (the forward join; logits may still run)',
    'worker.finish.qwen': 'QwenAlignLifecycle.after_forward, per request (deferred)',
    'worker.finish.qwen.drain': 'CudaWorkerBackend.forward_complete (already joined: no drain)',
    'worker.finish.qwen.compact': 'PageCompactor.compact: Page vectors where Pages close (deferred)',
    'worker.finish.qwen.capture': 'capture_latest_hybrid: latest GDN at an M-aligned end (deferred)',
    'rpc.allocation_batch': '--decode-fastpath2: the step allocation updates in one RPC (scheduler.schedule end)',
    'rpc.allocation_batch.worker': 'evidence.measured_command per batch entry on the worker',
    'rpc.allocation_batch.worker.scan': 'counters to command (per-request claim counts)',
    'rpc.allocation_batch.worker.command': 'the allocation owner command per entry',
    'rpc.allocation_batch.rowless': '--decode-fastpath3: decode_fastpath3.rowless_entry per batch entry (an entry '
                                    'without rows: its update and frame, no command; with rows: returns at once)',
    'step.sample': 'model_executor.sample_tokens (logits/sampling, token D2H)',
    'step.update': 'scheduler.update_from_output',
    'sched.stop': 'MarkerStop: decode the output so far, look for the marker',
    'sched.complete': 'completion: seal, publish, drain (lifecycle.before_success)',
    'sched.complete.drain': 'LiveEndpoint.drain: the complete RPC, the engine checks and commit',
    'rpc.complete': 'complete RPC: the worker drain and Page publication of a finished request',
    'rpc.complete.copy': 'the JSON copy of the complete receipt (EngineWorkerProxy.call)',
    'sched.complete.publish': 'query policies: the turn\'s Page vectors published (publish_completion)',
    'sched.complete.after': 'LiveEndpoint.after_drain: transcript commit, idle-release order',
    'sched.complete.transcript': 'commit_transcript: the turn\'s tokens into the engine transcript',
    'sched.complete.terminal': 'evidence.native_terminal: the native file (and goals.jsonl row)',
    'sched.complete.terminal.config': 'the startup config read, hashed and checked again (load)',
    'sched.complete.terminal.copy': 'the JSON copy of the drain receipt (copy_receipt)',
    'sched.complete.terminal.goal': 'goal_query: the goals.jsonl row encoded and appended (fsync)',
    'sched.complete.terminal.save': 'the native file: encode, write, fsync, SHA-256',
    'sched.complete.terminal.encode': 'its JSON encoding (encoded)',
    'sched.complete.terminal.snapshot': '--deferred-completion-io: the record\'s shared parts encoded now; '
                                        'the writer thread does the rest (completion_writer)',
    'sched.free': 'scheduler._free_request (native and lifecycle free)',
    'rpc.cancel': 'cancel RPC (a cancelled request; an abort in the input pass lands here too)',
    'rpc.phase1_arm': '--phase1-in-step: arm the probe row',
    'rpc.phase1_collect': '--phase1-in-step: collect the probe row',
    'step.misc': 'grammar bitmask, future, aborts, classification',
    'post': 'output queue, post_step',
    'gpu.forward': 'CUDA events around the forward (device time; not in the host tree)',
    'gc': 'Python GC pauses (any thread; already inside the components)',
}

def label_of(path):
    stem = Path(path).stem
    return stem[len('server-'):] if stem.startswith('server-') else stem


def lines_of(path):
    with open(path, encoding='utf-8', errors='replace') as stream:
        for line in stream:
            at = line.find(PREFIX)
            if at < 0:
                continue
            try:
                yield json.loads(line[at + len(PREFIX):])
            except ValueError:
                continue


def read_log(path):
    """{label, path, install, windows, total} of one server log (several engine pids: the last)."""
    install, windows, total = None, [], None
    for value in lines_of(path):
        kind = value.get('kind')
        if kind == 'install':
            install, windows, total = value, [], None     # a new engine process starts over
        elif kind == 'window':
            windows.append(value)
        elif kind == 'total':
            total = value
    return dict(label=label_of(path), path=str(path), install=install, windows=windows, total=total)


def _stat(raw, cols):
    return dict(zip(cols, raw))


def _groups(value):
    cols = value.get('cols') or list(COLS)
    return {(g['phase'], g['b'], g['tag']): dict(steps=g['steps'], c={k: _stat(v, cols) for k, v in g['c'].items()})
            for g in value.get('groups') or ()}


def _probes(value):
    """A line's probe-only pass groups: {(kind, graph, bin): dict(steps=passes, counts, c)}."""
    cols = value.get('cols') or list(COLS)
    return {(g['kind'], g['graph'], g['bin']): dict(steps=g['passes'], counts=dict(g.get('counts') or {}),
                                                    c={k: _stat(v, cols) for k, v in g['c'].items()})
            for g in value.get('probes') or ()}


def _dispatch(value):
    """A line's dispatch groups: {(phase, b, tag, mode): dict(steps, c)}."""
    cols = value.get('cols') or list(COLS)
    return {(g['phase'], g['b'], g['tag'], g['mode']): dict(steps=g['steps'],
                                                           c={k: _stat(v, cols) for k, v in g['c'].items()})
            for g in value.get('dispatch') or ()}


def merge_probes(parts):
    """merge() of probe groups, their counts added."""
    out = merge(parts)
    for key, target in out.items():
        target['counts'] = {name: sum((p[key].get('counts') or {}).get(name, 0) for p in parts if key in p)
                            for name in PROBE_COUNTS}
    return out


def merge(parts):
    """Merge group stats: counts and sums exact; p50/p95 the n-weighted mean (approximate)."""
    out = {}
    for groups in parts:
        for key, group in groups.items():
            target = out.setdefault(key, dict(steps=0, c={}, approx=False))
            target['steps'] += group['steps']
            for name, s in group['c'].items():
                t = target['c'].setdefault(name, dict(n=0, calls=0, sum=0.0, p50w=0.0, p95w=0.0))
                t['n'] += s['n']
                t['calls'] += s['calls']
                t['sum'] += s['sum']
                t['p50w'] += s['p50'] * s['n']
                t['p95w'] += s['p95'] * s['n']
    for key, target in out.items():
        for name, t in target['c'].items():
            n = t['n'] or 1
            t.update(mean=t['sum'] / n, p50=t.pop('p50w') / n, p95=t.pop('p95w') / n)
    return out


def faults(log):
    """Profiler faults the runtime counted (errors, the last one): the largest count seen on any line."""
    values = [v for v in [log['install'], log['total']] + log['windows'] if v]
    worst = max(values, key=lambda v: v.get('errors') or 0, default={})
    return dict(errors=worst.get('errors') or 0, last_error=worst.get('last_error'))


def profile(path):
    """One server's groups: from its total line, else its merged windows."""
    log = read_log(path)
    log = dict(log, faults=faults(log))
    if log['total'] is not None:
        total = log['total']
        groups = _groups(total)
        for group in groups.values():
            group['approx'] = False
        probes = _probes(total)
        for group in probes.values():
            group['approx'] = False
        dispatch = _dispatch(total)
        for group in dispatch.values():
            group['approx'] = False
        return dict(log, source='total', groups=groups, steps=total['steps'], hist=total.get('hist') or {},
                    idle=total.get('idle') or {}, probes=probes, dispatch=dispatch)
    windows = log['windows']
    groups = merge([_groups(w) for w in windows])
    probes = merge_probes([_probes(w) for w in windows])
    for group in probes.values():
        group['approx'] = len(windows) > 1
    dispatch = merge([_dispatch(w) for w in windows])
    for group in dispatch.values():
        group['approx'] = len(windows) > 1
    for group in groups.values():
        group['approx'] = len(windows) > 1
    hist = {}
    for w in windows:
        for b, n in (w.get('hist') or {}).items():
            hist[b] = hist.get(b, 0) + n
    idle = dict(n=sum((w.get('idle') or {}).get('n', 0) for w in windows),
                ms=round(sum((w.get('idle') or {}).get('ms', 0) for w in windows), 3))
    return dict(log, source='windows' if windows else 'none', groups=groups,
                steps=sum(w.get('steps', 0) for w in windows), hist=hist, idle=idle, probes=probes,
                dispatch=dispatch)


def select(groups, phase, tag):
    """Groups of one phase by b; ``tag`` None merges every tag of a b into one group (steps,
    counts and sums added; p50/p95 the n-weighted mean, so marked approximate)."""
    picked = {}
    for (p, b, t), group in groups.items():
        if p == phase and (tag is None or t == tag):
            # Every tag under one key: merge() adds parts of the same key only.
            picked.setdefault(b, []).append({(p, b, '*' if tag is None else t): group})
    out = {}
    for b, parts in sorted(picked.items()):
        if len(parts) == 1:
            out[b] = next(iter(parts[0].values()))
        else:
            (merged,) = merge(parts).values()
            out[b] = dict(merged, approx=True)
    return out


def declared_parent(name):
    """TREE's parent; else the longest dotted prefix that is a component or an RPC; else None."""
    if name in PARENT:
        return PARENT[name]
    head = name
    while '.' in head:
        head = head.rpartition('.')[0]
        if head in PARENT or head.startswith('rpc.'):
            return head
    return None


def _parent(name, present):
    """Nearest recorded ancestor (a component whose own parent has no timer hangs one level up);
    a name with no declared parent hangs under ``loop``."""
    parent = declared_parent(name)
    while parent is not None and parent not in present:
        parent = declared_parent(parent)
    if parent is None and name != 'loop' and name not in PARENT and 'loop' in present:
        return 'loop'
    return parent


def rows(group):
    """The tree of one group: [dict(key, name, depth, ms, stat)], ms = per-step mean over all its steps.

    ``(other)`` rows (key ``(other):<parent>``) follow their parent's children;
    ``unaccounted`` (loop minus every leaf) and the components outside the tree end it.
    """
    steps = group['steps'] or 1
    c = group['c']
    present = [n for n in ORDER if n in c] + sorted(n for n in c if n not in PARENT and n not in OUTSIDE)
    per = {n: c[n]['sum'] / steps for n in c}
    parent = {n: _parent(n, set(present)) for n in present}
    tree = []

    def emit(name, depth):
        tree.append(dict(key=name, name=name, depth=depth, ms=per[name], stat=c[name]))
        children = [k for k in present if parent[k] == name]
        for k in children:
            emit(k, depth + 1)
        if children:
            tree.append(dict(key='(other):' + name, name='(other)', depth=depth + 1,
                             ms=per[name] - sum(per[k] for k in children), stat=None))
    for name in present:
        if parent[name] is None:
            emit(name, 0)
    leaves = [n for n in present if n != 'loop' and not any(parent[k] == n for k in present)]
    if 'loop' in c:
        tree.append(dict(key='unaccounted', name='unaccounted', depth=0,
                         ms=per['loop'] - sum(per[n] for n in leaves), stat=None))
    for name in OUTSIDE:
        if name in c:
            tree.append(dict(key=name, name=name, depth=0, ms=per[name], stat=c[name]))
    return tree


def calls_per_step(stat, steps):
    return stat['calls'] / steps if stat and steps else None


def matrix(by_b):
    """Row order (the union of every b's tree, in tree order) and each b's rows by key."""
    trees = {b: rows(g) for b, g in by_b.items()}
    order = []
    for tree in sorted(trees.values(), key=len, reverse=True):
        for row in tree:
            if all(row['key'] != r['key'] for r in order):
                order.append(row)
    return order, {b: {r['key']: r for r in tree} for b, tree in trees.items()}


def text_table(prof, phase='decode', tag=''):
    """Fixed-width table: one column per b (ms per step; [calls/step] where not 1, {steps with it})."""
    by_b = select(prof['groups'], phase, tag)
    if not by_b:
        return '  (no %s steps%s)\n' % (phase, '' if tag is None else ' with tag %r' % tag)
    order, cells = matrix(by_b)
    width = max(len('  ' * r['depth'] + r['name']) for r in order) + 2
    head = '%-*s' % (width, 'component (ms per step)') + ''.join('%24s' % ('b=%d (%d steps)' % (b, g['steps']))
                                                              for b, g in by_b.items())
    out = [head]
    for row in order:
        line = []
        for b in by_b:
            hit = cells[b].get(row['key'])
            if hit is None:
                line.append('%24s' % '-')
                continue
            extra, stat, steps = '', hit['stat'], by_b[b]['steps']
            if stat is not None:
                cps = calls_per_step(stat, steps)
                if cps is not None and abs(cps - 1) > 1e-9:
                    extra = ' [%.2fx]' % cps
                if stat['n'] < steps:
                    extra += ' {%d}' % stat['n']
            line.append('%24s' % ('%.3f%s' % (hit['ms'], extra)))
        out.append('%-*s' % (width, '  ' * row['depth'] + row['name']) + ''.join(line))
    tail = []
    for b, g in by_b.items():
        loop = g['c'].get('loop')
        tail.append('%24s' % ('%s%.3f/%.3f' % ('~' if g.get('approx') else '', loop['p50'], loop['p95'])
                              if loop else '-'))
    out.append('%-*s' % (width, 'loop p50/p95 (ms)') + ''.join(tail))
    return '\n'.join(out) + '\n'


def dispatch_rows(prof):
    """One row per (phase, b, tag, mode) of the dispatch groups, by b, phase (decode first), tag and mode:
    steps, the mean tokens scheduled and padded size, and loop, step, step.execute, worker.forward and
    gpu.forward in ms (each the mean over the steps that have it; n_gpu the steps with a device time)."""
    out = []
    for (phase, b, tag, mode), g in sorted((prof.get('dispatch') or {}).items(),
                                           key=lambda kv: (kv[0][1], kv[0][0] != 'decode', kv[0][0], kv[0][2],
                                                           kv[0][3])):
        c = g['c']
        mean = lambda name: round(c[name]['sum'] / c[name]['n'], 4) if name in c and c[name]['n'] else None
        out.append(dict(label=prof['label'], phase=phase, b=b, tag=tag or '-', mode=mode, steps=g['steps'],
                        tokens=mean('tokens'), padded=mean('padded'), loop_ms=mean('loop'), step_ms=mean('step'),
                        execute_ms=mean('step.execute'), worker_forward_ms=mean('worker.forward'),
                        gpu_forward_ms=mean('gpu.forward'), n_gpu=c['gpu.forward']['n'] if 'gpu.forward' in c else 0,
                        step_p50=round(c['step']['p50'], 4) if 'step' in c else None,
                        approx=bool(g.get('approx'))))
    return out


def dispatch_table(prof):
    """Fixed-width table of the dispatch rows ('' when the log has none): tag x CUDA-graph mode per b."""
    found = dispatch_rows(prof)
    if not found:
        return ''
    fmt = lambda v, spec='%.3f': '-' if v is None else spec % v
    out = ['Steps by tag and dispatched CUDA-graph mode (ms per step):',
           '%-8s %3s  %-20s %-10s %7s %8s %8s %9s %9s %10s %10s' % (
               'phase', 'b', 'tag', 'mode', 'steps', 'tokens', 'padded', 'step', 'step p50', 'host fwd', 'gpu fwd')]
    for r in found:
        out.append('%-8s %3d  %-20s %-10s %7d %8s %8s %9s %9s %10s %10s' % (
            r['phase'], r['b'], r['tag'], r['mode'], r['steps'], fmt(r['tokens'], '%.1f'), fmt(r['padded'], '%.1f'),
            fmt(r['step_ms']), ('~' if r['approx'] else '') + fmt(r['step_p50']), fmt(r['worker_forward_ms']),
            fmt(r['gpu_forward_ms'])))
    return '\n'.join(out) + '\n'


def probe_rows(prof):
    """Long form of the probe-only passes: label, kind, graph, bin, passes, component, depth, n, calls,
    ms per pass (over every pass of the group), mean, p50, p95 (PROBE_TREE order; then the rest)."""
    parent = dict(PROBE_TREE)
    order = [name for name, _ in PROBE_TREE]
    out = []
    for (kind, graph, bin_), g in sorted((prof.get('probes') or {}).items(),
                                         key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        passes = g['steps'] or 1
        names = [n for n in order if n in g['c']] + sorted(n for n in g['c'] if n not in parent)
        for name in names:
            depth, up = 0, parent.get(name)
            while up is not None:
                if up in g['c']:
                    depth += 1
                up = parent.get(up)
            stat = g['c'][name]
            out.append(dict(label=prof['label'], kind=kind, graph=graph, bin=bin_, passes=g['steps'],
                            component=name, depth=depth, n=stat['n'], calls=stat['calls'],
                            per_pass_ms=round(stat['sum'] / passes, 4), mean_ms=round(stat['mean'], 4),
                            p50_ms=round(stat['p50'], 4), p95_ms=round(stat['p95'], 4),
                            approx=bool(g.get('approx'))))
    return out


def probe_table(prof):
    """Fixed-width table of the probe-only passes: one column per (kind, graph, bin), ms per pass
    ({n} when only n of the passes had the component); '' when the log has none."""
    groups = sorted((prof.get('probes') or {}).items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1]))
    if not groups:
        return ''
    found = probe_rows(prof)
    keys = [k for k, _ in groups]
    names = [n for n, _ in PROBE_TREE]
    order = []
    for r in found:
        if (r['component'], r['depth']) not in order:
            order.append((r['component'], r['depth']))
    order.sort(key=lambda cd: names.index(cd[0]) if cd[0] in names else len(names))
    cell = {((r['kind'], r['graph'], r['bin']), r['component']): r for r in found}
    width = max([len('  ' * d + n) for n, d in order] + [len('component')]) + 2
    out = ['Probe-only passes (ms per pass):',
           '%-*s' % (width, 'component') + ''.join('%24s' % ('%s %s <=%d' % k) for k in keys)]
    out.append('%-*s' % (width, 'passes') + ''.join('%24d' % g['steps'] for _, g in groups))
    for name in PROBE_COUNTS:
        out.append('%-*s' % (width, name) + ''.join('%24s' % (g.get('counts') or {}).get(name, '-')
                                                    for _, g in groups))
    for name, depth in order:
        line = []
        for k in keys:
            hit = cell.get((k, name))
            line.append('%24s' % ('-' if hit is None else '%.3f%s' % (
                hit['per_pass_ms'], '' if hit['n'] == hit['passes'] else ' {%d}' % hit['n'])))
        out.append('%-*s' % (width, '  ' * depth + name) + ''.join(line))
    return '\n'.join(out) + '\n'


def headline(prof, phase='decode', tag='', min_ms=1.0, top=8):
    """Per b: steps, loop mean/p50, gpu.forward, host (loop - gpu.forward) and the largest leaves >= min_ms.

    Leaves are the named components without timed children and the (other) remainders (whose sum is
    ``unaccounted``); together they add up to ``loop``.
    """
    out = []
    for b, g in select(prof['groups'], phase, tag).items():
        tree = rows(g)
        keys = [r['key'] for r in tree]
        leaves = []
        for i, r in enumerate(tree):
            if r['key'] in OUTSIDE or r['key'] in ('loop', 'unaccounted'):
                continue
            nxt = tree[i + 1] if i + 1 < len(tree) else None
            if nxt is not None and nxt['depth'] > r['depth'] and r['name'] != '(other)':
                continue          # has children: its time is in them
            label = r['key'] if r['name'] != '(other)' else '(other) ' + r['key'].split(':', 1)[1]
            leaves.append((label, r['ms']))
        loop, gpu = g['c'].get('loop'), g['c'].get('gpu.forward')
        steps = g['steps'] or 1
        loop_ms = loop['sum'] / steps if loop else None
        gpu_ms = gpu['sum'] / steps if gpu else None
        big = sorted((x for x in leaves if x[1] >= min_ms), key=lambda x: -x[1])[:top]
        out.append(dict(label=prof['label'], phase=phase, b=b, steps=g['steps'], loop_ms=loop_ms,
                        loop_p50=loop['p50'] if loop else None, gpu_forward_ms=gpu_ms,
                        host_ms=None if loop_ms is None or gpu_ms is None else loop_ms - gpu_ms,
                        leaves=big, approx=bool(g.get('approx')), components=len(keys)))
    return out


def summary_rows(prof):
    """One row per (phase, b, tag): steps, loop mean/p50/p95, gpu.forward, host = loop - gpu.forward."""
    out = []
    for (phase, b, tag), g in sorted(prof['groups'].items(), key=lambda kv: (kv[0][0] != 'decode', kv[0][1],
                                                                             kv[0][0], kv[0][2])):
        loop, gpu = g['c'].get('loop'), g['c'].get('gpu.forward')
        steps = g['steps'] or 1
        per_loop = loop['sum'] / steps if loop else None
        per_gpu = gpu['sum'] / steps if gpu else None
        out.append(dict(label=prof['label'], phase=phase, b=b, tag=tag or '-', steps=g['steps'],
                        loop_ms=per_loop, loop_p50=loop['p50'] if loop else None,
                        loop_p95=loop['p95'] if loop else None, gpu_forward_ms=per_gpu,
                        host_ms=(per_loop - per_gpu) if per_loop is not None and per_gpu is not None else None,
                        approx=bool(g.get('approx'))))
    return out


def csv_rows(prof):
    """Long form: label, phase, b, tag, component, depth, steps, n, calls, per-step ms, mean, p50, p95."""
    out = []
    for (phase, b, tag), g in sorted(prof['groups'].items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        for row in rows(g):
            stat = row['stat']
            out.append(dict(label=prof['label'], phase=phase, b=b, tag=tag, component=row['key'],
                            depth=row['depth'], steps=g['steps'], n=stat['n'] if stat else None,
                            calls=stat['calls'] if stat else None, per_step_ms=round(row['ms'], 4),
                            mean_ms=round(stat['mean'], 4) if stat else None,
                            p50_ms=round(stat['p50'], 4) if stat else None,
                            p95_ms=round(stat['p95'], 4) if stat else None, approx=bool(g.get('approx'))))
    return out


def headline_markdown(profiles, phase='decode', tag=''):
    """One table over every label: per b the loop, the forward on the device, the host rest and the big leaves."""
    out = ['| server | b | steps | loop ms/step | loop p50 | gpu.forward | host (loop - gpu) | largest parts (ms/step) |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    fmt = lambda v: '' if v is None else '%.2f' % v
    for prof in profiles:
        for r in headline(prof, phase, tag):
            out.append('| %s | %d | %d | %s | %s%s | %s | %s | %s |' % (
                r['label'], r['b'], r['steps'], fmt(r['loop_ms']), '~' if r['approx'] else '', fmt(r['loop_p50']),
                fmt(r['gpu_forward_ms']), fmt(r['host_ms']),
                ', '.join('%s %.2f' % (name, ms) for name, ms in r['leaves'])))
    return '\n'.join(out) + '\n'


def markdown(profiles, phase='decode', tag=''):
    """Markdown: per label, components x b (ms per step), then the per-group summary."""
    out = []
    for prof in profiles:
        by_b = select(prof['groups'], phase, tag)
        out.append('### Step profile: %s (%s, %d steps, source %s)\n' % (prof['label'], phase if tag != '' else
                   phase + ' steps without admission or completion', prof['steps'], prof['source']))
        if not by_b:
            out.append('No such steps.\n')
            continue
        order, cells = matrix(by_b)
        out.append('| component | ' + ' | '.join('b=%d (%d steps)' % (b, g['steps']) for b, g in by_b.items()) + ' |')
        out.append('| --- |' + ' ---: |' * len(by_b))
        for row in order:
            line = []
            for b in by_b:
                hit = cells[b].get(row['key'])
                if hit is None:
                    line.append('')
                    continue
                cps = calls_per_step(hit['stat'], by_b[b]['steps'])
                line.append('%.2f%s' % (hit['ms'], ' (%.1fx)' % cps if cps is not None and abs(cps - 1) > 1e-9 else ''))
            out.append('| %s%s | %s |' % ('&nbsp;&nbsp;' * row['depth'], row['name'], ' | '.join(line)))
        out.append('')
        out.append('| phase | b | tag | steps | loop ms/step | p50 | p95 | gpu.forward | host (loop - gpu) |')
        out.append('| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
        fmt = lambda v: '' if v is None else '%.2f' % v
        for r in summary_rows(prof):
            out.append('| %s | %d | %s | %d | %s | %s%s | %s%s | %s | %s |' % (
                r['phase'], r['b'], r['tag'], r['steps'], fmt(r['loop_ms']), '~' if r['approx'] else '',
                fmt(r['loop_p50']), '~' if r['approx'] else '', fmt(r['loop_p95']), fmt(r['gpu_forward_ms']),
                fmt(r['host_ms'])))
        out.append('')
    return '\n'.join(out)


def find_logs(paths):
    found = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found += sorted(p for p in path.rglob('*.log') if p.is_file() and any(True for _ in lines_of(p)))
        elif path.is_file():
            found.append(path)
        else:
            raise SystemExit('no such file or directory: %s' % raw)
    return found


def load(paths):
    return [profile(p) for p in find_logs(paths)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('paths', nargs='+', help='server logs or directories searched for *.log')
    parser.add_argument('--phase', default='decode', choices=('decode', 'mixed', 'prefill', 'empty'))
    parser.add_argument('--tag', default='', help='"" (default: no admission/completion), admit, cancel, close, '
                                                  'finish, admit+finish, or "*" for all tags merged')
    parser.add_argument('--csv', help='write the long-form rows of every group here')
    parser.add_argument('--markdown', help='write the markdown tables here')
    parser.add_argument('--probe-csv', help='write the long-form rows of the probe-only passes here')
    parser.add_argument('--dispatch-csv', help='write the dispatch rows (phase, b, tag, CUDA-graph mode) here')
    args = parser.parse_args(argv)
    tag = None if args.tag == '*' else args.tag
    profiles = load(args.paths)
    if not profiles:
        raise SystemExit('no REMOTE_RA_STEP lines found')
    print('Largest parts per decode step (%s, tag %r):' % (args.phase, args.tag))
    print(headline_markdown(profiles, args.phase, tag))
    for prof in profiles:
        idle = prof.get('idle') or {}
        print('== %s  (%s)  steps %d, idle iterations %s (%.1f ms), source %s, profiler faults %s, histogram %s' % (
            prof['label'], prof['path'], prof['steps'], idle.get('n'), idle.get('ms') or 0.0, prof['source'],
            prof['faults']['errors'] and '%(errors)d (%(last_error)s)' % prof['faults'] or 0,
            json.dumps(prof['hist'], sort_keys=True)))
        print(text_table(prof, args.phase, tag))
        for r in summary_rows(prof):
            print('  %-8s b=%-2d %-13s steps %6d  loop %8.3f ms (p50 %s%.3f, p95 %s%.3f)  gpu.forward %s  host %s' % (
                r['phase'], r['b'], r['tag'], r['steps'], r['loop_ms'] or 0.0, '~' if r['approx'] else '',
                r['loop_p50'] or 0.0, '~' if r['approx'] else '', r['loop_p95'] or 0.0,
                '-' if r['gpu_forward_ms'] is None else '%.3f' % r['gpu_forward_ms'],
                '-' if r['host_ms'] is None else '%.3f' % r['host_ms']))
        print()
        table = dispatch_table(prof)
        if table:
            print(table)
        table = probe_table(prof)
        if table:
            print(table)
    if args.csv:
        import csv
        fields = ('label', 'phase', 'b', 'tag', 'component', 'depth', 'steps', 'n', 'calls', 'per_step_ms',
                  'mean_ms', 'p50_ms', 'p95_ms', 'approx')
        with open(args.csv, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            for prof in profiles:
                writer.writerows(csv_rows(prof))
    if args.dispatch_csv:
        import csv
        fields = ('label', 'phase', 'b', 'tag', 'mode', 'steps', 'tokens', 'padded', 'loop_ms', 'step_ms', 'step_p50',
                  'execute_ms', 'worker_forward_ms', 'gpu_forward_ms', 'n_gpu', 'approx')
        with open(args.dispatch_csv, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            for prof in profiles:
                writer.writerows(dispatch_rows(prof))
    if args.probe_csv:
        import csv
        fields = ('label', 'kind', 'graph', 'bin', 'passes', 'component', 'depth', 'n', 'calls', 'per_pass_ms',
                  'mean_ms', 'p50_ms', 'p95_ms', 'approx')
        with open(args.probe_csv, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            for prof in profiles:
                writer.writerows(probe_rows(prof))
    if args.markdown:
        Path(args.markdown).write_text(markdown(profiles, args.phase, tag) + '\n', encoding='utf-8', newline='\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
