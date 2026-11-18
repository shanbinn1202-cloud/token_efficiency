"""Local only (never delivered): the sealed plans' gates of perf-swe32, printed for the validation log.

For every plan (the Table 1 five full, goalb4, obs, rfill, stock and the optional goal: one ghx4 batch job each; and
the ghx4-interactive variant of each, 2x billing): controller.validate_plan (phrase, partition,
the server, K = 16, the pinned host and attention pool, the memory fit, the time fit, the limit within 1.5 x the
expected time), its TIME line (limit, expected, agent cap and minimum, worst case, with the evaluation wait), its
MEMORY line (--mem=113560M; ours 1-GiB connector pool + 2.29-GiB carries + 70-GiB arena + 32-GiB reserve = 105.29 GiB,
5.61 GiB spare, at least 5 required; stock 32 GiB), STARTUP_SPIKE (ELIMINATED: --language-model-only on every server)
and COST (billed GPU-hours at the limit and at the expected time; the totals by role), and the pinned release-check
receipt of the light login gates (RELEASE_RECEIPT). Then the refusals the gates must make:
memory (--mem above the share: 113561M / 144g / 200g; the 88-GiB arena of swe16, a 71-GiB arena, the 32-GiB default
pool, the 34-GiB reserve: each inside the 5-GiB margin; an arena below 1.5 x the measured 27.71 GiB or below the
58.32-GiB bound of swe16's 16 largest sessions; a server without --language-model-only), time (a cap that overruns the
limit, a minimum that does not fit after the evaluation wait, a limit above 1.5 x the expected time, a phrase naming
other minutes or INTERACTIVE), concurrency (agents, stock or overlap not K), the features (a runtime without
--language-model-only, --connector-pool-gib, --decode-fastpath, or without --decode-fastpath2 while its switch is on;
the switch without --decode-fastpath), the Goal rewriter batch of plan goalb4 (runtime 5b15b2f: N = 1 or 17, N above
--admission-overlap, the batch on full or stock, a pin without --goal-rewriter-batch; 120bc3c: a token cap below 0,
above 16 x 262144, without the batch or on a pin without the cap), the optimized switches (--decode-fastpath3 on a
runtime without it or without --decode-fastpath2, --deferred-completion-io on a runtime without it), the interactive
variants (a batch phrase on ghx4-interactive and the reverse, a variant whose server differs, above 02:00:00), the
receipt (another result line, log, status, test count or commit; none) and the queue gate (another tbench job of the
user - swe32 / swe10 / swe16 / gswe32 / tb16 / gem0 - outside the -AfterJob chain, an interactive plan behind a queued
tbench job, a second ghx4-interactive job, an -AfterJob not in the queue, squeue failing). integrate-1030 (plan
vllm25): the vLLM Goal rewriter with the batch, without --admission-overlap K >= 2, a rewriter KV of 1 or 17 GiB, an
unknown backend, an incomplete setting, on observation_query; --goal-overlap on full, on stock or false; a pin without
either switch (plan and build). stock25 (the stock control at 262,144 tokens and 25 GiB): a stock key other than
kv_cache_gib / max_model_len / preemption, an unknown preemption rule, a context the runtime servers lack, a KV pool
below one request at 262144, stock settings on a plan without a stock control, a cap above its limit; its STOCK line
(only the two sizes differ from plan stock's argv). stock25x (stock25 with the 3,600-s cap): a cap above its 01:24:00,
an expected time that leaves the limit above 1.5 x it. smoke (the release smoke, role release-smoke, 2026-11-03):
its own limit rule (2.25 x; refused above, and a table1 plan at 2.15 x still refused), a cap above its 01:49:00, the
plan switch --async-scheduling as a server key, not boolean, on a stock plan, on a pin without it, without
--decode-fastpath2 or with --phase1-in-step; its ASYNC line; integrate-1103's speed defaults (a switch turned off is
passed as its --no-<switch>; nothing off on the sealed config). smoket (2026-11-09, main ba4d1ba: smoke's servers with
thinking and async): a limit above 2.25 x, caps above its 02:34:00, an evaluation wait its minimums do not fit after, a
phrase with other minutes, thinking as a server key or not boolean, a pin without --enable-thinking / --async-scheduling;
the ASYNC_FIX line: a pin without the async fixes blocks exactly the async plans (smoke, smoket). goalti / akvsi /
rtoolsti (2026-11-11, the thinking-on interactive variants): a cap above the batch plan's, another task count, thinking
off, akvsi as a variant of akvs, a refit of goali (goal fits ghx4-interactive), a cap that overruns 01:22:00; the queue
gate lets an interactive plan start beside a held tbench job and one pending only on it, and still refuses a released
one, one pending on a running job, a batch plan without -AfterJob and a held job on ghx4-interactive itself.
2026-11-12 (the 64-instance proxy, idle release): a task count other than 64, 64 tasks on pins without the sealed
64-task list or with a list that does not start with the 32 one, 64 tasks above the session pool's registration bound,
a variant with another task count; release_idle_pages false / 1 / on a pin without --release-idle-pages, a variant
without it; the TASKS64 and RELEASE_IDLE lines (which plans, the 32-task plans unchanged: no task count, idle release off).
2026-11-14 (perf-swe32-ledger, 51b4085): the LEDGER_QUERY line (the sealed feature, the row, the plans and their caps);
a cap of 75 passes as --ledger-cap-percent 75; refused: caps 0 / 101 / -5 / 50.0 / '50' / true, the cap on
observation_query or a stock server, a runtime without --ledger-cap-percent, a row that is not hooked, records under
another key, serves kv-session only or pins another default, ledger_query with a rewriter or without a budget, and
unregistered.
Exit 0 and GATE_CHECKS_PASS iff every plan fits and every refusal happens.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402

PINS = c.load(ROOT / 'pins.json')
PLANS = c.cfg(PINS)['plan_order']
# 2026-11-17: release gate 2's plan (parity.py: our full server then the stock control, the probe client); the
# earlier plan-wide lists below hold for every other plan, gate 2's own checks follow at the end
GATE2 = [n for n in PLANS if c.PARITY_KEY in c.cfg(PINS)['plans'][n]]
PLANS_EARLIER = [n for n in PLANS if n not in GATE2]


def variant(change):
    pins = copy.deepcopy(PINS)
    change(pins)
    return pins


def refused(name, function, pattern=''):
    try:
        function()
    except RuntimeError as exc:
        ok = pattern in str(exc)
        print('REFUSED %s %s: %s' % (name, 'as expected' if ok else 'BUT WITH ANOTHER REASON', str(exc)[:240]))
        return ok
    print('NOT_REFUSED %s' % name)
    return False


def plan_refused(name, change, plan='goal', pattern=''):
    pins = variant(change)
    return refused(name, lambda: c.validate_plan(pins, plan), pattern)


def features_refused(name, change, pattern=''):
    pins = variant(change)
    return refused(name, lambda: c.check_features(pins['runtime_features'], pins['config']), pattern)


class Run:
    def __init__(self, stdout, code=0):
        self.returncode, self.stdout, self.stderr = code, stdout, '' if code == 0 else 'slurm down'


def main():
    ok = True
    features = PINS['runtime_features']
    print('RUNTIME commit=%s decode_fastpath=%s decode_fastpath2=%s (switch %s) decode_fastpath3=%s (switch %s) '
          'deferred_completion_io=%s (switch %s) connector_pool=%s language_model_only=%s goal_text=%s '
          'goal_rewriter_timing=%s goal_rewriter_batch=%s (plans %s)' % (
              PINS['commit'], bool(features.get('decode_fastpath')), bool(features.get('decode_fastpath2')),
              'ON' if c.decode_fastpath2_on(PINS) else 'OFF', bool(features.get('decode_fastpath3')),
              'ON' if c.decode_fastpath3_on(PINS) else 'OFF', bool(features.get('deferred_completion_io')),
              'ON' if c.deferred_completion_io_on(PINS) else 'OFF', c.connector_pool_gib(PINS),
              bool(features.get('language_model_only')), (features.get('goal_text') or {}).get('files'),
              bool(features.get('goal_rewriter_timing')), bool(features.get('goal_rewriter_batch')),
              ','.join('%s:%d:tokens%s' % (n, s[c.BATCH_KEY], c.batch_tokens_of(PINS, s)) for n in PLANS
                       for s in c.plan_servers(PINS, n) if s.get(c.BATCH_KEY)) or 'none'))
    print('RUNTIME goal_rewriter_engine=%s (plans %s) goal_overlap=%s (plans %s)' % (
        bool(features.get(c.ENGINE_KEY)), ','.join('%s:%s:%s:%dgib' % (n, s['label'], e['guided_backend'], e['kv_gib'])
                                                  for n in PLANS for s in c.plan_servers(PINS, n)
                                                  for e in [c.engine_settings(PINS, s)] if e) or 'none',
        bool(features.get(c.GOAL_OVERLAP_KEY)), ','.join('%s:%s' % (n, s['label']) for n in PLANS
                                                       for s in c.plan_servers(PINS, n)
                                                       if s['kind'] == 'runtime' and c.goal_overlap_on(PINS, s))
        or 'none'))
    c.check_features(features, PINS['config'])
    receipt = c.check_receipt(PINS)
    print('RELEASE_RECEIPT %s commit=%s status=%s tests_run=%d skipped=%d result_line=%s log=%s' % (
        receipt['pinned'], receipt['commit'], receipt['status'], receipt['tests_run'], receipt['skipped'],
        receipt['result_line_sha256'], receipt['log_sha256']))
    hours = {}
    for name in PLANS:
        fit = c.validate_plan(PINS, name)
        memory, cost = c.memory_fit(PINS, name), c.plan_cost(PINS, name)
        server = c.plan_servers(PINS, name)[0]
        role = c.cfg(PINS)['plans'][name].get('role', 'table1')
        total = hours.setdefault(role, [0.0, 0.0])
        total[0] += cost['gpu_hours_limit']
        total[1] += cost['gpu_hours_expected']
        print('TIME plan=%s server=%s:%s partition=%s time=%s (%d s) expected=%d s (x%.2f) cap=%d s min=%d s '
              'worst_case=%d s with_eval_wait=%d s spare=%d s fits=%s' % (
                  name, server['label'], server['policy'], c.cfg(PINS)['plans'][name]['partition'],
                  c.cfg(PINS)['plans'][name]['time'], fit['time_seconds'], cost['expected_seconds'],
                  cost['limit_over_expected'], server['agent_cap_seconds'], server['agent_min_seconds'],
                  fit['worst_case_seconds'], fit['with_eval_wait_seconds'], fit['spare_seconds'], fit['fits']))
        print('MEMORY plan=%s mem=%s need=%.2f GiB (pinned %.2f + reserve %s) spare=%.2f GiB (min %s) arena=%s '
              'fits=%s' % (name, PINS['slurm']['mem'], memory['need_gib'], memory['pinned_host_gib'],
                           memory['overhead_gib'], memory['headroom_gib'], memory['min_spare_gib'],
                           memory['arena_gib'], memory['fits']))
        print('STARTUP_SPIKE plan=%s %s' % (name, memory['startup']['status']))
        print('COST plan=%s role=%s billing_factor=%d gpu_hours_limit=%.2f gpu_hours_expected=%.2f' % (
            name, role, cost['billing_factor'], cost['gpu_hours_limit'], cost['gpu_hours_expected']))
        ok &= fit['fits'] and memory['fits'] and memory['startup']['status'] == 'ELIMINATED' \
            and memory['headroom_gib'] >= 5
    for role in c.PLAN_ROLES:
        if role in hours:
            print('COST_TOTAL role=%s gpu_hours_limit=%.2f gpu_hours_expected=%.2f' % ((role,) + tuple(hours[role])))
    for name in PLANS:  # the plans with their own server settings (ctx24 / ctx25: the group's settings, 2026-10-26)
        if not c.cfg(PINS)['plans'][name].get('server'):
            continue
        first = c.plan_servers(PINS, name)[0]
        s, fit = c.server_cfg(PINS, first), c.attention_fit(PINS, c.LAYOUT, c.running_of(PINS), first)
        blocked = c.plan_blocked(PINS, name)
        print('CTX plan=%s kv_cache_gib=%d max_model_len=%d dram_arena_gib=%d servers=%s attention_blocks=%d usable=%d '
              'lone_turn=%d fits=%s qwen_kv_cache_gib=%s %s' % (
                  name, s['kv_cache_gib'], s['max_model_len'], s['dram_arena_gib'],
                  ','.join('%s:%s' % (v['label'], v['policy']) for v in c.plan_servers(PINS, name)), fit['fa_blocks'],
                  fit['usable_blocks'], fit['one_turn_blocks'], fit['fits'],
                  PINS['runtime_features'].get(c.QWEN_KV_KEY), 'BLOCKED reason=' + blocked if blocked else 'SUBMITTABLE'))
        # blocked exactly by the Qwen KV size or (2026-11-09, D-G) by the async fixes of an async plan
        # 2026-11-16: or by keep-rebuild-1116 for the scripted edge-case plans (keepedge / keepedgesync)
        ok &= fit['fits'] and (blocked is None) == (s['kv_cache_gib'] in PINS['runtime_features'][c.QWEN_KV_KEY] and (
            c.cfg(PINS)['plans'][name].get(c.ASYNC_KEY) is not True or bool(PINS['runtime_features'].get(c.ASYNC_FIX_KEY)))
            and (not any(v.get(c.CLIENT_KEY) for v in c.plan_servers(PINS, name))
                 or bool(PINS['runtime_features'].get(c.KEEP_REBUILD_KEY)))
            # (phase B: or by goal-rollback-1116 for the plans that name goal_rollback)
            and (c.cfg(PINS)['plans'][name].get(c.GOAL_ROLLBACK_KEY) is not True
                 or bool(PINS['runtime_features'].get(c.GOAL_ROLLBACK_KEY))))
    for name in PLANS:  # the plans with their own stock settings (stock25, 2026-11-02): only the two sizes differ
        if not c.cfg(PINS)['plans'][name].get('stock'):
            continue
        for s in c.plan_servers(PINS, name):
            if s['kind'] == c.STOCK:
                st, argv, plain = c.stock_config(PINS, s), c.stock_argv(PINS, 8000, s), c.stock_argv(PINS, 8000)
                if c.thinking_on(PINS, s):  # 2026-11-12: stock64t / stock32t: stock25's argv + --reasoning-parser qwen3
                    at = argv.index('--reasoning-parser')
                    ok &= argv[at + 1] == c.stock_thinking(PINS)['reasoning_parser']
                    argv = argv[:at] + argv[at + 2:]
                    kept = c.stock_kept(PINS, s)  # 2026-11-16: the keep plans add exactly the default kwargs
                    if kept:
                        at = argv.index(c.STOCK_DEFAULT_KWARGS_ARG)
                        ok &= argv[at + 1] == '{"preserve_thinking": true}'
                        argv = argv[:at] + argv[at + 2:]
                    else:
                        ok &= c.STOCK_DEFAULT_KWARGS_ARG not in argv
                if not st['async_scheduling']:  # 2026-11-16: stockkeepsync32t's own stock async switch, off
                    ok &= argv.count('--no-async-scheduling') == 1 and \
                        c.cfg(PINS)['plans'][name]['stock'].get('async_scheduling') is False
                    argv = [a for a in argv if a != '--no-async-scheduling']
                differ = [(a, b) for a, b in zip(argv, plain) if a != b]
                print('STOCK plan=%s server=%s kv_cache_gib=%d max_model_len=%d preemption=%s argv_vs_plan_stock=%s' % (
                    name, s['label'], st['kv_cache_gib'], st['max_model_len'], st.get('preemption', 'fail'), differ))
                seqs = [(str(st['max_num_seqs']), str(c.cfg(PINS)['stock']['max_num_seqs']))]                     if st['max_num_seqs'] != c.cfg(PINS)['stock']['max_num_seqs'] else []  # (11-16: stockkeep64k24)
                ok &= len(argv) == len(plain) and differ == seqs + [(str(st['max_model_len']), str(c.cfg(PINS)['stock'][
                    'max_model_len'])), (str(st['kv_cache_gib'] * c.GIB), str(c.cfg(PINS)['stock']['kv_cache_gib'] * c.GIB))]

    def server(plan):
        return lambda pins: pins['config']['plans'][plan]['servers'][0]

    def only_smoket_thinks(pins):  # every other thinking-on plan removed, the pin without --enable-thinking
        for name in [n for n, plan in pins['config']['plans'].items() if n != 'smoket' and plan.get('thinking')]:
            del pins['config']['plans'][name]
            pins['config']['plan_order'].remove(name)
        pins['runtime_features'].update(thinking=None)

    checks = [
        # a plan with its own server settings (ctx24 / ctx25)
        plan_refused('ctx24: an override of another key', lambda p: p['config']['plans']['ctx24']['server'].update(
            page_tokens=32), 'ctx24', 'server may set'),
        # 2026-11-16: max_tokens is a plan server key (keepedge), within serve.py's --max-tokens choices
        plan_refused('ctx24: a reply bound serve.py lacks', lambda p: p['config']['plans']['ctx24']['server'].update(
            max_tokens=2048), 'ctx24', '--max-tokens choices'),
        # akvs (2026-11-07): a plan's own K at most config.server's, at least 1, and its tasks at least its K
        # (2026-11-16, sessions-32-1116: at most the sealed MAX_SESSIONS 32)
        plan_refused('akvs: more sessions than the sealed MAX_SESSIONS', lambda p: p['config']['plans']['akvs'][
            'server'].update(max_running_sessions=33), 'akvs', 'above the sealed MAX_SESSIONS 32'),
        plan_refused('akvs: zero sessions', lambda p: p['config']['plans']['akvs']['server'].update(
            max_running_sessions=0), 'akvs', 'server may set'),
        plan_refused('akvs: fewer tasks than its K', lambda p: p['config']['plans']['akvs']['servers'][0].update(
            tasks=3), 'akvs', 'at least K = 4'),
        plan_refused('ctx24: a context serve.py lacks', lambda p: p['config']['plans']['ctx24']['server'].update(
            max_model_len=200000), 'ctx24', 'max-model-len choices'),
        plan_refused('ctx24: a KV size outside the argparse choices', lambda p: p['config']['plans']['ctx24']['server']
                     .update(kv_cache_gib=20), 'ctx24', 'kv-cache-gib choices'),
        plan_refused('ctx24: an arena below one session at 262144', lambda p: p['config']['plans']['ctx24']['server']
                     .update(dram_arena_gib=12), 'ctx24', 'one session'),
        plan_refused('ctx24: arena 71 GiB (4.61 spare)', lambda p: p['config']['plans']['ctx24']['server'].update(
            dram_arena_gib=71), 'ctx24', 'stay spare'),
        plan_refused('ctx25: the stock control in a plan with its own settings', lambda p: p['config']['plans']['ctx25'][
            'servers'].append(dict(label='stock262', policy='stock', step_limit=120, agent_cap_seconds=600,
                                   agent_min_seconds=300)), 'ctx25', 'stock control'),
        # stock25 (2026-11-02): a plan with its own stock settings
        plan_refused('stock25: a stock key outside the stock keys', lambda p: p['config']['plans']['stock25'][
            'stock'].update(max_num_batched_tokens=8192), 'stock25', 'stock may set'),
        # 2026-11-16: the stock control's own max_num_seqs, an int 1-32
        plan_refused('stock25: max_num_seqs above 32', lambda p: p['config']['plans']['stock25']['stock'].update(
            max_num_seqs=33), 'stock25', 'stock may set'),
        plan_refused('stock25: max_num_seqs not an int', lambda p: p['config']['plans']['stock25']['stock'].update(
            max_num_seqs='24'), 'stock25', 'stock may set'),
        plan_refused('stock25: an unknown preemption rule', lambda p: p['config']['plans']['stock25']['stock'].update(
            preemption='ignore'), 'stock25', 'stock may set'),
        plan_refused('stock25: a context the runtime servers lack', lambda p: p['config']['plans']['stock25'][
            'stock'].update(max_model_len=200000), 'stock25', 'not a context of the runtime servers'),
        plan_refused('stock25: a KV pool below one request at 262144', lambda p: p['config']['plans']['stock25'][
            'stock'].update(kv_cache_gib=15), 'stock25', 'below one request'),
        plan_refused('stock settings on a plan without a stock control', lambda p: p['config']['plans']['obs'].update(
            stock=dict(kv_cache_gib=25)), 'obs', 'without a stock control'),
        plan_refused('stock25: a cap above its 00:54:00', lambda p: p['config']['plans']['stock25']['servers'][0].update(
            agent_cap_seconds=2000), 'stock25', 'does not fit'),
        # stock25x (2026-11-02, evening): stock25 with the 3,600-s cap in 01:24:00
        plan_refused('stock25x: a cap above its 01:24:00', lambda p: p['config']['plans']['stock25x']['servers'][
            0].update(agent_cap_seconds=3800), 'stock25x', 'does not fit'),
        plan_refused('stock25x: an expected time that makes 01:24:00 loose', lambda p: p['config']['plans'][
            'stock25x'].update(expected_seconds=3300), 'stock25x', 'tight limits'),
        # smoke (2026-11-03, the release smoke): its own limit rule (2.25 x, D4) and the async plan switch (D2)
        plan_refused('smoke: a limit above 2.25 x its expected time', lambda p: p['config']['plans']['smoke'].update(
            expected_seconds=2900), 'smoke', 'tight limits'),
        plan_refused('full: 2.15 x (only the release-smoke role may exceed 1.5 x)', lambda p: p['config']['plans'][
            'full'].update(expected_seconds=1700), 'full', 'tight limits'),
        plan_refused('smoke: caps above its 01:49:00', lambda p: [s.update(agent_cap_seconds=1100) for s in p[
            'config']['plans']['smoke']['servers']], 'smoke', 'does not fit'),
        plan_refused('smoke: async_scheduling as a server key', lambda p: server('smoke')(p).update(
            async_scheduling=True), 'smoke', 'plan switch, not a server key'),
        plan_refused('smoke: async_scheduling not a boolean', lambda p: p['config']['plans']['smoke'].update(
            async_scheduling='true'), 'smoke', 'must be true or false'),
        plan_refused('stock: async_scheduling on a plan with a stock control', lambda p: p['config']['plans'][
            'stock'].update(async_scheduling=True), 'stock', 'without a stock control'),
        plan_refused('smoke: --async-scheduling on a pin without it', lambda p: p['runtime_features'].update(
            async_scheduling=None), 'smoke', 'no --async-scheduling'),
        plan_refused('smoke: --async-scheduling without --decode-fastpath2', lambda p: p['config']['server'].update(
            decode_fastpath2=False, decode_fastpath3=False), 'smoke', 'requires --decode-fastpath2'),
        plan_refused('smoke: --async-scheduling with --phase1-in-step', lambda p: p['config']['policies'][
            'random_fill'].update(admission=['phase1_in_step']), 'smoke', 'refuses --gdn-device-handoff'),
        # smoket (2026-11-09, main ba4d1ba): smoke's servers with thinking and async, caps 1,440 s, wait 1,800 s
        plan_refused('smoket: a limit above 2.25 x its expected time', lambda p: p['config']['plans']['smoket'].update(
            expected_seconds=4000), 'smoket', 'tight limits'),
        plan_refused('smoket: caps above its 02:34:00', lambda p: [s.update(agent_cap_seconds=1950) for s in p[
            'config']['plans']['smoket']['servers']], 'smoket', 'does not fit'),
        plan_refused('smoket: an evaluation wait its minimums no longer fit after', lambda p: p['config']['plans'][
            'smoket'].update(eval_wait_seconds=3800), 'smoket', 'does not fit'),
        plan_refused('smoket: a phrase with other minutes', lambda p: p['config']['plans']['smoket'].update(
            authorization=p['config']['plans']['smoket']['authorization'].replace('_154MIN_', '_109MIN_')), 'smoket',
                     'phrase'),
        plan_refused('smoket: thinking as a server key', lambda p: server('smoket')(p).update(thinking=True), 'smoket',
                     'thinking is a plan switch'),
        plan_refused('smoket: thinking not a boolean', lambda p: p['config']['plans']['smoket'].update(thinking='on'),
                     'smoket', 'thinking must be true or false'),
        plan_refused('smoket: --enable-thinking on a pin without it', lambda p: p['runtime_features'].update(
            thinking=None), 'smoket', 'no --enable-thinking'),
        plan_refused('smoket: --async-scheduling on a pin without it', lambda p: p['runtime_features'].update(
            async_scheduling=None), 'smoket', 'no --async-scheduling'),
        features_refused('a pin without --enable-thinking while smoket asks for it', only_smoket_thinks,
                         "plans ['smoket'] ask for --enable-thinking"),
        # integrate-1030 (vllm25): the vLLM Goal rewriter and --goal-overlap where serve.py refuses them
        plan_refused('vllm25: the vLLM rewriter with the batch', lambda p: server('vllm25')(p).update(
            goal_rewriter_batch=4, goal_rewriter_batch_tokens=16384), 'vllm25', 'excludes'),
        plan_refused('vllm25: the vLLM rewriter without --admission-overlap', lambda p: p['config']['policies'][
            'goal_query'].update(admission=[]), 'vllm25', 'K >= 2'),
        plan_refused('vllm25: rewriter KV 17 GiB', lambda p: server('vllm25')(p).update(goal_rewriter_vllm_kv_gib=17),
                     'vllm25', 'kv-gib'),
        plan_refused('vllm25: rewriter KV 1 GiB', lambda p: server('vllm25')(p).update(goal_rewriter_vllm_kv_gib=1),
                     'vllm25', 'kv-gib'),
        plan_refused('vllm25: an unknown backend', lambda p: server('vllm25')(p).update(
            goal_rewriter_guided_backend='guidance'), 'vllm25', 'guided-backend'),
        plan_refused('vllm25: the engine without its KV setting', lambda p: server('vllm25')(p).pop(
            'goal_rewriter_vllm_kv_gib'), 'vllm25', 'all three named'),
        # c07d42a (vllm25 r2): batch invariance only as a switch of the sealed launcher, true or absent
        plan_refused('vllm25: batch invariance on a pin without the switch', lambda p: p['runtime_features'][
            'goal_rewriter_engine'].update(batch_invariant_arg=None), 'vllm25', '--goal-rewriter-batch-invariant'),
        plan_refused('vllm25: batch invariance not a boolean', lambda p: server('vllm25')(p).update(
            goal_rewriter_batch_invariant='true'), 'vllm25', 'must be true or absent'),
        plan_refused('vllm25: on observation_query', lambda p: server('vllm25')(p).update(policy='observation_query'),
                     'vllm25', 'goal_query with the Goal rewriter'),
        plan_refused('--goal-overlap on full', lambda p: server('full')(p).update(goal_overlap=True), 'full',
                     'goal_query'),
        plan_refused('--goal-overlap on stock', lambda p: server('stock')(p).update(goal_overlap=True), 'stock',
                     'stock server'),
        plan_refused('vllm25: goal_overlap false', lambda p: server('vllm25')(p).update(goal_overlap=False), 'vllm25',
                     'must be true'),
        plan_refused('vllm25: a pin without --goal-rewriter-engine', lambda p: p['runtime_features'].update(
            goal_rewriter_engine=None), 'vllm25', 'no --goal-rewriter-engine'),
        plan_refused('vllm25: a pin without --goal-overlap', lambda p: p['runtime_features'].update(goal_overlap=None),
                     'vllm25', 'no --goal-overlap'),
        features_refused('a pin without --goal-rewriter-engine while vllm25 asks for it',
                         lambda p: p['runtime_features'].update(goal_rewriter_engine=None), 'goal-rewriter-engine'),
        features_refused('a pin without --goal-overlap while vllm25 asks for it',
                         lambda p: p['runtime_features'].update(goal_overlap=None), 'goal-overlap'),
        # memory
        plan_refused('mem 113561M (above the share)', lambda p: p['slurm'].update(mem='113561M'), 'full', 'share'),
        plan_refused('mem 144g', lambda p: p['slurm'].update(mem='144g'), 'stock', 'share'),
        plan_refused('mem 200g', lambda p: p['slurm'].update(mem='200g'), 'goal', 'share'),
        plan_refused('swe16 arena 88 GiB', lambda p: p['config']['server'].update(dram_arena_gib=88), 'goal',
                     'stay spare'),
        plan_refused('arena 71 GiB (4.61 spare)', lambda p: p['config']['server'].update(dram_arena_gib=71), 'full',
                     'stay spare'),
        plan_refused('the 32-GiB default connector pool', lambda p: p['config']['server'].pop('connector_pool_gib'),
                     'obs', 'stay spare'),
        plan_refused('reserve 34 GiB', lambda p: p['config']['memory'].update(process_overhead_gib=34), 'rfill',
                     'stay spare'),
        plan_refused('arena 41 GiB (< 1.5 x 27.71)', lambda p: p['config']['server'].update(dram_arena_gib=41), 'goal',
                     'measured arena demand'),
        plan_refused('arena 58 GiB (< the 58.32-GiB bound)', lambda p: p['config']['server'].update(dram_arena_gib=58),
                     'goal', 'measured arena demand'),
        plan_refused('stock without --language-model-only',
                     lambda p: p['config']['stock'].update(language_model_only=False), 'stock', 'multi-modal warmup'),
        plan_refused('ours without --language-model-only',
                     lambda p: p['config']['server'].update(language_model_only=False), 'full', 'multi-modal warmup'),
        # time
        plan_refused('cap overruns the limit', lambda p: server('full')(p).update(agent_cap_seconds=1921), 'full',
                     'does not fit'),
        plan_refused('minimum after the evaluation wait', lambda p: server('goal')(p).update(agent_min_seconds=4300),
                     'goal', 'does not fit'),
        plan_refused('limit above 1.5 x expected', lambda p: p['config']['plans']['stock'].update(expected_seconds=2000),
                     'stock', 'tight limits'),
        plan_refused('phrase with other minutes', lambda p: p['config']['plans']['goal'].update(
            authorization=p['config']['plans']['goal']['authorization'].replace('_102MIN_', '_120MIN_')), 'goal',
                     'phrase'),
        plan_refused('phrase INTERACTIVE on ghx4', lambda p: p['config']['plans']['full'].update(
            authorization=p['config']['plans']['full']['authorization'].replace('_BATCH_', '_INTERACTIVE_')), 'full',
                     'phrase'),
        plan_refused('time above the ghx4-interactive maximum', lambda p: p['config']['plans']['goal'].update(
            partition='ghx4-interactive', time='02:30:00'), 'goal', 'maximum'),
        plan_refused('goali above the ghx4-interactive maximum', lambda p: p['config']['plans']['goali'].update(
            time='02:01:00', authorization=p['config']['plans']['goali']['authorization'].replace('_102MIN_', '_121MIN_')),
                     'goali', 'maximum'),
        plan_refused('a batch phrase on ghx4-interactive', lambda p: p['config']['plans']['goalb4i'].update(
            authorization=p['config']['plans']['goalb4i']['authorization'].replace('_INTERACTIVE_', '_BATCH_')),
                     'goalb4i', 'phrase'),
        plan_refused('the batch plan moved to ghx4-interactive', lambda p: p['config']['plans']['goalb4'].update(
            partition='ghx4-interactive'), 'goalb4', 'phrase'),
        plan_refused('an interactive variant with another server', lambda p: server('obsi')(p).update(
            agent_cap_seconds=2000), 'obsi', 'variant_of'),
        plan_refused('an interactive variant of an unknown plan', lambda p: p['config']['plans']['obsi'].update(
            variant_of='nope'), 'obsi', 'variant_of'),
        plan_refused('a batch plan with the interactive role', lambda p: p['config']['plans']['obs'].update(
            role='interactive'), 'obs', 'role'),
        # 2026-11-11: the thinking-on variants refit only the cap and minimum, only lower, only for a batch plan above
        # the partition's maximum, and copy the plan settings and thinking
        plan_refused('goalti: a cap above goalt\'s', lambda p: server('goalti')(p).update(agent_cap_seconds=7300),
                     'goalti', 'variant_of'),
        plan_refused('rtoolsti: another task count', lambda p: server('rtoolsti')(p).update(tasks=16), 'rtoolsti',
                     'variant_of'),
        plan_refused('akvsi: thinking off', lambda p: p['config']['plans']['akvsi'].update(thinking=False), 'akvsi',
                     'variant_of'),
        plan_refused('akvsi as a variant of akvs (8 tasks, 4 sessions)', lambda p: p['config']['plans']['akvsi'].update(
            variant_of='akvs'), 'akvsi', 'variant_of'),
        plan_refused('goali: a refit while goal fits ghx4-interactive', lambda p: server('goali')(p).update(
            agent_min_seconds=3000), 'goali', 'variant_of'),
        plan_refused('rtoolsti: a cap that overruns 01:22:00', lambda p: server('rtoolsti')(p).update(
            agent_cap_seconds=3300), 'rtoolsti', 'does not fit'),
        # concurrency
        plan_refused('agents not K', lambda p: p['config']['agent'].update(max_running=10), 'full', 'must be equal'),
        plan_refused('stock seqs not K', lambda p: p['config']['stock'].update(max_num_seqs=10), 'stock',
                     'must be equal'),
        plan_refused('overlap 17', lambda p: p['config']['policies']['goal_query'].update(
            admission=['admission_overlap=17']), 'goal'),
        # features
        features_refused('a runtime without --language-model-only', lambda p: p['runtime_features'].update(
            language_model_only=None), 'language-model-only'),
        features_refused('a runtime without --connector-pool-gib', lambda p: p['runtime_features'].update(
            connector_pool=None), 'connector-pool-gib'),
        features_refused('a runtime without --decode-fastpath', lambda p: p['runtime_features'].update(
            decode_fastpath=None), 'decode-fastpath'),
        features_refused('the fastpath2 switch on a runtime without it', lambda p: (
            p['config']['server'].update(decode_fastpath2=True), p['runtime_features'].update(decode_fastpath2=None)),
                         'decode-fastpath2'),
        features_refused('the fastpath2 switch without --decode-fastpath', lambda p: p['config']['server'].update(
            decode_fastpath2=True, decode_fastpath=False), 'requires decode_fastpath'),
        features_refused('connector pool 32 GiB', lambda p: p['config']['server'].update(connector_pool_gib=32),
                         'connector_pool_gib'),
        features_refused('the fastpath3 switch on a runtime without it', lambda p: p['runtime_features'].update(
            decode_fastpath3=None), 'decode-fastpath3'),
        features_refused('the fastpath3 switch without --decode-fastpath2', lambda p: p['config']['server'].update(
            decode_fastpath2=False), 'requires decode_fastpath2'),
        features_refused('the deferred-completion switch on a runtime without it', lambda p: p['runtime_features'].update(
            deferred_completion_io=None), 'deferred-completion-io'),
        features_refused('a token cap on a pin without --goal-rewriter-batch-tokens (5b15b2f)',
                         lambda p: p['runtime_features'].update(goal_rewriter_batch=dict(
                             serve_arg='--goal-rewriter-batch', config_key='goal_rewriter_batch', min=2, max=16)),
                         'goal-rewriter-batch-tokens'),
        # the Goal rewriter batch (goalb4 only; runtime 5b15b2f)
        plan_refused('batch 1', lambda p: server('goalb4')(p).update(goal_rewriter_batch=1), 'goalb4', 'goal-rewriter-batch'),
        plan_refused('batch 17', lambda p: server('goalb4')(p).update(goal_rewriter_batch=17), 'goalb4',
                     'goal-rewriter-batch'),
        plan_refused('batch 4 above --admission-overlap 2', lambda p: p['config']['policies']['goal_query'].update(
            admission=['admission_overlap=2']), 'goalb4', 'goal-rewriter-batch'),
        plan_refused('batch on full', lambda p: server('full')(p).update(goal_rewriter_batch=4), 'full',
                     'goal-rewriter-batch'),
        plan_refused('batch on stock', lambda p: server('stock')(p).update(goal_rewriter_batch=4), 'stock',
                     'goal-rewriter-batch'),
        features_refused('a pin without --goal-rewriter-batch while goalb4 asks for it',
                         lambda p: p['runtime_features'].update(goal_rewriter_batch=None), 'goal-rewriter-batch'),
        plan_refused('token cap -1', lambda p: server('goalb4')(p).update(goal_rewriter_batch_tokens=-1), 'goalb4',
                     'batch_tokens'),
        plan_refused('token cap above 16 x 262144', lambda p: server('goalb4')(p).update(
            goal_rewriter_batch_tokens=16 * 262144 + 1), 'goalb4', 'batch_tokens'),
        plan_refused('token cap missing on a batched server', lambda p: server('goalb4')(p).pop(
            'goal_rewriter_batch_tokens'), 'goalb4', 'batch_tokens'),
        plan_refused('token cap without the batch', lambda p: server('goal')(p).update(goal_rewriter_batch_tokens=16384),
                     'goal', 'batch_tokens'),
        # the pinned release-check receipt (light login gates): any other value, or none, is refused
        refused('receipt: none', lambda: c.check_receipt(variant(lambda p: p.pop('release_receipt'))),
                'release-check receipt'),
        refused('receipt: another commit', lambda: c.check_receipt(variant(lambda p: p.update(
            commit='5b15b2ff2d7497717dffdb4a77641781a8c1090e'))), 'no pinned release-check receipt'),
    ] + [refused('receipt: another %s' % key, lambda key=key, value=value: c.check_receipt(variant(
        lambda p: p['release_receipt'].update({key: value}))), 'release-check receipt')
         for key, value in (('result_line_sha256', '0' * 64), ('log_sha256', '1' * 64), ('status', 'FAIL'),
                            ('tests_run', 1383), ('skipped', 0))]
    with tempfile.TemporaryDirectory() as tmp:
        plan = c.cfg(PINS)['plans']['goalb4']
        gate = lambda out, after, name, plan_name='goalb4': c.queue_gate(
            Path(tmp), PINS, c.cfg(PINS)['plans'][plan_name], dict(plan=plan_name, after_job=after), name,
            runner=lambda *a, **k: Run(out), user='ycao11')
        for i, other in enumerate(('swe32-full', 'swe10-goal', 'swe16-stock', 'gswe32-goal', 'tb16-dryo', 'gem0-x')):
            checks.append(refused('queue gate: %s queued' % other, lambda other=other, i=i: gate(
                '777|PENDING|ghx4|%s|(null)\n' % other, None, 'q%d.json' % i), 'queue holds'))
        chain = ('777|RUNNING|ghx4|swe32-full|(null)\n666|PENDING|ghx4|swe32-goalb4|afterany:777(unfulfilled)\n'
                 '888|PENDING|ghx4|other|(null)\n')
        checks.append(refused('queue gate: -AfterJob names the first job of a queued chain', lambda: gate(
            chain, '777', 'qc1.json'), 'queue holds'))
        checks.append(refused('queue gate: -AfterJob not in the queue', lambda: gate(
            '777|RUNNING|ghx4|swe32-full|(null)\n', '776', 'qc2.json'), 'names no job of yours'))
        checks.append(refused('queue gate: an interactive plan behind a queued tbench job', lambda: gate(
            '777|PENDING|ghx4|swe32-full|(null)\n', None, 'qi1.json', 'goalb4i'), 'queue holds'))
        checks.append(refused('queue gate: a second ghx4-interactive job', lambda: gate(
            '999|RUNNING|ghx4-interactive|sctx2-x|(null)\n', None, 'qi2.json', 'fulli'), 'at most'))
        checks.append(refused('queue gate: squeue failing', lambda: c.queue_gate(
            Path(tmp), PINS, plan, dict(plan='goalb4', after_job=None), 'qf.json',
            runner=lambda *a, **k: Run('', 1), user='ycao11'), 'squeue failed'))
        value = gate('777|RUNNING|ghx4|swe32-full|(null)\n888|PENDING|ghx4|other|(null)\n', '777', 'qa.json')
        print('QUEUE_GATE named -AfterJob 777: %s' % value['status'])
        checks.append(value['status'] == 'QUEUE_GATE_PASS')
        value = gate(chain, '666', 'qb.json')
        print('QUEUE_GATE chain 777 <- 666, -AfterJob 666: %s chain=%s' % (value['status'], value['chain']))
        checks.append(value['status'] == 'QUEUE_GATE_PASS' and value['chain'] == ['666', '777'])
        value = gate('888|PENDING|ghx4|other|(null)\n', None, 'qi3.json', 'goalb4i')
        print('QUEUE_GATE interactive goalb4i, no tbench job queued: %s' % value['status'])
        checks.append(value['status'] == 'QUEUE_GATE_PASS')
        # 2026-11-11: an interactive plan beside a held tbench job and one pending only on it (gctx25 3560498 held,
        # smokeqt 3568310 waiting on it); a released one, one pending on a running job, a batch plan: refused
        held = ('3560498|PENDING|ghx4|gswe32-gctx25|(null)|JobHeldUser\n'
                '3568310|PENDING|ghx4|tb16-smokeqt|afterany:3560498(unfulfilled)|Dependency\n')
        value = gate(held, None, 'qh1.json', 'goalti')
        print('QUEUE_GATE interactive goalti beside held 3560498 and 3568310 pending on it: %s parked=%d' % (
            value['status'], len(value['parked'])))
        checks.append(value['status'] == 'QUEUE_GATE_PASS' and value['blocking'] == [] and len(value['parked']) == 2)
        checks.append(refused('queue gate: interactive beside a released tbench job', lambda: gate(
            held.replace('JobHeldUser', 'Priority'), None, 'qh2.json', 'goalti'), 'queue holds'))
        checks.append(refused('queue gate: interactive beside a tbench job pending on a running one', lambda: gate(
            held.replace('3560498(unfulfilled)', '777(unfulfilled)') + '777|RUNNING|ghx4|swe32-full|(null)|None\n',
            None, 'qh3.json', 'rtoolsti'), 'queue holds'))
        checks.append(refused('queue gate: a batch plan beside a held tbench job without -AfterJob', lambda: gate(
            held, None, 'qh4.json', 'goalt'), 'queue holds'))
        checks.append(refused('queue gate: a held job on ghx4-interactive itself', lambda: gate(
            '3560498|PENDING|ghx4-interactive|gswe32-gctx25|(null)|JobHeldUser\n', None, 'qh5.json', 'akvsi'),
            'at most'))
    # the switch on (the user's decision after spd2, 2026-10-16: the sealed config) and off: both valid on this pin
    on = variant(lambda p: p['config']['server'].update(decode_fastpath2=True))
    c.check_features(on['runtime_features'], on['config'])
    args = c.launch_arguments(on, c.plan_servers(on, 'full')[0], '/m')
    print('FASTPATH2_SWITCH_ON sealed=%s valid=%s args_tail=%s' % (
        c.decode_fastpath2_on(PINS), all(c.validate_plan(on, n)['fits'] for n in PLANS), args[-4:]))
    checks.append(args.count('--decode-fastpath2') == 1 and c.decode_fastpath2_on(PINS))
    # the optimized runtime of record: fastpath3 and the deferred completion IO on every runtime server, never stock
    tails = {n: c.launch_arguments(PINS, c.plan_servers(PINS, n)[0], '/m') for n in PLANS
             if c.plan_servers(PINS, n)[0]['kind'] == 'runtime'}
    print('OPTIMIZED_FLAGS plans=%s every_runtime_server=%s stock=%s' % (
        ','.join(sorted(tails)), all(a.count(f) == 1 for a in tails.values() for f in (
            '--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3', '--deferred-completion-io',
            '--language-model-only')),
        [f for f in ('--decode-fastpath3', '--deferred-completion-io') if f in c.stock_argv(PINS, 8000)]))
    checks.append(all(a.count(f) == 1 for a in tails.values() for f in (
        '--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3', '--deferred-completion-io',
        '--language-model-only')) and not any(f in c.stock_argv(PINS, 8000) for f in (
            '--decode-fastpath3', '--deferred-completion-io')))
    # off (and smoke's --async-scheduling with it: serve.py refuses async without --decode-fastpath2); on a pin with the
    # speed defaults (integrate-1103) the off switches are passed explicitly
    off = variant(lambda p: p['config']['server'].update(decode_fastpath2=False, decode_fastpath3=False) or [
        plan.update(async_scheduling=False) for plan in p['config']['plans'].values() if c.ASYNC_KEY in plan])
    c.check_features(off['runtime_features'], off['config'])
    args = c.launch_arguments(off, c.plan_servers(off, 'full')[0], '/m')
    print('FASTPATH2_SWITCH_OFF valid=%s off_args=%s' % (all(c.validate_plan(off, n)['fits'] for n in PLANS),
                                                         [a for a in args if a.startswith('--no-')]))
    checks.append('--decode-fastpath2' not in args and (
        [a for a in args if a.startswith('--no-')] == ['--no-decode-fastpath2', '--no-decode-fastpath3']
        if PINS['runtime_features'].get(c.SPEED_KEY) else not [a for a in args if a.startswith('--no-')]))
    # integrate-1103: the sealed config names every speed switch on, so no off switch reaches any plan's server
    sealed_off = sorted({a for n in PLANS for s in c.plan_servers(PINS, n) if s['kind'] == 'runtime'
                         for a in c.launch_arguments(PINS, s, '/m') if a.startswith('--no-')})
    asynchronous = ['%s:%s:%s' % (n, s['label'], 'on' if c.async_scheduling_on(PINS, s) else 'off') for n in PLANS
                    for s in c.plan_servers(PINS, n) if c.ASYNC_KEY in s]
    print('SPEED_DEFAULTS feature=%s off_switches_passed=%s' % (bool(PINS['runtime_features'].get(c.SPEED_KEY)),
                                                                sealed_off or 'none'))
    print('ASYNC feature=%s plans=%s every_other_plan=--no-async-scheduling' % (
        bool(PINS['runtime_features'].get(c.ASYNC_KEY)), ','.join(asynchronous) or 'none'))
    checks.append(not sealed_off and all(
        (c.ASYNC_SWITCH in c.launch_arguments(PINS, s, '/m')) == c.async_scheduling_on(PINS, s)
        for n in PLANS for s in c.plan_servers(PINS, n) if s['kind'] == 'runtime'))
    # 2026-11-09 (D-G): the async fixes; a pin without them (ea18256 / f4ccd4b) blocks exactly the async plans
    asynchronous_plans = [n for n in PLANS if c.cfg(PINS)['plans'][n].get(c.ASYNC_KEY) is True]
    without = variant(lambda p: p['runtime_features'].update({c.ASYNC_FIX_KEY: None}))
    blocked_without = [n for n in PLANS if c.async_fix_block(without, n)]
    print('ASYNC_FIX feature=%s plans=%s blocked_now=%s blocked_without_the_fixes=%s' % (
        bool(features.get(c.ASYNC_FIX_KEY)), ','.join(asynchronous_plans) or 'none',
        ','.join(n for n in PLANS if c.plan_blocked(PINS, n)) or 'none', ','.join(blocked_without) or 'none'))
    checks.append(blocked_without == asynchronous_plans and all(c.validate_plan(without, n)['fits'] for n in PLANS)
                  and all((c.async_fix_block(PINS, n) is None) == bool(features.get(c.ASYNC_FIX_KEY)) for n in
                          asynchronous_plans))
    # 2026-11-12: the 64-task plans (`tasks` 64, the sealed list pins.task_sets."64") and the idle-release plan key
    sixty = [n for n in PLANS if c.plan_task_count(PINS, n) == 64]
    # 2026-11-16: the second-half plans (`tasks` "h2" = instances 33-64 of the 64 list, disjoint from the 32 list)
    half = [n for n in PLANS if c.plan_task_count(PINS, n) == 'h2']
    # 2026-11-17 (integrate-1117: the session pool registers up to 256 tasks): the 150-task plans (`tasks` 150, the
    # sealed list pins.task_sets."150": the first 150 in dataset order, its first 64 the 64 list), K = 24
    hundred50 = [n for n in PLANS_EARLIER if c.plan_task_count(PINS, n) == 150]
    print('TASKS150 plans=%s list=%s first64_equal=%s pool=%s session_pool_max_tasks=%s' % (
        ','.join(hundred50) or 'none', len(c.instance_ids(PINS, hundred50[0])) if hundred50 else None,
        bool(hundred50) and c.instance_ids(PINS, hundred50[0])[:64] == c.instance_ids(PINS, sixty[0]),
        c.pool_file(PINS, hundred50[0]) if hundred50 else None,
        (PINS['runtime_features'].get('session_pool') or {}).get('max_tasks')))
    checks.append(hundred50 == ['full150k24', 'ledger150k24', 'stockkeep150k24'] and all(
        len(c.instance_ids(PINS, n)) == 150 and c.instance_ids(PINS, n)[:64] == c.instance_ids(PINS, sixty[0])
        and c.pool_file(PINS, n) == 'tasks-pool-150.json' and c.plan_running(PINS, n) == 24
        and all(s['tasks'] == 150 for s in c.plan_servers(PINS, n)) for n in hundred50))
    checks.append(plan_refused('tasks-150-above-the-session-pool', lambda p: p['runtime_features'][
        'session_pool'].update(max_tasks=64), 'full150k24', 'more than the sealed session pool registers'))
    checks.append(plan_refused('tasks-150-not-starting-with-the-32-list', lambda p: p[c.TASK_SETS_KEY]['150'].update(
        instances=list(reversed(p[c.TASK_SETS_KEY]['150']['instances']))), 'full150k24', 'does not start with'))
    print('TASKS_H2 plans=%s list=%s disjoint=%s halves_make_64=%s pool=%s' % (
        ','.join(half), len(c.instance_ids(PINS, half[0])) if half else None,
        bool(half) and not set(c.instance_ids(PINS, half[0])) & set(c.instance_ids(PINS)),
        bool(half) and c.instance_ids(PINS) + c.instance_ids(PINS, half[0]) == c.instance_ids(PINS, sixty[0]),
        c.pool_file(PINS, half[0]) if half else None))
    checks.append(len(half) == 10 and all(
        c.instance_ids(PINS, n) == c.instance_ids(PINS, sixty[0])[32:] and c.pool_file(PINS, n) == 'tasks-pool-h2.json'
        and all(s['tasks'] == 32 for s in c.plan_servers(PINS, n)) for n in half))
    checks.append(plan_refused('tasks-h2-not-the-second-half', lambda p: p[c.TASK_SETS_KEY]['h2'].update(
        instances=p[c.TASK_SETS_KEY]['64']['instances'][:32]), half[0], 'instances 33-64'))
    released = ['%s:%s' % (n, s['label']) for n in PLANS for s in c.plan_servers(PINS, n) if c.release_idle_on(PINS, s)]
    print('TASKS64 plans=%s list=%s first32_equal=%s pool=%s every_other_plan=32' % (
        ','.join(sixty) or 'none', len(c.instance_ids(PINS, sixty[0])) if sixty else None,
        bool(sixty) and c.instance_ids(PINS, sixty[0])[:32] == c.instance_ids(PINS),
        c.pool_file(PINS, sixty[0]) if sixty else None))
    print('RELEASE_IDLE feature=%s plans=%s every_other_plan=off' % (
        bool(features.get(c.RELEASE_KEY)), ','.join(released) or 'none'))
    checks.append(bool(sixty) and all(len(c.instance_ids(PINS, n)) == 64 and all(
        s['tasks'] == 64 for s in c.plan_servers(PINS, n)) for n in sixty)
        and all(c.plan_task_count(PINS, n) is None and c.instance_ids(PINS, n) == PINS['tasks']['instances']
                and all(s['tasks'] <= 32 for s in c.plan_servers(PINS, n)) for n in PLANS_EARLIER
                if n not in sixty + half + hundred50))
    checks.append(bool(released) and all(
        (c.RELEASE_SWITCH in c.launch_arguments(PINS, s, '/m')) == c.release_idle_on(PINS, s)
        for n in PLANS for s in c.plan_servers(PINS, n) if s['kind'] == 'runtime')
        and all(not c.release_idle_on(PINS, s) for n in PLANS[:PLANS.index('fullti') + 1]
                for s in c.plan_servers(PINS, n)))
    first64 = sixty[0] if sixty else 'obs64'
    checks.append(plan_refused('tasks-48', lambda p: p['config']['plans'][first64].update(tasks=48), first64,
                               'names 48 tasks'))
    checks.append(plan_refused('tasks-64-without-the-sealed-list', lambda p: p.pop(c.TASK_SETS_KEY), first64,
                               'no sealed list of that size'))
    checks.append(plan_refused('tasks-64-not-starting-with-the-32-list', lambda p: p[c.TASK_SETS_KEY]['64'].update(
        instances=list(reversed(p[c.TASK_SETS_KEY]['64']['instances']))), first64, 'does not start with'))
    checks.append(plan_refused('tasks-64-above-the-session-pool', lambda p: p['runtime_features'][
        'session_pool'].update(max_tasks=32), first64, 'more than the sealed session pool registers'))
    checks.append(plan_refused('variant-with-another-task-count', lambda p: p['config']['plans']['obs64i'].pop(
        'tasks'), 'obs64i', 'variant_of'))
    checks.append(plan_refused('release-idle-pages-false', lambda p: p['config']['plans']['fulltr']['server'].update(
        release_idle_pages=False), 'fulltr', 'release_idle_pages true'))
    checks.append(plan_refused('release-idle-pages-1', lambda p: p['config']['plans']['fulltr']['server'].update(
        release_idle_pages=1), 'fulltr', 'release_idle_pages true'))
    checks.append(plan_refused('release-idle-pages-without-the-switch', lambda p: p['runtime_features'].update(
        {c.RELEASE_KEY: None}), 'fulltr', "--release-idle-pages"))
    checks.append(plan_refused('variant-without-idle-release', lambda p: p['config']['plans']['fulltri'][
        'server'].pop('release_idle_pages'), 'fulltri', 'variant_of'))
    # 2026-11-12 (policy-registry-1112): the sealed registry's rows (describe()); every plan's runtime policy registered,
    # runnable here and agreeing with this harness's evidence checks (CHECKED); refusals without the registry, without
    # a policy, with a row that differs from the checks' reference, a plug-in, random50
    rows = features.get(c.REGISTRY_KEY) or []
    named = sorted({s['policy'] for n in PLANS for s in c.plan_servers(PINS, n) if s['kind'] == 'runtime'})
    print('POLICY_REGISTRY rows=%s runnable=%s window=%s rule=%s budget=%s rewriter=%s plans_run=%s' % (
        ','.join('%s:%s:%s' % (r['name'], r['kind'], r['budget']) for r in rows), ','.join(c.runtime_policies(PINS)),
        ','.join(c.window_policies(PINS)), ','.join(c.rule_policies(PINS)), ','.join(c.budget_policies(PINS)),
        ','.join(c.rewriter_policies(PINS)), ','.join(named)))
    checks.append([r['name'] for r in rows] == features.get('policies') and set(c.runtime_policies(PINS)) == set(c.CHECKED)
                  and set(named) <= set(c.runtime_policies(PINS)))
    first = lambda p, name: next(r for r in p['runtime_features'][c.REGISTRY_KEY] if r['name'] == name)
    checks.append(features_refused('registry-missing', lambda p: p['runtime_features'].pop(c.REGISTRY_KEY),
                                   'policy registry'))
    checks.append(features_refused('registry-without-akv-rule', lambda p: p['runtime_features'].update(
        {c.REGISTRY_KEY: [r for r in rows if r['name'] != 'akv_rule']}), 'does not register'))
    checks.append(features_refused('registry-recent-tools-order-differs', lambda p: first(p, 'recent_tools').update(
        order_id='reply_cover_v1', pin=dict(order='reply_cover_v1')), 'order_id, pin'))
    checks.append(plan_refused('registry-goal-engine-on-observation-query', lambda p: first(
        p, 'observation_query').update(goal_engine=True), 'obs', 'goal_engine'))
    checks.append(plan_refused('registry-plug-in-policy', lambda p: (p['runtime_features'][c.REGISTRY_KEY].append(dict(
        first(p, 'recent_fill'), name='oldest_first', builtin=False, contract_key='oldest_first',
        order_id='oldest_first_v1', pin=dict(order='oldest_first_v1'))), p['config']['policies'].update(
        oldest_first=dict(p['config']['policies']['recent_fill'])), p['config']['plans']['recentt']['servers'][0].update(
        policy='oldest_first')), 'recentt', 'no evidence check'))
    checks.append(plan_refused('random50-plan', lambda p: (p['config']['policies'].update(
        random50=dict(policy_args=[], admission=[], rewriter=False)), p['config']['plans']['obs']['servers'][0].update(
        policy='random50')), 'obs', 'not a swe32 runtime policy'))
    # 2026-11-12 (later): the stock control with thinking ON (config.stock.thinking; plans stock64t / stock32t)
    thinking_stock = ['%s:%s' % (n, s['label']) for n in PLANS_EARLIER for s in c.plan_servers(PINS, n)
                      if s['kind'] == c.STOCK and c.thinking_on(PINS, s)]
    contract = c.stock_thinking(PINS)
    print('STOCK_THINKING reasoning_parser=%s history=%s kwargs=%s plans=%s' % (
        contract['reasoning_parser'], contract['history'], json.dumps(contract['stock_model_kwargs'], sort_keys=True),
        ','.join(thinking_stock) or 'none'))
    thinking_plans = ('stock64t', 'stock32t', 'stock32ti', 'stockkeep32t', 'stockkeepsync32t', 'stockh2t',
                      'stockkeeph2t', 'stockkeep64t', 'stockkeep64k24',  # 2026-11-16: + keep, + the second half, phase B
                      'stockkeep150k24')  # 2026-11-17: the 150-task control
    checks.append(thinking_stock == ['stock64t:stock64t', 'stock32t:stock32t', 'stock32ti:stock32t',
                                     'stockkeep32t:stockkeep32t', 'stockkeepsync32t:stockkeepsync32t',
                                     'stockh2t:stockh2t', 'stockkeeph2t:stockkeeph2t', 'stockkeep64t:stockkeep64t',
                                     'stockkeep64k24:stockkeep64k24', 'stockkeep150k24:stockkeep150k24'] and all(
        '--reasoning-parser' in c.stock_argv(PINS, 8000, s) for n in thinking_plans
        for s in c.plan_servers(PINS, n)) and not any('--reasoning-parser' in c.stock_argv(PINS, 8000, s)
                                                       for n in PLANS_EARLIER if n not in thinking_plans
                                                       for s in c.plan_servers(PINS, n) if s['kind'] == c.STOCK))
    # 2026-11-16: the keep-reasoning history: only stockkeep32t / stockkeepsync32t carry it (the kwargs, the model
    # class), every other stock plan's contract is config.stock.thinking itself
    kept_plans = [n for n in PLANS_EARLIER for s in c.plan_servers(PINS, n) if s['kind'] == c.STOCK
                  and c.stock_kept(PINS, s)]
    print('STOCK_KEPT plans=%s' % ','.join(kept_plans))
    checks.append(kept_plans == ['stockkeep32t', 'stockkeepsync32t', 'stockkeeph2t', 'stockkeep64t', 'stockkeep64k24',
                                 'stockkeep150k24']
                  and all(
        c.stock_contract(PINS, s) is c.stock_thinking(PINS) for n in PLANS_EARLIER for s in c.plan_servers(PINS, n)
        if s['kind'] == c.STOCK and c.thinking_on(PINS, s) and n not in kept_plans) and all(
        'model_class: swe32_keep_reasoning.KeepReasoningTextbasedModel' in c.model_override_text(PINS, s, 24001)
        for n in kept_plans for s in c.plan_servers(PINS, n)) and not any(
        'swe32_keep_reasoning' in c.model_override_text(PINS, s, 24001)
        for n in PLANS_EARLIER if n not in kept_plans for s in c.plan_servers(PINS, n)))
    checks.append(plan_refused('stock-kept-without-thinking', lambda p: p['config']['plans']['stockkeep32t'].update(
        thinking=False), 'stockkeep32t', 'stock history may only be'))
    checks.append(plan_refused('stock-kept-other-kwargs', lambda p: p['config']['stock']['thinking']['kept'].update(
        default_chat_template_kwargs=dict(preserve_thinking=True, enable_thinking=True)), 'stockkeep32t',
        'kept must pin'))
    checks.append(plan_refused('stock-kept-module-elsewhere', lambda p: p['config']['stock']['thinking']['kept'][
        'agent_module'].update(tbench_path='/tmp/swe32_keep_reasoning.py'), 'stockkeepsync32t', 'kept must pin'))
    checks.append(plan_refused('stock-history-unknown', lambda p: p['config']['plans']['stockkeep32t']['stock'].update(
        history='reasoning_mixed'), 'stockkeep32t', 'stock history may only be'))
    checks.append(plan_refused('stock-thinking-with-a-stop', lambda p: p['config']['stock']['thinking'][
        'stock_model_kwargs'].update(stop=['</mswea_bash_command>']), 'stock64t', 'no stop'))
    checks.append(plan_refused('stock-thinking-history-kept', lambda p: p['config']['stock']['thinking'].update(
        history='kept'), 'stock64t', 'history reasoning_dropped'))
    checks.append(plan_refused('stock-thinking-without-the-contract', lambda p: p['config']['stock'].pop('thinking'),
                               'stock32t', 'no thinking contract'))
    checks.append(plan_refused('thinking-runtime-and-stock-together', lambda p: p['config']['plans']['stock64t'][
        'servers'].append(dict(label='obs64x', policy='observation_query', step_limit=120, agent_cap_seconds=100,
                               agent_min_seconds=50)), 'stock64t', 'runtime servers only or of stock servers only'))
    # 2026-11-16 (harness-1116): the scripted edge-case client (plan keepedge; keep-rebuild-1116 51af079 through the
    # fixture only): blocked exactly without the feature, the client only on a thinking-on runtime server, the module pin
    edge_plans = [n for n in PLANS_EARLIER if any(v.get(c.CLIENT_KEY) for v in c.plan_servers(PINS, n))]
    fixture = json.loads((Path(__file__).resolve().parent / 'fixtures' / 'keep-rebuild-51af079.json').read_text())
    with_feature = variant(lambda p: p['runtime_features'].update({c.KEEP_REBUILD_KEY: fixture['feature']}))
    print('KEEP_REBUILD feature=%s edge_plans=%s blocked_now=%s blocked_with_51af079=%s' % (
        bool(features.get(c.KEEP_REBUILD_KEY)), ','.join(edge_plans), ','.join(
            n for n in edge_plans if c.keep_rebuild_block(PINS, n)) or 'none',
        ','.join(n for n in edge_plans if c.plan_blocked(with_feature, n)) or 'none'))
    # (2026-11-16 phase B: the pin a2f9e36 carries the feature; the gdn pairs use the same client; without the
    # feature exactly the client plans are blocked)
    without_keep = variant(lambda p: p['runtime_features'].pop(c.KEEP_REBUILD_KEY, None))
    gdn = ['gdnobs', 'gdnobson', 'gdngoal', 'gdngoalon', 'gdnidle', 'gdnidleon', 'gdnobs24', 'gdnobs24on']
    goal_edge = ['keepedgegoal', 'keepedgegoalov', 'keepedgegoalp', 'keepedgegoalpi']  # goal-rollback-1116: blocked
    # (11-16, the user's decision: the functional jobs short on ghx4-interactive; one job per OFF / ON pair)
    func = ['keepedgei', 'keepedgesynci', 'gdnobsp', 'gdnobspi', 'gdngoalp', 'gdngoalpi', 'gdnidlep', 'gdnidlepi',
            'gdnobs24p', 'gdnobs24pi', 'sess24', 'sess24i', 'keepedgegoalp', 'keepedgegoalpi', 'sess24r', 'sess24ri']
    checks.append(edge_plans == ['keepedge', 'keepedgesync'] + gdn + ['keepedgegoal', 'keepedgegoalov'] + func
                  and features.get(c.KEEP_REBUILD_KEY) == fixture['feature']
                  and not any(c.keep_rebuild_block(PINS, n) for n in edge_plans)
                  and not any(c.plan_blocked(with_feature, n) for n in edge_plans if n not in goal_edge)
                  and [n for n in PLANS if c.keep_rebuild_block(without_keep, n)] == edge_plans)
    # 2026-11-16 (phase B, goal-rollback-1116 through its fixture): the plans that name goal_rollback are blocked
    # exactly while the pin lacks the feature; with it every runtime server's expectation carries goal_rollback
    # (strict on the scripted client's servers); a2f9e36: no expectation carries it
    rollback_fixture = json.loads((Path(__file__).resolve().parent / 'fixtures' / 'goal-rollback-43ff7b0.json')
                                  .read_text())
    with_rollback = variant(lambda p: p['runtime_features'].update({c.GOAL_ROLLBACK_KEY: rollback_fixture['feature']}))
    # (7a8c36e carries the feature: the block is checked on the pins without it)
    without_rollback = variant(lambda p: p['runtime_features'].pop(c.GOAL_ROLLBACK_KEY, None))
    blocked_rollback = [n for n in PLANS if c.goal_rollback_block(without_rollback, n)]
    print('GOAL_ROLLBACK feature=%s plans=%s blocked_without_it=%s blocked_with_43ff7b0=%s' % (
        bool(features.get(c.GOAL_ROLLBACK_KEY)), ','.join(goal_edge), ','.join(blocked_rollback) or 'none',
        ','.join(n for n in PLANS if c.plan_blocked(with_rollback, n)) or 'none'))
    checks.append(blocked_rollback == goal_edge and not any(c.plan_blocked(with_rollback, n) for n in PLANS)
                  and features.get(c.GOAL_ROLLBACK_KEY) == rollback_fixture['feature']
                  and not any(c.plan_blocked(PINS, n) for n in goal_edge)
                  and not any(c.GOAL_ROLLBACK_KEY in c.server_expectation(without_rollback, v, dict(status='x'), 'm',
                                                                          ['t'])
                              for n in PLANS for v in c.plan_servers(without_rollback, n) if v['kind'] == 'runtime')
                  and all(c.server_expectation(with_rollback, v, dict(status='x'), 'm', ['t'])[c.GOAL_ROLLBACK_KEY][
                      'strict'] is (v.get(c.CLIENT_KEY) == c.EDGE_CLIENT)
                      for n in PLANS for v in c.plan_servers(with_rollback, n) if v['kind'] == 'runtime'))
    checks.append(plan_refused('goal-rollback-not-true', lambda p: p['config']['plans']['keepedgegoal'].update(
        goal_rollback=1), 'keepedgegoal', 'true only'))
    checks.append(plan_refused('goal-rollback-on-a-full-server', lambda p: p['config']['plans']['keepedge'].update(
        goal_rollback=True), 'keepedge', 'Goal policy'))
    # 2026-11-16 (phase B, gdn-inplace-1116): the plan key gdn_inplace_crossing: --gdn-inplace-crossing exactly on the
    # ON plans of the pairs, the S + 1 layout record and Mamba pool, the expectation's guard; refused without the
    # sealed switch, as a non-true value, on a stock server
    inplace = [n for n in PLANS if any(c.inplace_on(PINS, v) for v in c.plan_servers(PINS, n))]
    print('GDN_INPLACE feature=%s plans=%s' % (json.dumps(features.get(c.INPLACE_KEY), sort_keys=True),
                                               ','.join(inplace) or 'none'))
    checks.append(inplace == ['gdnobson', 'gdngoalon', 'gdnidleon', 'gdnobs24on', 'gdnobsp', 'gdnobspi', 'gdngoalp',
                              'gdngoalpi', 'gdnidlep', 'gdnidlepi', 'gdnobs24p', 'gdnobs24pi'] and all(
        (c.INPLACE_SWITCH in c.launch_arguments(PINS, v, '/m')) == c.inplace_on(PINS, v)
        and (c.INPLACE_KEY in c.server_expectation(PINS, v, dict(status='x'), 'm', ['t'])) == c.inplace_on(PINS, v)
        for n in PLANS for v in c.plan_servers(PINS, n) if v['kind'] == 'runtime') and all(
        c.server_expectation(PINS, v, dict(status='x'), 'm', ['t'])['layout']['record'].get('mamba_slots')
        == c.plan_running(PINS, n) + 1 for n in inplace for v in c.plan_servers(PINS, n) if c.inplace_on(PINS, v)))
    # 2026-11-17: a server's own --admission-overlap K (sess24r, goal64ak24: 24)
    checks.append(plan_refused('server-admission-overlap-above-the-sealed-permits', lambda p: p['config']['plans'][
        'sess24r']['servers'][0].update(admission_overlap=33), 'sess24r', 'admission_overlap'))
    checks.append(plan_refused('server-admission-overlap-on-a-policy-without-it', lambda p: p['config']['plans'][
        'gdnobs']['servers'][0].update(admission_overlap=16), 'gdnobs', 'admission_overlap'))
    checks.append(plan_refused('gdn-inplace-server-key-1', lambda p: p['config']['plans']['gdnobsp']['servers'][1].update(
        gdn_inplace_crossing=1), 'gdnobsp', 'true only'))
    checks.append(plan_refused('gdn-inplace-server-key-without-the-switch', lambda p: p['runtime_features'].pop(
        c.INPLACE_KEY), 'gdnobsp', '--gdn-inplace-crossing'))
    checks.append(plan_refused('gdn-inplace-without-the-switch', lambda p: p['runtime_features'].pop(c.INPLACE_KEY),
                               'gdnobson', '--gdn-inplace-crossing'))
    checks.append(plan_refused('gdn-inplace-false', lambda p: p['config']['plans']['gdnobson']['server'].update(
        gdn_inplace_crossing=False), 'gdnobson', 'server may set'))
    checks.append(plan_refused('gdn-inplace-1', lambda p: p['config']['plans']['gdnobson']['server'].update(
        gdn_inplace_crossing=1), 'gdnobson', 'server may set'))
    # 2026-11-16 (phase B, sessions-32-1116): K = 24 plans: the agents' client timeout from K (4,400 s), the pool spec
    k24 = [n for n in PLANS_EARLIER if any(c.server_running(PINS, v) == 24 for v in c.plan_servers(PINS, n))]
    print('K24 plans=%s client_timeout=%s' % (','.join(k24), sorted({c.client_timeout(PINS, v) for n in k24
                                                                      for v in c.plan_servers(PINS, n)})))
    checks.append(k24 == ['full64ak24', 'recent64ak24', 'obs64ak24', 'goal64ak24', 'ledger64ak24', 'stockkeep64k24',
                          'gdnobs24', 'gdnobs24on', 'gdnobs24p', 'gdnobs24pi', 'sess24', 'sess24i', 'sess24r',
                          'sess24ri', 'full150k24', 'ledger150k24', 'stockkeep150k24']
                  and all(c.plan_running(PINS, n) == 24 for n in k24)
                  and all(c.client_timeout(PINS, v) == (4400 if v['kind'] == 'runtime' else 3600)
                          for n in k24 for v in c.plan_servers(PINS, n)))
    checks.append(plan_refused('edge-client-unknown', lambda p: p['config']['plans']['keepedge']['servers'][0].update(
        client='agent'), 'keepedge', 'client'))
    checks.append(plan_refused('edge-client-thinking-off', lambda p: p['config']['plans']['keepedge'].update(
        thinking=False), 'keepedge', 'thinking on'))
    checks.append(plan_refused('edge-module-elsewhere', lambda p: p['config']['agent']['edge_script'][
        'agent_module'].update(tbench_path='/tmp/swe32_edge_script.py'), 'keepedge', 'edge_script must pin'))
    # 2026-11-14 (perf-swe32-ledger): ledger_query (the query hook's release plug-in of 51b4085) and its plan key
    # ledger_cap_percent: the row as CHECKED expects it (hooked, under hook, the XML protocol only), the plans that run
    # it (ledger262t / ledger262 / ledger64t, the default cap: no flag), --ledger-cap-percent exactly where a server
    # names its cap; refused: a cap outside 1-100, not an integer, on another policy or a stock server, a runtime
    # without the switch, a row that is no hooked plug-in or serves another protocol, the policy with a rewriter
    ledgered = ['%s:%s:cap%s' % (n, s['label'], c.ledger_cap(PINS, s)) for n in PLANS for s in c.plan_servers(PINS, n)
                if s['policy'] == c.LEDGER]
    row = next(r for r in rows if r['name'] == c.LEDGER)
    print('LEDGER_QUERY feature=%s hooked=%s row=%s plans=%s' % (
        json.dumps(features.get('ledger'), sort_keys=True), ','.join(c.hooked_policies(PINS)),
        json.dumps({k: row.get(k) for k in ('kind', 'query_vector', 'hooked', 'evidence_key', 'protocols', 'budget')},
                   sort_keys=True), ','.join(ledgered)))
    checks.append(ledgered == ['ledgert:ledger262t:cap50', 'ledger:ledger262:cap50', 'ledger64:ledger64t:cap50',
                               'ledgerti:ledger262t:cap50', 'ledger64a:ledger64ta:cap50',
                               'ledger64ak24:ledger64ta24:cap50', 'ledger150k24:ledger150ta24:cap50']
                  and c.hooked_policies(PINS)[0] == c.LEDGER and c.policy_problem(PINS, c.LEDGER) is None
                  and (row['kind'], row['query_vector'], row['hooked'], row['evidence_key'], row['protocols'])
                  == ('query', 'observation', True, 'hook', ['mswea_xml_v1'])
                  and all((c.LEDGER_SWITCH in c.launch_arguments(PINS, s, '/m')) == (c.LEDGER_KEY in s)
                          for n in PLANS for s in c.plan_servers(PINS, n) if s['kind'] == 'runtime'))
    cap75 = variant(lambda p: p['config']['plans']['ledgert']['servers'][0].update(ledger_cap_percent=75))
    args75 = c.launch_arguments(cap75, c.plan_servers(cap75, 'ledgert')[0], '/m')
    checks.append(c.validate_plan(cap75, 'ledgert')['fits'] and args75[args75.index(c.LEDGER_SWITCH) + 1] == '75'
                  and c.ledger_pin(cap75, c.plan_servers(cap75, 'ledgert')[0])['ledger_cap_percent'] == 75)
    for value in (0, 101, -5, 50.0, '50', True):
        checks.append(plan_refused('ledger-cap-%r' % (value,), lambda p, v=value: p['config']['plans']['ledgert'][
            'servers'][0].update(ledger_cap_percent=v), 'ledgert', 'must be an integer 1-100'))
    checks.append(plan_refused('ledger-cap-on-observation-query', lambda p: p['config']['plans']['obst']['servers'][
        0].update(ledger_cap_percent=50), 'obst', 'applies to ledger_query only'))
    checks.append(plan_refused('ledger-cap-on-a-stock-server', lambda p: p['config']['plans']['stock32t']['servers'][
        0].update(ledger_cap_percent=50), 'stock32t', 'on a stock server'))
    checks.append(features_refused('ledger-without-the-switch', lambda p: p['runtime_features'].update(ledger=None),
                                   '--ledger-cap-percent'))
    checks.append(plan_refused('ledger-row-not-hooked', lambda p: first(p, c.LEDGER).update(hooked=False), 'ledgert',
                               'hooked'))
    checks.append(plan_refused('ledger-row-other-record-key', lambda p: first(p, c.LEDGER).update(evidence_key='window'),
                               'ledgert', 'evidence_key'))
    checks.append(plan_refused('ledger-row-kv-session-only', lambda p: first(p, c.LEDGER).update(
        protocols=['kv_session_v1']), 'ledgert', 'serves the protocols'))
    checks.append(plan_refused('ledger-row-other-default-pin', lambda p: first(p, c.LEDGER).update(pin=dict(
        c.LEDGER_PIN, reasoning_turns=3)), 'ledgert', 'pin'))
    checks.append(plan_refused('ledger-with-a-rewriter', lambda p: p['config']['policies'][c.LEDGER].update(
        rewriter=True), 'ledgert', 'only goal_query starts the rewriter'))
    checks.append(plan_refused('ledger-without-a-budget', lambda p: p['config']['policies'][c.LEDGER].update(
        policy_args=[]), 'ledgert', 'takes --active-token-budget'))
    checks.append(plan_refused('ledger-unregistered', lambda p: p['runtime_features'].update({c.REGISTRY_KEY: [
        r for r in rows if r['name'] != c.LEDGER]}), 'ledgert', 'does not register'))
    # 2026-11-17: release gate 2 (parity150): exactly full150k24's runtime server and stockkeep150k24's stock server
    # (their launch / stock arguments), both with the probe client, the sealed prompts, role release-gate
    import parity
    for name in GATE2:
        ours, theirs = c.plan_servers(PINS, name)
        settings = c.parity_settings(PINS, name)
        print('GATE2 plan=%s servers=%s:%s,%s:%s prompts=%d window=%d logprobs=%d near_tie_nats=%s resyncs=%d '
              'prompts_sha256=%s' % (name, ours['label'], ours['policy'], theirs['label'], theirs['policy'],
                                     settings['prompts'], settings['window'], settings['logprobs'],
                                     settings['near_tie_nats'], settings['resyncs'], settings['prompts_sha256']))
    checks.append(GATE2 == ['parity150'] and all(
        c.launch_arguments(PINS, c.plan_servers(PINS, n)[0], '/m')
        == c.launch_arguments(PINS, c.plan_servers(PINS, 'full150k24')[0], '/m')
        and c.stock_argv(PINS, 8000, c.plan_servers(PINS, n)[1])
        == c.stock_argv(PINS, 8000, c.plan_servers(PINS, 'stockkeep150k24')[0])
        and all(s.get(c.CLIENT_KEY) == c.PARITY_CLIENT for s in c.plan_servers(PINS, n))
        and c.cfg(PINS)['plans'][n]['role'] == c.PARITY_ROLE for n in GATE2))
    g = lambda p: p['config']['plans']['parity150']
    checks.append(plan_refused('gate2-window', lambda p: g(p)['parity'].update(window=8), 'parity150', 'window must be'))
    checks.append(plan_refused('gate2-prompts-sha', lambda p: g(p)['parity'].update(prompts_sha256='0' * 64),
                               'parity150', 'differs from the plan'))
    checks.append(plan_refused('gate2-order', lambda p: g(p)['servers'].reverse(), 'parity150',
                               'exactly our full server then the stock control'))
    checks.append(plan_refused('gate2-other-policy', lambda p: g(p)['servers'][0].update(policy='observation_query'),
                               'parity150', 'exactly our full server'))
    checks.append(plan_refused('gate2-role', lambda p: g(p).update(role='validation'), 'parity150', 'role release-gate'))
    checks.append(plan_refused('gate2-without-settings', lambda p: g(p).pop('parity'), 'parity150',
                               'own server settings'))
    checks.append(plan_refused('gate2-client-elsewhere', lambda p: p['config']['plans']['full150k24']['servers'][
        0].update(client=c.PARITY_CLIENT), 'full150k24', 'servers of a gate-2 plan only'))
    checks.append(plan_refused('gate2-role-elsewhere', lambda p: p['config']['plans']['full150k24'].update(
        role=c.PARITY_ROLE), 'full150k24', 'without parity settings'))
    bad = sum(1 for x in checks if not x)
    ok &= bad == 0
    print('%s plans=%d refusals=%d bad=%d' % ('GATE_CHECKS_PASS' if ok else 'GATE_CHECKS_FAIL', len(PLANS),
                                              len(checks) - 1, bad))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
