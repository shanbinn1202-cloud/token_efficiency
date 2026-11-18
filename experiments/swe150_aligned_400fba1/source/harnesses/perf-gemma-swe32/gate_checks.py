"""Local only (never delivered): the sealed plans' gates of perf-gemma-swe32, printed for the validation log.

For every plan (full, goal, obs, rfill, stock: one ghx4 batch job each, Gemma 4 31B IT; fulli, obsi, rfilli,
stocki: their ghx4-interactive variants): controller.validate_plan
(phrase, partition, the server, K = 16, the pinned host, the window pools at the policy's KV size, the GPU-memory rule,
the memory fit, the time fit, the limit within 1.5 x the expected time), its TIME line (limit, expected, agent cap and
minimum, worst case, with the evaluation wait), its MEMORY line (--mem=113560M; ours 1-GiB connector pool + the 72-GiB
arena in 204-Page chunks (72.22) + 32-GiB reserve = 105.22 GiB, 5.68 GiB spare, at least 5 required; stock 32 GiB), its
GPU line (the estimated peak at the policy's KV size and what it leaves of 97,871 MiB), its WINDOW line (pool 0 / pool 1
and the launcher's selective or release rule), STARTUP_SPIKE (ELIMINATED: --language-model-only on every server) and
COST (GPU-hours at the limit and at the expected time; the total). Then the refusals the gates must make: memory
(--mem above the share: 113561M / 144g / 200g; a 73-GiB arena (4.69 spare), the 32-GiB connector pool, the 34-GiB
reserve: each inside the 5-GiB margin; an arena below 1.5 x the replayed 37.75 GiB or below the 65.25-GiB bound of one
run's 16 largest Gemma sessions; a server without --language-model-only), the KV size (the user's decision of
2026-10-21: observation_query / random_fill / goal_query at another size than 25 GiB; 25 GiB on a runtime without the
selective rule's request_pages: vLLM's startup check needs 28.95 / 30.51 GiB at 16 sessions and 131,072 tokens; full
at 28 GiB: serve.py's release rule needs 9,083 global Pages), time (a cap that overruns the limit, a minimum that does
not fit after the evaluation wait, a limit above 1.5 x the expected time, a phrase naming other minutes, INTERACTIVE
on ghx4 or BATCH on ghx4-interactive), the light login gates (no pinned release-check receipt, another commit's, a
cloned HEAD other than the receipt's),
concurrency (agents, stock or overlap not K), the Gemma profile and features (a runtime without the Gemma profile, with
another geometry or KV choices, a config naming another model revision, full off the profile's pinned 32 GiB, a KV size
off 24 / 25 / 28 / 32, a runtime without --release-idle-pages, --language-model-only, --connector-pool-gib or
--decode-fastpath, the fastpath2 switch without --decode-fastpath, a plan asking for --goal-rewriter-batch, the
--phase1-in-step admission flag), --async-scheduling (the 2026-11-09 re-pin, plan smokea: async_scheduling as a
server key or not a boolean, on a plan with the stock control, on a pin without the switch or whose Gemma profile refuses
it, without --decode-fastpath2, and NOT_SUBMITTABLE on a runtime without the async fixes a0798a8 / 4ed1013 / 13ee618) and the
queue gate (perf-swe32's tbench chain: a queued gswe32 / swe32 / swe10 /
swe16 / tb16 / gem0 job this job would not wait for, -AfterJob naming no queued job, an interactive plan behind a queued
tbench job or with -AfterJob, a second interactive job, squeue failing; the chain's last job named: pass).
Exit 0 and GATE_CHECKS_PASS iff every plan fits and every refusal happens.
"""
import copy
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
    print('RUNTIME commit=%s decode_fastpath=%s decode_fastpath2=%s (switch %s) connector_pool=%s language_model_only=%s '
          'goal_text=%s goal_rewriter_timing=%s goal_rewriter_batch=%s (plans %s)' % (
              PINS['commit'], bool(features.get('decode_fastpath')), bool(features.get('decode_fastpath2')),
              'ON' if c.decode_fastpath2_on(PINS) else 'OFF', c.connector_pool_gib(PINS),
              bool(features.get('language_model_only')), (features.get('goal_text') or {}).get('files'),
              bool(features.get('goal_rewriter_timing')), bool(features.get('goal_rewriter_batch')),
              ','.join('%s:%d' % (n, s[c.BATCH_KEY]) for n in PLANS for s in c.plan_servers(PINS, n)
                       if s.get(c.BATCH_KEY)) or 'none'))
    c.check_features(features, PINS['config'])
    hours = [0.0, 0.0]
    for name in PLANS:
        fit = c.validate_plan(PINS, name)
        memory, cost = c.memory_fit(PINS, name), c.plan_cost(PINS, name)
        server = c.plan_servers(PINS, name)[0]
        if c.cfg(PINS)['plans'][name]['partition'] == 'ghx4':  # an interactive variant replaces its batch plan
            hours[0] += cost['gpu_hours_limit']
            hours[1] += cost['gpu_hours_expected']
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
        gpu = c.gpu_fit(PINS, name)
        print('GPU plan=%s kv_cache_gib=%s peak_mib=%d free_mib=%d largest_step=%s pinned_by_runtime=%s fits=%s' % (
            name, gpu['kv_cache_gib'], gpu['peak_mib'], gpu['free_mib'], gpu['largest_step'], gpu['pinned'],
            gpu['fits']))
        if server['kind'] == 'runtime':
            window = c.window_fit(PINS, c.LAYOUT, c.running_of(PINS), server)
            print('WINDOW plan=%s rule=%s pool0=%d pool1=%d (%.2f GiB) needs=%d usable=%d fits=%s' % (
                name, window['rule'], window['pool0_pages'], window['pool1_blocks'], window['pool1_gib'],
                window['needed_pages'], window['usable_pages'], window['fits']))
            ok &= window['fits']
        ok &= gpu['fits']
        print('STARTUP_SPIKE plan=%s %s' % (name, memory['startup']['status']))
        print('COST plan=%s gpu_hours_limit=%.2f gpu_hours_expected=%.2f billed=x%d' % (
            name, cost['gpu_hours_limit'], cost['gpu_hours_expected'], cost['billing_factor']))
        ok &= fit['fits'] and memory['fits'] and memory['startup']['status'] == 'ELIMINATED' \
            and memory['headroom_gib'] >= 5
    print('COST_TOTAL batch plans gpu_hours_limit=%.2f gpu_hours_expected=%.2f' % tuple(hours))

    def server(plan):
        return lambda pins: pins['config']['plans'][plan]['servers'][0]

    def smoke_goal(pins):  # the release smoke's goal_query server (goalvllm4ks, the third)
        return pins['config']['plans']['smoke']['servers'][2]

    checks = [
        # memory
        plan_refused('mem 113561M (above the share)', lambda p: p['slurm'].update(mem='113561M'), 'full', 'share'),
        plan_refused('mem 144g', lambda p: p['slurm'].update(mem='144g'), 'stock', 'share'),
        plan_refused('mem 200g', lambda p: p['slurm'].update(mem='200g'), 'goal', 'share'),
        plan_refused('the runtime\'s arena 79 GiB', lambda p: p['config']['server'].update(dram_arena_gib=79), 'goal',
                     'stay spare'),
        plan_refused('arena 73 GiB (4.69 spare)', lambda p: p['config']['server'].update(dram_arena_gib=73), 'full',
                     'stay spare'),
        plan_refused('a 31-GiB connector pool', lambda p: p['config']['server'].update(connector_pool_gib=31),
                     'obs', 'host memory'),
        plan_refused('reserve 34 GiB', lambda p: p['config']['memory'].update(process_overhead_gib=34), 'rfill',
                     'stay spare'),
        plan_refused('arena 56 GiB (< 1.5 x 37.75)', lambda p: p['config']['server'].update(dram_arena_gib=56), 'goal',
                     'measured arena demand'),
        plan_refused('arena 65 GiB (< the 65.25-GiB bound)', lambda p: p['config']['server'].update(dram_arena_gib=65),
                     'goal', 'measured arena demand'),
        # GPU memory and the window pools (the user's KV decision of 2026-10-21; serve.py's pool-0 refusal; vLLM's
        # startup check without the selective rule's request_pages)
        plan_refused('observation_query at 28 GiB (not the decided 25)', lambda p: p['config']['policies'][
            'observation_query'].update(kv_cache_gib=28), 'obs', 'decision'),
        plan_refused('goal_query at 24 GiB (not the decided 25)', lambda p: p['config']['policies']['goal_query'].update(
            kv_cache_gib=24), 'goal', 'decision'),
        plan_refused('random_fill at 25 GiB without request_pages (vLLM needs 28.95)', lambda p: p['runtime_features'][
            'gemma4'].update(request_pages=False), 'rfill', 'startup check needs 28.95'),
        plan_refused('goal_query at 25 GiB without request_pages (vLLM needs 30.51)', lambda p: p['runtime_features'][
            'gemma4'].update(request_pages=False), 'goal', 'startup check needs 30.51'),
        plan_refused('full at 28 GiB (release rule 9,083 Pages)', lambda p: (
            p['config']['policies']['full'].update(kv_cache_gib=28), p['config']['gpu'].update(pinned_by_runtime=[])),
            'full', 'release rule'),
        plan_refused('stock without --language-model-only',
                     lambda p: p['config']['stock'].update(language_model_only=False), 'stock', 'multi-modal warmup'),
        plan_refused('ours without --language-model-only',
                     lambda p: p['config']['server'].update(language_model_only=False), 'full', 'multi-modal warmup'),
        # time
        plan_refused('cap overruns the limit', lambda p: server('full')(p).update(agent_cap_seconds=2281), 'full',
                     'does not fit'),
        plan_refused('minimum after a 900-s evaluation wait', lambda p: (p['config']['plans']['goal'].update(
            eval_wait_seconds=900), server('goal')(p).update(agent_min_seconds=5100)), 'goal', 'does not fit'),
        plan_refused('limit above 1.5 x expected', lambda p: p['config']['plans']['stock'].update(expected_seconds=2500),
                     'stock', 'tight limits'),
        plan_refused('phrase with other minutes', lambda p: p['config']['plans']['goal'].update(
            authorization=p['config']['plans']['goal']['authorization'].replace('_121MIN_', '_120MIN_')), 'goal',
                     'phrase'),
        plan_refused('phrase INTERACTIVE on ghx4', lambda p: p['config']['plans']['full'].update(
            authorization=p['config']['plans']['full']['authorization'].replace('_BATCH_', '_INTERACTIVE_')), 'full',
                     'phrase'),
        plan_refused('time above the ghx4-interactive maximum', lambda p: p['config']['plans']['goal'].update(
            partition='ghx4-interactive', time='02:30:00'), 'goal', 'maximum'),
        plan_refused('phrase BATCH on ghx4-interactive', lambda p: p['config']['plans']['obsi'].update(
            authorization=p['config']['plans']['obsi']['authorization'].replace('_INTERACTIVE_', '_BATCH_')), 'obsi',
                     'phrase'),
        plan_refused('the goal plan on ghx4-interactive (02:01:00)', lambda p: p['config']['plans']['goal'].update(
            partition='ghx4-interactive'), 'goal', 'maximum'),
        # light login gates: the pinned local release-check receipt
        refused('no pinned release-check receipt', lambda: c.check_release_receipt(variant(
            lambda p: p.pop('release_receipt'))), 'no pinned release-check receipt'),
        refused('another commit\'s receipt', lambda: c.check_release_receipt(variant(
            lambda p: p['release_receipt'].update(commit='0' * 40))), 'not the sealed'),
        refused('a failed receipt', lambda: c.check_release_receipt(variant(
            lambda p: p['release_receipt'].update(status='GIT_ARCHIVE_TESTS_FAIL'))), 'did not pass'),
        refused('a cloned HEAD other than the receipt\'s', lambda: c.check_release_receipt(PINS, '1' * 40),
                'cloned HEAD'),
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
        # the Gemma 4 profile (runtime 120bc3c, README 4.14)
        features_refused('a runtime without the Gemma profile', lambda p: p['runtime_features'].update(gemma4=None),
                         'Gemma 4 profile'),
        features_refused('another sealed geometry', lambda p: p['runtime_features']['gemma4'].update(
            global_token_bytes=65536), 'differs from this harness'),
        features_refused('other sealed KV choices', lambda p: p['runtime_features']['gemma4'].update(
            kv_cache_gib_choices=[24, 28, 30, 32]), 'differs from this harness'),
        features_refused('another model revision', lambda p: p['config']['model'].update(revision='0' * 40),
                         'config.model'),
        features_refused('full off the pinned 32 GiB', lambda p: p['config']['policies']['full'].update(kv_cache_gib=28),
                         'pinned --kv-cache-gib'),
        features_refused('a KV size off the steps (30)', lambda p: p['config']['policies']['goal_query'].update(
            kv_cache_gib=30), 'kv_cache_gib'),
        features_refused('a runtime without --release-idle-pages', lambda p: p['runtime_features'].update(
            release_idle_pages=None), 'release-idle-pages'),
        # wave 2 (9ef0d6d): goal's batch is the decision; refused where serve.py refuses it
        features_refused('the goal batch on a runtime without --goal-rewriter-batch', lambda p: p['runtime_features'].update(
            goal_rewriter_batch=None), 'goal-rewriter-batch'),
        plan_refused('the rewriter batch on observation_query', lambda p: server('obs')(p).update(goal_rewriter_batch=4),
                     'obs', 'needs goal_query'),
        plan_refused('goal batch 17 (above 16 and --admission-overlap 16)', lambda p: server('goal')(p).update(goal_rewriter_batch=17),
                     'goal', 'integer 2-16'),
        plan_refused('goal batch at 32 GiB without T', lambda p: (
            p['config']['policies']['goal_query'].update(kv_cache_gib=32),
            p['config']['gpu']['decided_kv_cache_gib'].update(goal_query=32),
            server('goal')(p).pop('goal_rewriter_batch_tokens')), 'goal', 'needs a token cap T'),
        # 2026-10-22: goal's explicit T must leave config.gpu.min_free_mib at the 3410948-calibrated estimate
        plan_refused('goal T 16384 (2,002 MiB free, below the 2-GiB rule)', lambda p: server('goal')(p).update(
            goal_rewriter_batch_tokens=16384), 'goal', 'tokens 16384 leaves 2002 MiB'),
        plan_refused('goal T 25600 (the profile default: 80 MiB free)', lambda p: server('goal')(p).pop(
            'goal_rewriter_batch_tokens'), 'goal', 'tokens 25600 leaves 80 MiB'),
        plan_refused('goal T 0 (no cap: an unbounded batch)', lambda p: server('goal')(p).update(
            goal_rewriter_batch_tokens=0), 'goal', 'tokens 0 leaves no MiB'),
        features_refused('a runtime without --decode-fastpath3', lambda p: p['runtime_features'].update(
            decode_fastpath3=None), 'decode-fastpath3'),
        features_refused('a runtime without --deferred-completion-io', lambda p: p['runtime_features'].update(
            deferred_completion_io=None), 'deferred-completion-io'),
        features_refused('a Gemma profile refusing --decode-fastpath3 (120bc3c)', lambda p: p['runtime_features'][
            'gemma4'].update(decode_fastpath3=False), 'Gemma 4 profile refuses'),
        features_refused('a Gemma profile refusing --deferred-completion-io (120bc3c)', lambda p: p['runtime_features'][
            'gemma4'].update(deferred_completion_io=False), 'Gemma 4 profile refuses'),
        features_refused('a Gemma profile refusing the batch (120bc3c)', lambda p: p['runtime_features'][
            'gemma4'].update(goal_rewriter_batch=False), 'Gemma 4 profile refuses'),
        features_refused('fastpath3 without fastpath2', lambda p: p['config']['server'].update(decode_fastpath2=False),
                         'requires decode_fastpath2'),
        features_refused('--phase1-in-step on goal_query', lambda p: p['config']['policies']['goal_query'].update(
            admission=['phase1_in_step', 'admission_overlap=16']), 'admission-overlap only'),
        # gemma-forward4k (integrate-1026): obs4k's --max-num-batched-tokens 4096, refused where serve.py refuses it
        plan_refused('obs4k on a Gemma profile without --max-num-batched-tokens (9ef0d6d)', lambda p: p[
            'runtime_features']['gemma4'].pop('max_num_batched_tokens', None), 'obs4k', 'lacks'),
        plan_refused('obs4k forward 2048 (not a choice)', lambda p: server('obs4k')(p).update(max_num_batched_tokens=2048),
                     'obs4k', 'must be one of'),
        plan_refused('obs4k forward 8192 named (the default: leave it out)', lambda p: server('obs4k')(p).update(
            max_num_batched_tokens=8192), 'obs4k', 'must be one of'),
        plan_refused('a 4,096 forward on the stock control', lambda p: server('stock')(p).update(
            max_num_batched_tokens=4096), 'stock', 'on a stock server'),
        # the release smoke (2026-11-03): the vLLM Goal rewriter, --goal-overlap, no async on Gemma, role release-smoke
        plan_refused('smoke: the vLLM rewriter with --goal-rewriter-batch', lambda p: smoke_goal(p).update(
            goal_rewriter_batch=4, goal_rewriter_batch_tokens=15360), 'smoke', 'excludes'),
        plan_refused('smoke: the vLLM rewriter without --admission-overlap K >= 2', lambda p: p['config']['policies'][
            'goal_query'].update(admission=[]), 'smoke', 'K >= 2'),
        plan_refused('smoke: the rewriter KV 1 GiB (2-16)', lambda p: smoke_goal(p).update(goal_rewriter_vllm_kv_gib=1),
                     'smoke', 'kv-gib'),
        plan_refused('smoke: --goal-overlap on observation_query', lambda p: p['config']['plans']['smoke']['servers'][
            0].update(goal_overlap=True), 'smoke', 'goal_query'),
        plan_refused('smoke: async_scheduling as a server key (a plan switch since the re-pin)', lambda p: smoke_goal(
            p).update(async_scheduling=True), 'smoke', 'plan switch, not a server key'),
        plan_refused('smoke: a Gemma profile without the vLLM rewriter', lambda p: p['runtime_features'].update(
            gemma4_goal=dict(goal_rewriter_engine=False, goal_overlap=True, async_scheduling=True)), 'smoke',
            'does not take --goal-rewriter-engine'),
        plan_refused('smoke: the child leaving less than min_free_mib', lambda p: p['config']['gpu'].update(
            vllm_rewriter_child_mib=5700), 'smoke', 'vLLM Goal rewriter child'),
        plan_refused('smoke: limit above 2.25 x expected (release-smoke)', lambda p: p['config']['plans']['smoke'].update(
            expected_seconds=4000), 'smoke', 'within 2.25 x it'),
        plan_refused('smoke without its role: 1.5 x', lambda p: p['config']['plans']['smoke'].pop('role'), 'smoke',
                     'within 1.5 x it'),
        plan_refused('smoke: another role', lambda p: p['config']['plans']['smoke'].update(role='table1'), 'smoke',
                     'plan role'),
        features_refused('smoke on a runtime without --goal-rewriter-engine', lambda p: p['runtime_features'].update(
            goal_rewriter_engine=None), 'goal-rewriter-engine'),
        features_refused('smoke on a runtime without --goal-overlap', lambda p: p['runtime_features'].update(
            goal_overlap=None), 'goal-overlap'),
        features_refused('async_scheduling as a server key (build)', lambda p: p['config']['plans']['smoke'][
            'servers'][1].update(async_scheduling=True), 'never a server key'),
        # the 2026-11-09 re-pin (main ba4d1ba): plan smokea's --async-scheduling (D-C) and the async-fix guard (D-G)
        plan_refused('smokea: async_scheduling not a boolean', lambda p: p['config']['plans']['smokea'].update(
            async_scheduling='true'), 'smokea', 'must be true or false'),
        plan_refused('stock: async_scheduling on a plan with the stock control', lambda p: p['config']['plans'][
            'stock'].update(async_scheduling=True), 'stock', 'without a stock control'),
        plan_refused('smokea: a pin without --async-scheduling (ea18256)', lambda p: p['runtime_features'].update(
            async_scheduling=None), 'smokea', 'no --async-scheduling'),
        plan_refused('smokea: a Gemma profile that refuses it (ea18256)', lambda p: p['runtime_features'][
            'gemma4_goal'].update(async_scheduling=False), 'smokea', 'does not take --async-scheduling'),
        plan_refused('smokea: NOT_SUBMITTABLE without the async fixes (D-G)', lambda p: p['runtime_features'].update(
            async_fix=None), 'smokea', 'NOT_SUBMITTABLE'),
        plan_refused('smokea: --async-scheduling without --decode-fastpath2', lambda p: p['config']['server'].update(
            decode_fastpath2=False, decode_fastpath3=False), 'smokea', 'requires --decode-fastpath2'),
        features_refused('smokea on a runtime without the async fixes (D-G: NOT_SUBMITTABLE)', lambda p: p[
            'runtime_features'].update(async_fix=None), 'NOT_SUBMITTABLE'),
        features_refused('smokea on a runtime whose Gemma profile refuses it (ea18256)', lambda p: p['runtime_features'][
            'gemma4_goal'].update(async_scheduling=False), 'ask for --async-scheduling'),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        plan = c.cfg(PINS)['plans']['goal']
        for i, other in enumerate(('gswe32-full', 'swe32-goal', 'swe10-goal', 'swe16-stock', 'tb16-tbr', 'gem0-a1')):
            checks.append(refused('queue gate: %s queued' % other, lambda other=other, i=i: c.queue_gate(
                Path(tmp), PINS, plan, dict(plan='goal', after_job=None), 'q%d.json' % i,
                runner=lambda *a, **k: Run('777|PENDING|ghx4|%s\n' % other), user='ycao11'), 'queue holds'))
        checks.append(refused('queue gate: squeue failing', lambda: c.queue_gate(
            Path(tmp), PINS, plan, dict(plan='goal', after_job=None), 'qf.json',
            runner=lambda *a, **k: Run('', 1), user='ycao11'), 'squeue failed'))
        value = c.queue_gate(Path(tmp), PINS, plan, dict(plan='goal', after_job='777'), 'qa.json',
                             runner=lambda *a, **k: Run('777|RUNNING|ghx4|gswe32-full\n888|PENDING|ghx4|other\n'),
                             user='ycao11')
        print('QUEUE_GATE named -AfterJob 777: %s' % value['status'])
        checks.append(value['status'] == 'QUEUE_GATE_PASS')
        # the tbench chain (squeue %E): 100 gem0 running, 200 swe32 after 100, 300 tb16 after 200
        chain = ('100|RUNNING|ghx4|gem0-a1|(null)\n200|PENDING|ghx4|swe32-full|afterany:100(unfulfilled)\n'
                 '300|PENDING|ghx4|tb16-tbr|afterany:200(unfulfilled)\n')
        gate = lambda plan, name, after, out, label: c.queue_gate(
            Path(tmp), PINS, c.cfg(PINS)['plans'][plan], dict(plan=plan, after_job=after), label,
            runner=lambda *a, **k: Run(out), user='ycao11')
        value = gate('full', 'full', '300', chain, 'qc.json')
        print('QUEUE_GATE chain head 300 (waits for 100, 200): %s chain=%s' % (value['status'], value['chain']))
        checks.append(value['status'] == 'QUEUE_GATE_PASS' and value['chain'] == ['100', '200', '300'])
        checks.append(refused('queue gate: -AfterJob 200 leaves 300 beside it', lambda: gate(
            'full', 'full', '200', chain, 'qd.json'), 'would not wait for'))
        checks.append(refused('queue gate: -AfterJob names no queued job', lambda: gate(
            'obs', 'obs', '999', chain, 'qe.json'), 'names no job of yours'))
        checks.append(refused('queue gate: an interactive plan behind a queued tbench job', lambda: gate(
            'obsi', 'obsi', None, chain, 'qf2.json'), 'cannot join a chain'))
        checks.append(refused('queue gate: an interactive plan with -AfterJob', lambda: gate(
            'fulli', 'fulli', '300', chain, 'qg.json'), 'takes no -AfterJob'))
        checks.append(refused('queue gate: a second interactive job', lambda: gate(
            'stocki', 'stocki', None, '500|RUNNING|ghx4-interactive|sctx2-x|(null)\n', 'qh.json'),
            'at most 1 submitted job'))
        value = gate('rfill', 'rfill', '400', '400|RUNNING|ghx4-interactive|gswe32-obsi|(null)\n', 'qi.json')
        print('QUEUE_GATE batch plan behind an interactive head: %s' % value['status'])
        checks.append(value['status'] == 'QUEUE_GATE_PASS')
    # the switch on (the user's decision after spd2, 2026-10-16: the sealed config) and off: both valid on this pin
    on = variant(lambda p: p['config']['server'].update(decode_fastpath2=True))
    c.check_features(on['runtime_features'], on['config'])
    args = c.launch_arguments(on, c.plan_servers(on, 'full')[0], '/m')
    print('FASTPATH2_SWITCH_ON sealed=%s valid=%s args_tail=%s' % (
        c.decode_fastpath2_on(PINS), all(c.validate_plan(on, n)['fits'] for n in PLANS), args[-4:]))
    checks.append(args.count('--decode-fastpath2') == 1 and c.decode_fastpath2_on(PINS))
    # (and smokea's --async-scheduling with it: serve.py refuses async without --decode-fastpath2; the 2026-11-09 re-pin)
    off = variant(lambda p: [p['config']['server'].update(decode_fastpath2=False, decode_fastpath3=False)] + [
        plan.update(async_scheduling=False) for plan in p['config']['plans'].values() if c.ASYNC_KEY in plan])
    c.check_features(off['runtime_features'], off['config'])
    args = c.launch_arguments(off, c.plan_servers(off, 'full')[0], '/m')
    print('FASTPATH2_SWITCH_OFF valid=%s args_tail=%s' % (all(c.validate_plan(off, n)['fits'] for n in PLANS), args[-3:]))
    checks.append('--decode-fastpath2' not in args)
    # the 2026-11-09 re-pin: --async-scheduling exactly on smokea's three servers (serve.py and vLLM), the async fixes
    # sealed; every other plan's servers run --no-async-scheduling (no switch, no key)
    asynchronous = ['%s:%s:%s' % (n, s['label'], 'on' if c.async_scheduling_on(PINS, s) else 'off') for n in PLANS
                    for s in c.plan_servers(PINS, n) if c.ASYNC_KEY in s]
    print('ASYNC feature=%s gemma4=%s async_fix=%s plans=%s every_other_plan=--no-async-scheduling' % (
        bool(features.get(c.ASYNC_KEY)), (features.get('gemma4_goal') or {}).get(c.ASYNC_KEY),
        bool(features.get(c.ASYNC_FIX_KEY)), ','.join(asynchronous)))
    checks.append(asynchronous == ['smokea:obs4ks:on', 'smokea:rfill4ks:on', 'smokea:goalvllm4ks:on']
                  and all((c.ASYNC_SWITCH in c.launch_arguments(PINS, s, '/m')) == c.async_scheduling_on(PINS, s)
                          for n in PLANS for s in c.plan_servers(PINS, n) if s['kind'] == 'runtime'))
    bad = sum(1 for x in checks if not x)
    ok &= bad == 0
    print('%s plans=%d refusals=%d bad=%d' % ('GATE_CHECKS_PASS' if ok else 'GATE_CHECKS_FAIL', len(PLANS),
                                              len(checks) - 1, bad))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
