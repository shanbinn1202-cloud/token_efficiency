"""Local only (never delivered): the sealed plans' gates of perf-tb16, printed for the validation log.

For every plan (full, obs, rfill, stock: one ghx4 batch job each on the 89 tasks; dry: two servers on the 16-task
subset; dryi: the same on ghx4-interactive, urgent, billed 2x; goal: goal_query on the 89 tasks, goaldry / goaldryi on the
subset (2026-10-21)): controller.validate_plan (phrase, partition, the servers, Harbor -n 16 =
stock seqs <= K, the Terminus model_info = the servers' context and reply bound, one attempt without Harbor retries,
the harbor minimum covering the longest task timeout, the memory fit, the time fit, the limit within 1.5 x the expected
time), its TIME, MEMORY, STARTUP_SPIKE and COST lines and the total. Then the refusals the gates must make: memory
(--mem above the share, the 32-GiB default connector pool, a larger reserve, a server without --language-model-only,
K = 20 with the arena of every open session at the limit), time (a cap that overruns the limit, a minimum below
build-pov-ray's 12,000-s timeout, a limit above 1.5 x the expected time, a phrase naming other minutes, tasks or
INTERACTIVE, the ghx4-interactive maximum), concurrency and the agent (Harbor -n above K, stock seqs not Harbor -n,
model_info or max_tokens not the servers', two attempts, Harbor retries, a timeout multiplier), the KV budget (below one
Page), the stock control's per-call template switch (thinking on, an unknown key, other call kwargs; no server flag,
the job config differing from ours by llm_call_kwargs only), a blocked plan and the unplanned goal plan, the runtime pin
(PLACEHOLDER without a chat protocol, with another header, without observation_query, with too few or too many slots,
with an unaccepted module; PINNED only with every pin and the accepted module), the Harbor spec size, the queue gate (a
tb16 / swe32 / swe10 / swe16 / gem0 job queued, squeue failing; the -AfterJob job accepted; a job PENDING on the
user's hold (JobHeldUser) not blocking and recorded with its release, an admin hold or a RUNNING one blocking; dryi:
one ghx4-interactive job of the user, of any name or held, blocking; a ghx4-interactive plan without an urgent reason
refused; the billed GPU-hours x2), --release-idle-pages (a runtime without the switch, a non-boolean value), the
stall watchdog's budget (too short, polled too rarely, no poll), and K = 20 with
--dram-arena-gib 70 fitting (the user's ~20 slots, once the runtime serves them).
The optimized runtime and goal_query (2026-10-21): --decode-fastpath3 without --decode-fastpath2 or on a runtime without
the switch, --deferred-completion-io on a runtime without it, a Gemma server with either on a Gemma launcher that
refuses it (before integrate-1021), goal plans on a runtime without the chat-text Goal inputs or with Terminus 2 pins
other than the pinned Harbor RECORD's, --goal-rewriter-batch on another policy, above the overlap or without the
policy's batch; the queue gate's submission chain (-AfterJob naming the last job: the jobs it waits for pass).
The integrate-1021 9ef0d6d re-pin (2026-10-22): a Gemma goal plan on a launcher without Gemma goal_query (unless
blocked), the Gemma goal server at K = 18 on 25 GiB ('use at most 17', its Goal transient), its T at serve.py's
default 25,600 or absent (the GPU estimate below 2 GiB free) and at 32 GiB without an explicit T, an explicit T on a
Qwen server, admission flags on a Gemma server other than goal_query. The 53e3b7a re-pin (2026-10-22, job 3429670):
goal plans on a runtime without the linear-time quoted-literal pass of the Goal features.
The Gemma plans (gfull, gobs, grfill, gstock, gdry; the user's decision of 2026-10-21: 25 GiB with K = 17 for the
selective policies, full at 32 GiB with K = 20): their pool-0 rule (full without its own 32 GiB, at 25 GiB with K = 17,
refused with the launcher's 'use at most 16 sessions'; observation_query at K = 20 on 25 GiB refused, 'use at most 19';
an observation_query server without --release-idle-pages refused by the strict rule; the most K for full at 32 GiB
reported: 24, the chat-text cap), the launcher's host rule (an 80-GiB arena refused), the
harness reserve (46 GiB refused), a phrase without GEMMA4, goal_query and admission flags on a Gemma server, unknown
config.gemma keys, a stock section without TRITON_ATTN or with another context, the stock seqs not Harbor -n, K above
the chat-text cap, the Gemma pin (PLACEHOLDER without Gemma's turns or without a Gemma chat-text launcher, once the
Gemma plans are unblocked; while they are blocked, pending the user's decisions, the pin does not need them).
The light login gates (2026-10-18): the pinned release-check receipt (release_receipt_gate refuses no receipt, a
receipt of another commit, another status, no tests, a malformed SHA-256; passes the pinned one with login_test_runs
0), setup records it before the claim, qualify runs no test suite (no unittest, no check_git_release) and none of the
dropped probes (IPC, sampler, sampling), and keeps the bundle verification, git fsck, S0 / S1, the official RECORD,
the compiler probe, the serve.py dry runs, the stock login gate and the tbench preflight.
Exit 0 and GATE_CHECKS_PASS iff every plan fits and every refusal happens.
"""
import copy
import inspect
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
import fake_stage  # noqa: E402

PINS = c.load(ROOT / 'pins.json')
PLANS = c.cfg(PINS)['plan_order']


def variant(change, pins=None):
    pins = copy.deepcopy(pins or PINS)
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


def unblock(pins):
    """The Gemma plans as they run once the user's pending decisions are made (no `blocked` reason)."""
    for plan in pins['config']['plans'].values():
        if plan.get('model') == c.GEMMA_MODEL:
            plan.pop('blocked', None)
    return pins


def block(pins):
    """Every Gemma plan blocked (as they were before the user's decision of 2026-10-21)."""
    for plan in pins['config']['plans'].values():
        if plan.get('model') == c.GEMMA_MODEL:
            plan['blocked'] = 'test: blocked'
    return pins


def plan_refused(name, change, plan='full', pattern=''):
    pins = variant(change)
    return refused(name, lambda: c.validate_plan(pins, plan), pattern)


class Run:
    def __init__(self, stdout, code=0):
        self.returncode, self.stdout, self.stderr = code, stdout, '' if code == 0 else 'slurm down'


def chat_feature(served=None, **constants):
    """A chat feature as controller.chat_text_feature reads it: the constants (fake_stage.CHAT_CONSTANTS, the pinned
    POLICIES) and, where the pinned module defines policies() (the policy registry, 2026-11-12) or `served` is given,
    the served policies (default: the pinned ones)."""
    value = dict(serve_arg='chat-text', protocol='chat_text_v1', module='source-overlay/remote_ra_live_v2/chat_text_protocol.py',
                 module_sha256='a' * 64, constants=copy.deepcopy(fake_stage.CHAT_CONSTANTS))
    value['constants'].update(constants)
    if served is not None or fake_stage.PINNED_SERVED is not None:
        value['policies'] = list(served if served is not None else fake_stage.PINNED_SERVED)
    return value


def pin_case(name, feature, config=None, accepted=None, want='PLACEHOLDER', pattern=''):
    saved = c.INTERFACE_ACCEPTED
    c.INTERFACE_ACCEPTED = accepted
    try:
        features = dict(PINS['runtime_features'], chat_text=feature, session_limits=dict(
            PINS['runtime_features'].get('session_limits') or {}, max_chat_text_running=24))
        status, reason = c.pin_status(features, config or PINS['config'])
    finally:
        c.INTERFACE_ACCEPTED = saved
    ok = status == want and pattern in str(reason)
    print('PIN %s %s: %s %s' % (name, 'as expected' if ok else 'UNEXPECTED', status, str(reason)[:200]))
    return ok


def main():
    ok = True
    features = PINS['runtime_features']
    status, reason = c.pin_status(features, PINS['config'])
    print('RUNTIME commit=%s pin=%s (%s) chat_text=%s decode_fastpath=%s connector_pool=%s language_model_only=%s '
          'kv_budget=%s' % (PINS['commit'], status, reason, (features.get('chat_text') or {}).get('protocol'),
                            bool(features.get('decode_fastpath')), c.connector_pool_gib(PINS),
                            bool(features.get('language_model_only')), c.cfg(PINS)['selection']['active_token_budget']))
    c.check_features(features, PINS['config'], placeholder=status != c.PIN_PINNED)
    hours, runnable = [0.0, 0.0], [0.0, 0.0]
    for name in PLANS:
        fit = c.validate_plan(PINS, name)
        memory, cost = c.memory_fit(PINS, name), c.plan_cost(PINS, name)
        servers = c.plan_servers(PINS, name)
        blocked = c.plan_blocked(PINS, name)  # (the sealed text, or the Qwen launcher's KV sizes: ctx25)
        hours[0] += cost['gpu_hours_limit']
        hours[1] += cost['gpu_hours_expected']
        if not blocked:
            runnable[0] += cost['gpu_hours_limit']
            runnable[1] += cost['gpu_hours_expected']
        print('TIME plan=%s servers=%s partition=%s time=%s (%d s) expected=%d s (x%.2f) caps=%s mins=%s worst_case=%d s '
              'with_eval_wait=%d s fits=%s%s' % (
                  name, ','.join('%s:%s:%s' % (s['label'], s['policy'], s['task_set']) for s in servers),
                  c.cfg(PINS)['plans'][name]['partition'], c.cfg(PINS)['plans'][name]['time'], fit['time_seconds'],
                  cost['expected_seconds'], cost['limit_over_expected'],
                  '/'.join(str(s['agent_cap_seconds']) for s in servers),
                  '/'.join(str(s['agent_min_seconds']) for s in servers), fit['worst_case_seconds'],
                  fit['with_eval_wait_seconds'], fit['fits'], ' BLOCKED' if blocked else ''))
        print('MEMORY plan=%s mem=%s need=%.2f GiB (pinned %.2f + reserve %s) spare=%.2f GiB (min %s) fits=%s' % (
            name, PINS['slurm']['mem'], memory['need_gib'], memory['pinned_host_gib'], memory['overhead_gib'],
            memory['headroom_gib'], memory['min_spare_gib'], memory['fits']))
        print('STARTUP_SPIKE plan=%s %s' % (name, memory['startup']['status']))
        print('COST plan=%s partition=%s gpu_hours_limit=%.2f gpu_hours_expected=%.2f billing=x%d '
              'billed_gpu_hours_limit=%.2f billed_gpu_hours_expected=%.2f' % (
                  name, c.cfg(PINS)['plans'][name]['partition'], cost['gpu_hours_limit'], cost['gpu_hours_expected'],
                  cost['billing_factor'], cost['gpu_hours_billed_limit'], cost['gpu_hours_billed_expected']))
        ok &= cost['billing_factor'] == (2 if c.cfg(PINS)['plans'][name]['partition'] == 'ghx4-interactive' else 1)
        ok &= fit['fits'] and memory['fits'] and memory['startup']['status'] == 'ELIMINATED' \
            and memory['headroom_gib'] >= 5
    print('COST_TOTAL gpu_hours_limit=%.2f gpu_hours_expected=%.2f; without the blocked: %.2f / %.2f'
          % (tuple(hours) + tuple(runnable)))

    def server(plan, k=0):
        return lambda pins: pins['config']['plans'][plan]['servers'][k]

    checks = [
        # memory
        plan_refused('mem 113561M (above the share)', lambda p: p['slurm'].update(mem='113561M'), 'full', 'share'),
        plan_refused('mem 144g', lambda p: p['slurm'].update(mem='144g'), 'stock', 'share'),
        plan_refused('the 32-GiB default connector pool', lambda p: p['config']['server'].pop('connector_pool_gib'),
                     'obs', 'stay spare'),
        plan_refused('reserve 39 GiB', lambda p: p['config']['memory'].update(process_overhead_gib=39), 'rfill',
                     'stay spare'),
        plan_refused('stock without --language-model-only',
                     lambda p: p['config']['stock'].update(language_model_only=False), 'stock', 'multi-modal warmup'),
        plan_refused('ours without --language-model-only',
                     lambda p: p['config']['server'].update(language_model_only=False), 'full', 'multi-modal warmup'),
        # time
        plan_refused('cap overruns the limit', lambda p: server('full')(p).update(agent_cap_seconds=18000), 'full',
                     'does not fit'),
        plan_refused('minimum below build-pov-ray', lambda p: server('obs')(p).update(agent_min_seconds=11000), 'obs',
                     'longest agent timeout'),
        plan_refused('dry minimum below its 900-s tasks', lambda p: server('dry', 1)(p).update(agent_min_seconds=800),
                     'dry', 'longest agent timeout'),
        plan_refused('limit above 1.5 x expected', lambda p: p['config']['plans']['stock'].update(expected_seconds=10000),
                     'stock', 'tight limits'),
        plan_refused('phrase with other minutes', lambda p: p['config']['plans']['full'].update(
            authorization=p['config']['plans']['full']['authorization'].replace('_302MIN_', '_300MIN_')), 'full',
                     'phrase'),
        plan_refused('phrase INTERACTIVE on ghx4', lambda p: p['config']['plans']['rfill'].update(
            authorization=p['config']['plans']['rfill']['authorization'].replace('_BATCH_', '_INTERACTIVE_')), 'rfill',
                     'phrase'),
        plan_refused('phrase naming 16 tasks for 89', lambda p: p['config']['plans']['obs'].update(
            authorization=p['config']['plans']['obs']['authorization'].replace('_89TASKS_', '_16TASKS_')), 'obs',
                     'phrase'),
        plan_refused('time above the ghx4-interactive maximum', lambda p: p['config']['plans']['full'].update(
            partition='ghx4-interactive', urgent='urgent'), 'full', 'maximum'),
        plan_refused('ghx4-interactive without an urgent reason', lambda p: p['config']['plans']['dryi'].pop('urgent'),
                     'dryi', 'urgent work only'),
        plan_refused('dryi phrase BATCH', lambda p: p['config']['plans']['dryi'].update(
            authorization=p['config']['plans']['dryi']['authorization'].replace('_INTERACTIVE_', '_BATCH_')), 'dryi',
                     'phrase'),
        plan_refused('dryi above the ghx4-interactive maximum', lambda p: p['config']['plans']['dryi'].update(
            time='02:01:00', authorization=p['config']['plans']['dryi']['authorization'].replace('_86MIN_',
                                                                                                 '_121MIN_')),
                     'dryi', 'maximum'),
        plan_refused('ghx4-interactive billing factor 1.5', lambda p: p['slurm']['partition_limits']['ghx4-interactive'].update(
            billing_factor=1.5), 'dryi', 'billing_factor'),
        # concurrency and the agent
        plan_refused('Harbor -n above K', lambda p: (p['config']['agent'].update(n_concurrent=21),
                                                     p['config']['stock'].update(max_num_seqs=21)), 'full', 'at most'),
        plan_refused('stock seqs not Harbor -n', lambda p: p['config']['stock'].update(max_num_seqs=20), 'stock',
                     'must equal'),
        plan_refused('K = 20 with the arena of every session at the limit', lambda p: p['config']['server'].update(
            dram_arena_gib=None), 'obs', 'stay spare'),
        plan_refused('model_info 131072', lambda p: p['config']['agent']['model_info'].update(max_input_tokens=131072),
                     'full', 'model_info'),
        plan_refused('llm max_tokens 8192', lambda p: p['config']['agent']['llm_kwargs'].update(max_tokens=8192),
                     'rfill', 'model_info'),
        plan_refused('two attempts', lambda p: p['config']['agent'].update(n_attempts=2), 'full'),
        # the stall watchdog (budget.stall_watchdog_seconds / stall_poll_seconds)
        plan_refused('stall watchdog 60 s', lambda p: p['config']['budget'].update(stall_watchdog_seconds=60), 'dryi',
                     'stall_watchdog_seconds'),
        plan_refused('stall watchdog polled twice', lambda p: p['config']['budget'].update(stall_poll_seconds=300),
                     'dry', 'stall_watchdog_seconds'),
        plan_refused('stall watchdog without a poll', lambda p: p['config']['budget'].pop('stall_poll_seconds'), 'full',
                     'stall_watchdog_seconds'),
        # --release-idle-pages: a sealed runtime without the switch, or a switch that is not a boolean
        refused('release without the sealed switch', lambda: c.check_features(
            dict(PINS['runtime_features'], release_idle_pages=None), PINS['config']), 'release-idle-pages'),
        refused('release_idle_pages "yes"', lambda: c.check_features(PINS['runtime_features'], variant(
            lambda p: p['config']['server'].update(release_idle_pages='yes'))['config']), 'release_idle_pages'),
        plan_refused('Harbor retries', lambda p: p['config']['agent'].update(harbor_retries=1), 'obs'),
        plan_refused('timeout multiplier 2', lambda p: p['config']['agent'].update(timeout_multiplier=2.0), 'stock'),
        # the KV budget
        plan_refused('active budget 8 tokens', lambda p: p['config']['selection'].update(active_token_budget=8), 'obs',
                     'active_token_budget'),
        # the stock control's per-call template switch (the server keeps vLLM's defaults, as SWE-bench)
        refused('stock thinking on', lambda: c.stock_template_kwargs(variant(lambda p: p['config']['stock'].update(
            llm_call_kwargs={'extra_body': {'chat_template_kwargs': {'enable_thinking': True}}}))),
                'enable_thinking false'),
        refused('stock template key unknown', lambda: c.stock_template_kwargs(variant(
            lambda p: p['config']['stock'].update(llm_call_kwargs={'extra_body': {'chat_template_kwargs': {
                'enable_thinking': False, 'reasoning_effort': 'low'}}}))), 'enable_thinking false'),
        refused('stock call kwargs beyond extra_body', lambda: c.stock_template_kwargs(variant(
            lambda p: p['config']['stock']['llm_call_kwargs'].update(temperature=0.7))), 'enable_thinking false'),
        # a blocked plan (the mechanism; the Gemma goal plans wait for the next runtime head)
        refused('a blocked plan', lambda: c.validate_options(fake_stage.options_for(PINS, 'obs'), variant(
            lambda p: p['config']['plans']['obs'].update(blocked='test'))), 'blocked'),
        # the 262,144-token plans (2026-10-26): ctx25 blocked by the sealed Qwen KV sizes (KV_CACHE_GIB 8 / 16 / 24 /
        # 32), their own settings held to the launcher's choices, the host budget and one Qwen plan
        refused('ctx25 on a runtime whose Qwen KV sizes lack 25 (4f5f4e7)', lambda: c.validate_options(
            fake_stage.options_for(PINS, 'ctx25'), variant(lambda p: p['runtime_features'].update(
                {c.QWEN_KV_KEY: [8, 16, 24, 32]}))), 'must be one of 8, 16, 24, 32'),
        plan_refused('ctx24 with a 71-GiB arena', lambda p: p['config']['plans']['ctx24']['server'].update(
            dram_arena_gib=71), 'ctx24', 'stay spare'),
        plan_refused('ctx24 with a 15-GiB arena', lambda p: p['config']['plans']['ctx24']['server'].update(
            dram_arena_gib=15), 'ctx24', 'one session at the 262144-token limit'),
        plan_refused('ctx24 at a context serve.py has no choice for', lambda p: p['config']['plans']['ctx24'][
            'server'].update(max_model_len=200000), 'ctx24', 'choices'),
        plan_refused('ctx24 setting K', lambda p: p['config']['plans']['ctx24']['server'].update(
            max_running_sessions=16), 'ctx24', 'server may set'),
        plan_refused('ctx24 below its arena demand', lambda p: p['config']['plans']['ctx24']['memory'].update(
            arena_demand_gib=48.0), 'ctx24', 'below the configured demand'),
        plan_refused('a Gemma plan\'s own settings beyond its keys', lambda p: p['config']['plans']['gobsdry'].update(
            server=dict(max_tokens=2048)), 'gobsdry', 'for a Gemma plan'),
        # gctx25 (2026-11-09; Gemma at 262,144 tokens, A.14): the arena that leaves less than 5 GiB, K above what the
        # goal server's pool 0 holds, 8,192-token forwards (K = 20 does not fit), the stock control at 8,192 (vLLM's
        # startup check: 27.04 GiB of 25), a forward serve.py has no choice for, K above the chat-text cap
        plan_refused('gctx25 with a 73-GiB arena', lambda p: p['config']['plans']['gctx25']['server'].update(
            dram_arena_gib=73), 'gctx25', 'stay spare'),
        plan_refused('gctx25 at K = 23 (goal_query holds at most 22)', lambda p: p['config']['plans']['gctx25'][
            'server'].update(max_running_sessions=23), 'gctx25', 'use at most 22 sessions'),
        plan_refused('gctx25 at 8,192-token forwards (the stock control refuses first)', lambda p: p['config'][
            'plans']['gctx25']['server'].pop('max_num_batched_tokens'), 'gctx25', '27.04 GiB'),
        plan_refused('gctx25 at 8,192-token forwards without its stock control', lambda p: p['config']['plans'][
            'gctx25']['servers'].pop() and p['config']['plans']['gctx25']['server'].pop('max_num_batched_tokens'),
            'gctx25', 'use at most 19 sessions'),
        plan_refused('gctx25\'s stock control at 8,192-token forwards (K = 17 fits the runtime servers)', lambda p: p[
            'config']['plans']['gctx25']['server'].update(max_running_sessions=17) or p['config']['plans']['gctx25'][
            'server'].pop('max_num_batched_tokens'), 'gctx25', '27.04 GiB'),
        plan_refused('gctx25 at a forward serve.py has no choice for', lambda p: p['config']['plans']['gctx25'][
            'server'].update(max_num_batched_tokens=2048), 'gctx25', 'max_num_batched_tokens 2048'),
        plan_refused('gctx25 above the chat-text cap', lambda p: p['config']['plans']['gctx25']['server'].update(
            max_running_sessions=25), 'gctx25', 'above the sealed chat-text cap'),
        plan_refused('the Gemma goal dry plan on a launcher without Gemma goal_query', lambda p: p['runtime_features'][
            'gemma'].update(goal_query=False), 'ggoaldry', 'refuses goal_query'),
        plan_refused('the Gemma goal plan on a launcher without Gemma goal_query', lambda p: p['runtime_features'][
            'gemma'].update(goal_query=False), 'ggoal', 'refuses goal_query'),
        refused('the goal plan with the obs phrase', lambda: c.validate_options(dict(
            fake_stage.options_for(PINS, 'goal'), authorization=c.cfg(PINS)['plans']['obs']['authorization']), PINS),
                'explicit authorization'),
        # the optimized runtime (2026-10-21)
        refused('decode_fastpath3 without decode_fastpath2', lambda: c.check_features(PINS['runtime_features'], variant(
            lambda p: p['config']['server'].update(decode_fastpath2=False))['config']), 'decode_fastpath3 requires'),
        refused('decode_fastpath3 on a runtime without it', lambda: c.check_features(
            dict(PINS['runtime_features'], decode_fastpath3=None), PINS['config']), '--decode-fastpath3'),
        refused('deferred_completion_io on a runtime without it', lambda: c.check_features(
            dict(PINS['runtime_features'], deferred_completion_io=None), PINS['config']), '--deferred-completion-io'),
        plan_refused('a Gemma server with --deferred-completion-io on a launcher without it', lambda p: p[
            'runtime_features']['gemma'].update(deferred_completion_io=False), 'gobsdry', 'launcher refuses'),
        plan_refused('a Gemma server with --decode-fastpath3 on a launcher without it', lambda p: p[
            'runtime_features']['gemma'].update(decode_fastpath3=False), 'gfull', 'launcher refuses'),
        plan_refused('the Gemma goal server at K = 18 on 25 GiB', lambda p: p['config']['gemma']['server'].update(
            max_running_sessions=18), 'ggoaldry', 'use at most 17 sessions'),
        plan_refused('the Gemma goal server at serve.py\'s default T (25,600)', lambda p: server('ggoaldry')(p).update(
            goal_rewriter_batch_tokens=25600), 'ggoaldry', 'GPU estimate'),
        plan_refused('the Gemma goal server without an explicit T', lambda p: server('ggoaldry')(p).pop(
            'goal_rewriter_batch_tokens'), 'ggoaldry', 'GPU estimate'),
        plan_refused('the Gemma goal server at 32 GiB without T', lambda p: (server('ggoaldry')(p).pop(
            'goal_rewriter_batch_tokens'), p['config']['gemma']['server'].update(kv_cache_gib=32)), 'ggoaldry',
            'needs an explicit goal_rewriter_batch_tokens'),
        plan_refused('an explicit T on a Qwen goal server', lambda p: server('goaldry')(p).update(
            goal_rewriter_batch_tokens=8192), 'goaldry', 'a Qwen server keeps'),
        # goal_query (2026-10-21)
        refused('goal plans on a runtime without chat-text goal_query', lambda: c.check_features(
            dict(PINS['runtime_features'], chat_text_goal=None), PINS['config']), 'serves no chat-text goal_query'),
        refused('Terminus 2 Goal-input pins other than the Harbor RECORD', lambda: c.check_features(
            PINS['runtime_features'], variant(lambda p: p['config']['harbor']['terminus2_goal_files'].update(
                {'harbor/agents/terminus_2/tmux_session.py': '0' * 64}))['config']), 'Harbor RECORD'),
        # job 3429670 (2026-10-22): every goal plan needs the linear-time quoted-literal pass (53e3b7a)
        refused('goal plans on a runtime without the quoted-literal fix', lambda: c.check_features(
            dict(PINS['runtime_features'], goal_quoted_literals=None), PINS['config']), 'quoted-literal pass'),
        refused('goal plans on a runtime without the batch token cap', lambda: c.check_features(
            dict(PINS['runtime_features'], goal_rewriter_batch=dict(PINS['runtime_features']['goal_rewriter_batch'],
                                                                    tokens_key=None)), PINS['config']), 'token cap'),
        plan_refused('--goal-rewriter-batch on observation_query', lambda p: server('obs')(p).update(
            goal_rewriter_batch=4), 'obs', 'needs goal_query'),
        plan_refused('--goal-rewriter-batch 17 (above the overlap)', lambda p: server('goal')(p).update(
            goal_rewriter_batch=17), 'goal', 'needs goal_query'),
        plan_refused('goal_query without the policy\'s batch', lambda p: server('goaldry')(p).pop(
            'goal_rewriter_batch'), 'goaldry', 'runs --goal-rewriter-batch'),
        plan_refused('goaldryi without an urgent reason', lambda p: p['config']['plans']['goaldryi'].pop('urgent'),
                     'goaldryi', 'urgent work only'),
        # the main ba4d1ba re-pin (2026-11-09): smokeqt (D-B) with another history mode, with --phase1-in-step (refused
        # by the strip and by --async-scheduling), on a Gemma plan, with another phrase; the async-fix guard (D-G): an
        # async plan on a runtime without the async fixes is blocked (a submission refused); the client block (D-A):
        # gctx25 on main ba4d1ba's client is blocked; D-C: a Gemma async plan on a launcher without async-gemma blocked
        plan_refused('smokeqt with the keep history', lambda p: p['config']['plans']['smokeqt'].update(
            thinking={'history': 'keep'}), 'smokeqt', 'thinking must be {"history": "strip"}'),
        plan_refused('smokeqt with --phase1-in-step', lambda p: p['config']['policies']['goal_query'].update(
            admission=['admission_overlap=16', 'phase1_in_step']), 'smokeqt', '--thinking-history strip refuses'),
        plan_refused('smokeqt on the Gemma model', lambda p: p['config']['plans']['smokeqt'].update(
            model=c.GEMMA_MODEL), 'smokeqt', 'on a Qwen plan'),
        plan_refused('smokeqt with smokeq\'s phrase', lambda p: p['config']['plans']['smokeqt'].update(
            authorization=p['config']['plans']['smokeq']['authorization']), 'smokeqt', 'phrase'),
        refused('smokeqt on a runtime without the async fixes', lambda: c.validate_options(
            fake_stage.options_for(PINS, 'smokeqt'), variant(lambda p: p['runtime_features'].update(
                {c.ASYNC_FIX_KEY: None}))), 'without the async fixes'),
        refused('smokeq on a runtime without the async fixes', lambda: c.validate_options(
            fake_stage.options_for(PINS, 'smokeq'), variant(lambda p: p['runtime_features'].update(
                {c.ASYNC_FIX_KEY: None}))), 'without the async fixes'),
        # (integrate-1112 4ebff97, 2026-11-12: the pinned client writes the Gemma job at 262,144; main ba4d1ba's 65,536
        # in a variant)
        refused('gctx25 with main ba4d1ba\'s Terminal-Bench client', lambda: c.validate_options(
            fake_stage.options_for(PINS, 'gctx25'), variant(lambda p: p['runtime_features'][c.TB_CLIENT_KEY].update(
                context=dict(p['runtime_features'][c.TB_CLIENT_KEY]['context'], **{c.GEMMA_MODEL: 65536}),
                context_choices=None))), 'writes the Gemma job at 65536 tokens'),
        refused('a Gemma async plan on a launcher without async-gemma', lambda: c.validate_options(
            fake_stage.options_for(PINS, 'smokeg'), variant(lambda p: (
                p['config']['plans']['smokeg'].update(async_scheduling=True),
                p['runtime_features']['gemma'].pop('async_scheduling', None)))), 'async-gemma'),
    ]
    argv = c.stock_argv(PINS, 8000)
    stock_kwargs = c.agent_kwargs(PINS, c.plan_servers(PINS, 'stock')[0], 8000)
    ours_kwargs = c.agent_kwargs(PINS, c.plan_servers(PINS, 'full')[0], 8000)
    print('STOCK_TEMPLATE server_flag=%s llm_call_kwargs=%s' % ('--default-chat-template-kwargs' in argv,
                                                                json.dumps(stock_kwargs.get('llm_call_kwargs'))))
    checks.append('--default-chat-template-kwargs' not in argv and 'llm_call_kwargs' not in ours_kwargs
                  and stock_kwargs['llm_call_kwargs'] == {'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
                  and {k: v for k, v in stock_kwargs.items() if k != 'llm_call_kwargs'} == ours_kwargs)
    # K = 20 with --dram-arena-gib 64 (the sealed config: the user's 20 slots with Harbor -n 16, INTERFACE.md's arena)
    memory = c.memory_fit(PINS, 'obs')
    server_ = c.cfg(PINS)['server']
    print('K=%d dram_arena_gib=%s harbor_n=%d need=%.2f GiB spare=%.2f GiB fits=%s' % (
        server_['max_running_sessions'], server_['dram_arena_gib'], c.cfg(PINS)['agent']['n_concurrent'],
        memory['need_gib'], memory['headroom_gib'], memory['fits']))
    checks.append(server_['max_running_sessions'] == 20 and memory['fits'] and memory['headroom_gib'] >= 5)
    # the runtime pin
    config = PINS['config']
    slots = lambda n: variant(lambda p: p['config']['server'].update(protocol_args=[
        '--chat-text-slots', str(n), '--chat-text-idle-timeout', '600', '--chat-text-evict-after', '30']))['config']
    checks += [
        pin_case('no chat protocol (this pin)', None, pattern='serves no chat_text'),
        pin_case('another header', chat_feature(HEADER_TEXT='<|im_start|>assistant\n'), pattern='HEADER_TEXT'),
        pin_case('another session header', chat_feature(SESSION_HEADER='x-trial'), pattern='SESSION_HEADER'),
        pin_case('without observation_query', chat_feature(served=['full', 'random_fill'], POLICIES=['full', 'random_fill']),
                 pattern='observation_query'),
        pin_case('64 slots', chat_feature(), config=slots(64), pattern='--chat-text-slots'),
        pin_case('2048 slots', chat_feature(), config=slots(2048), pattern='--chat-text-slots'),
        pin_case('module not accepted', chat_feature(), accepted=None, pattern='not adapted'),
        pin_case('module accepted', chat_feature(), accepted='a' * 64, want='PINNED'),
        pin_case('module accepted, another sha', chat_feature(), accepted='b' * 64, pattern='not adapted'),
        pin_case('K = 25 above the chat-text cap 24', chat_feature(), config=variant(
            lambda p: p['config']['server'].update(max_running_sessions=25))['config'], accepted='a' * 64,
                 pattern='chat-text cap'),
    ]
    # the Gemma plans
    def gemma(change):
        return lambda p: change(p['config']['gemma'])
    full_25 = lambda g: g.update(policy_server={})  # full without its own 32 GiB / K = 20: 25 GiB at K = 17
    checks += [
        plan_refused('Gemma full at K = 17 on 25 GiB', gemma(full_25), 'gfull', 'use at most 16 sessions'),
        plan_refused('Gemma gdry full at K = 17 on 25 GiB', gemma(full_25), 'gdry', 'use at most 16 sessions'),
        plan_refused('Gemma observation_query at K = 20 on 25 GiB', gemma(lambda g: g['server'].update(
            max_running_sessions=20)), 'gobs', 'use at most 19 sessions'),
        plan_refused('Gemma observation_query without --release-idle-pages', gemma(
            lambda g: g['server'].update(release_idle_pages=False)), 'gobs', 'the strict rule'),
        plan_refused('Gemma arena 80 GiB (the launcher\'s host rule)', lambda p: p['config']['server'].update(
            dram_arena_gib=80), 'grfill', 'more than --mem=113560M leaves'),
        plan_refused('Gemma reserve 46 GiB', lambda p: p['config']['memory'].update(process_overhead_gib=46), 'gobs',
                     'stay spare'),
        plan_refused('Gemma phrase without GEMMA4', lambda p: p['config']['plans']['gfull'].update(
            authorization=p['config']['plans']['gfull']['authorization'].replace('_GEMMA4_', '_')), 'gfull', 'GEMMA4'),
        plan_refused('Gemma goal_query server without the batch', lambda p: (unblock(p), p['config']['plans']['gobs'][
            'servers'][0].update(policy='goal_query')), 'gobs', 'runs --goal-rewriter-batch'),
        plan_refused('Gemma admission flags', lambda p: p['config']['policies']['observation_query'].update(
            admission=['admission_overlap=2']), 'gobs', 'only goal_query takes admission flags'),
        plan_refused('config.gemma.server unknown key', gemma(lambda g: g['server'].update(page_tokens=32)), 'gobs',
                     'may set'),
        plan_refused('Gemma stock without TRITON_ATTN', gemma(lambda g: g['stock'].update(attention_backend='FLASH_ATTN')),
                     'gstock', 'TRITON_ATTN'),
        plan_refused('Gemma stock with another context', gemma(lambda g: g['stock'].update(max_model_len=131072)),
                     'gstock', 'context'),
        plan_refused('Gemma stock seqs not Harbor -n', gemma(lambda g: g['stock'].update(max_num_seqs=20)), 'gstock',
                     'must equal'),
        plan_refused('Gemma minimum below build-pov-ray', lambda p: p['config']['plans']['gobs']['servers'][0].update(
            agent_min_seconds=11000), 'gobs', 'longest agent timeout'),
        refused('Gemma K = 25 above the chat-text cap', lambda: c.check_gemma_features(PINS['runtime_features'], variant(
            lambda p: p['config']['gemma']['policy_server'].update(observation_query=dict(max_running_sessions=25)))[
            'config']), 'chat-text cap'),
        refused('a launcher without Gemma chat-text', lambda: c.check_gemma_features(dict(
            PINS['runtime_features'], gemma=dict(PINS['runtime_features'].get('gemma') or {}, chat_text=False)),
            PINS['config']), 'no Gemma 4 chat-text server'),
    ]
    # the user's question of 2026-10-21: the most K the release rule allows for full at 32 GiB and 65,536
    fit32 = c.gemma_pool_fit(PINS, c.plan_servers(PINS, 'gfull')[0])
    top = c.gemma_pool_fit(PINS, c.plan_servers(PINS, 'gfull')[0], sessions=fit32['most_sessions'])
    print('GEMMA_FULL_32GIB K=%d pools=%s needed=%d usable=%d fits=%s most_sessions=%d (at K=%d: needed=%d usable=%d; '
          'the chat-text cap 24; < 2 GiB of GPU memory left by the estimates at 32 GiB)' % (
              fit32['sessions'], fit32['pools'], fit32['needed_pages'], fit32['usable_pages'], fit32['fits'],
              fit32['most_sessions'], top['sessions'], top['needed_pages'], top['usable_pages']))
    checks.append(fit32['fits'] and fit32['sessions'] == 20 and fit32['kv_cache_gib'] == 32 and top['fits'])
    for plan in ('gfull', 'gobs', 'grfill', 'gdry'):
        for s in c.plan_servers(PINS, plan):
            if s['kind'] == 'runtime':
                fit_, host = c.gemma_pool_fit(PINS, s), c.gemma_pinned_host(PINS, s)
                print('GEMMA_POOLS plan=%s server=%s K=%d kv=%d GiB pools=%s rule=%s needed=%d usable=%d spare_pages=%d '
                      'most_sessions=%d pinned_host=%.2f GiB arena_pages=%d' % (
                          plan, s['label'], fit_['sessions'], fit_['kv_cache_gib'], fit_['pools'], fit_['rule'],
                          fit_['needed_pages'], fit_['usable_pages'], fit_['usable_pages'] - fit_['needed_pages'],
                          fit_['most_sessions'], host['gib'], host['arena_pages']))
                checks.append(fit_['fits'] and host['fits'])
    gemma_feature = PINS['runtime_features'].get('gemma') or {}
    if gemma_feature.get('chat_text'):
        # the pin's Gemma rules apply to unblocked Gemma plans (the user's decision of 2026-10-21 unblocked all but the
        # goal plans); a harness whose Gemma plans are all blocked needs no Gemma turns
        unblocked = variant(unblock)['config']
        blocked_all = variant(block)['config']
        checks += [
            pin_case('Gemma: no Gemma turns', chat_feature(**{k: None for k in c.GEMMA_CHAT_PINS}), unblocked,
                     accepted='a' * 64, pattern='no Gemma turns'),
            pin_case('Gemma: another Gemma header', chat_feature(GEMMA4_HEADER_TEXT='<|turn>model\n'), unblocked,
                     accepted='a' * 64, pattern='GEMMA4_HEADER_TEXT'),
            pin_case('Gemma: every pin and the accepted module', chat_feature(), unblocked, accepted='a' * 64,
                     want='PINNED'),
            pin_case('Gemma blocked: no Gemma turns needed', chat_feature(**{k: None for k in c.GEMMA_CHAT_PINS}),
                     blocked_all, accepted='a' * 64, want='PINNED'),
        ]
        saved = c.INTERFACE_ACCEPTED
        c.INTERFACE_ACCEPTED = 'a' * 64
        try:
            features = dict(PINS['runtime_features'], chat_text=chat_feature(), gemma=dict(gemma_feature, chat_text=False))
            status, reason = c.pin_status(features, unblocked)
        finally:
            c.INTERFACE_ACCEPTED = saved
        print('PIN Gemma: a launcher without Gemma chat-text %s %s' % (status, reason))
        checks.append(status == 'PLACEHOLDER' and 'no Gemma chat-text' in str(reason))
    # the policy registry (2026-11-12): a Terminus 2 Goal-input module the harness has not reviewed keeps the pin a
    # PLACEHOLDER; a plan on a registered policy this harness's checks do not cover (recent_fill) is refused
    saved = c.INTERFACE_ACCEPTED
    c.INTERFACE_ACCEPTED = 'a' * 64
    try:
        features = dict(PINS['runtime_features'], chat_text=chat_feature(), session_limits=dict(
            PINS['runtime_features'].get('session_limits') or {}, max_chat_text_running=24),
            chat_text_goal=dict(PINS['runtime_features']['chat_text_goal'], module_sha256='f' * 64))
        status, reason = c.pin_status(features, PINS['config'])
    finally:
        c.INTERFACE_ACCEPTED = saved
    print('PIN an unreviewed Terminus 2 Goal-input module %s %s' % (status, str(reason)[:160]))
    checks.append(status == 'PLACEHOLDER' and 'not reviewed' in str(reason))

    def window_policy(p):
        p['config']['policies']['recent_fill'] = copy.deepcopy(p['config']['policies']['observation_query'])
        p['config']['plans']['obs']['servers'][0]['policy'] = 'recent_fill'
    checks.append(plan_refused('a plan on recent_fill (no Terminal-Bench checks)', window_policy, 'obs',
                               'this harness runs'))
    # the Harbor spec argument (89 tasks) within the bound
    spec, _ = c.harbor_spec(PINS, c.plan_servers(PINS, 'full')[0], 24000, 2e9, 'tb16-1234567-r1', 'full')
    print('HARBOR_SPEC bytes=%d bound=%d' % (len(spec), c.SPEC_MAX_BYTES))
    checks.append(len(spec) <= c.SPEC_MAX_BYTES)
    with tempfile.TemporaryDirectory() as tmp:
        plan = c.cfg(PINS)['plans']['obs']
        for i, other in enumerate(('tb16-full', 'swe32-full', 'swe10-goal', 'swe16-stock', 'gem0-smoke')):
            checks.append(refused('queue gate: %s queued' % other, lambda other=other, i=i: c.queue_gate(
                Path(tmp), PINS, plan, dict(plan='obs', after_job=None), 'q%d.json' % i,
                runner=lambda *a, **k: Run('777|PENDING|ghx4|%s\n' % other), user='ycao11'), 'queue holds'))
        checks.append(refused('queue gate: squeue failing', lambda: c.queue_gate(
            Path(tmp), PINS, plan, dict(plan='obs', after_job=None), 'qf.json',
            runner=lambda *a, **k: Run('', 1), user='ycao11'), 'squeue failed'))
        value = c.queue_gate(Path(tmp), PINS, plan, dict(plan='obs', after_job='777'), 'qa.json',
                             runner=lambda *a, **k: Run('777|RUNNING|ghx4|tb16-full\n888|PENDING|ghx4|other\n'),
                             user='ycao11')
        print('QUEUE_GATE named -AfterJob 777: %s' % value['status'])
        checks.append(value['status'] == 'QUEUE_GATE_PASS')
        # dryi (ghx4-interactive) and the user's holds: squeue per partition (id|state|partition|reason|name)
        dryi = c.cfg(PINS)['plans']['dryi']

        def queue(**by_partition):
            def runner(argv, **kwargs):
                part = argv[argv.index('-p') + 1]
                return Run(by_partition.get(part.replace('-', '_'), ''))
            return runner

        def gate(plan_name, runner, after=None, tag='g'):
            return c.queue_gate(Path(tmp), PINS, c.cfg(PINS)['plans'][plan_name], dict(plan=plan_name, after_job=after),
                                'q-%s.json' % tag, runner=runner, user='ycao11')
        value = gate('dryi', queue(), tag='empty')
        print('QUEUE_GATE dryi, empty queue: %s' % value['status'])
        checks.append(value['status'] == 'QUEUE_GATE_PASS' and [x['argv'][5] for x in value['calls']]
                      == ['ghx4', 'ghx4-interactive'])
        held = '3400001|PENDING|ghx4|JobHeldUser|swe10-goal\n'
        value = gate('dryi', queue(ghx4=held), tag='held')
        print('QUEUE_GATE dryi, swe10 held by the user: %s held=%s release=%s' % (value['status'], value['held_job_ids'],
                                                                             value['release_after_tb']))
        checks.append(value['status'] == 'QUEUE_GATE_PASS' and value['held_job_ids'] == ['3400001']
                      and value['release_after_tb'] == 'scontrol release 3400001' and value['held'] == [held.strip()])
        value = gate('obs', queue(ghx4=held), tag='held-batch')
        checks.append(value['status'] == 'QUEUE_GATE_PASS' and value['held_job_ids'] == ['3400001'])
        for tag, line, pattern in (
                ('admin-hold', '3400001|PENDING|ghx4|JobHeldAdmin|swe10-goal', 'queue holds'),
                ('running', '3400001|RUNNING|ghx4|None|swe10-goal', 'queue holds'),
                ('pending', '3400002|PENDING|ghx4|Priority|swe32-goal', 'queue holds'),
                ('tb16-pending', '3400003|PENDING|ghx4|Resources|tb16-full', 'queue holds')):
            checks.append(refused('queue gate dryi: %s' % tag, lambda line=line, tag=tag: gate(
                'dryi', queue(ghx4=line + '\n'), tag=tag), pattern))
        value = gate('dryi', queue(ghx4='3400002|PENDING|ghx4|Priority|swe32-goal\n'), after='3400002', tag='after')
        print('QUEUE_GATE dryi, -AfterJob names the pending swe32 job: %s' % value['status'])
        checks.append(value['status'] == 'QUEUE_GATE_PASS')
        for tag, line in (('interactive-shell', '3400010|RUNNING|ghx4-interactive|None|bash'),
                          ('interactive-held', '3400011|PENDING|ghx4-interactive|JobHeldUser|swe10-goal')):
            checks.append(refused('queue gate dryi: %s' % tag, lambda line=line, tag=tag: gate(
                'dryi', queue(ghx4_interactive=line + '\n'), tag=tag), 'allows 1 submitted job'))
        value = gate('obs', queue(ghx4_interactive='3400010|RUNNING|ghx4-interactive|None|bash\n'), tag='batch-shell')
        checks.append(value['status'] == 'QUEUE_GATE_PASS')  # a batch plan ignores the interactive limit
        checks.append(refused('queue gate dryi: squeue failing on ghx4-interactive', lambda: gate(
            'dryi', lambda argv, **k: Run('', 1) if 'ghx4-interactive' in argv else Run(''), tag='fail'),
            'squeue failed'))
        assert dryi['partition'] == 'ghx4-interactive'
    checks += light_gates()
    bad = sum(1 for x in checks if not x)
    ok &= bad == 0
    print('%s plans=%d refusals=%d bad=%d' % ('GATE_CHECKS_PASS' if ok else 'GATE_CHECKS_FAIL', len(PLANS), len(checks),
                                              bad))
    return 0 if ok else 1


def light_gates():
    """The light login gates: the receipt refusals and the login step's contents (read from the sealed source)."""
    checks = []
    value = c.release_receipt_gate(PINS)
    print('RELEASE_RECEIPT %s commit=%s tests=%s skipped=%s login_test_runs=%s' % (
        value['status'], value['commit'], value['tests_run'], value['skipped'], value['login_test_runs']))
    checks.append(value['status'] == 'RELEASE_RECEIPT_PINNED' and value['commit'] == PINS['commit']
                  and value['login_test_runs'] == 0 and PINS['runtime_pin'].get('release_receipt') is True)
    for name, change, pattern in (
            ('no receipt', lambda p: p.pop('release_receipt'), 'no local release-check receipt'),
            ('a receipt of another commit', lambda p: p['release_receipt'].update(commit='0' * 40), 'not the sealed commit'),
            ('another commit sealed', lambda p: p.update(commit='1' * 40), 'not the sealed commit'),
            ('a failed release check', lambda p: p['release_receipt'].update(status='GIT_ARCHIVE_TESTS_FAIL'),
             'GIT_ARCHIVE_TESTS_PASS'),
            ('no tests', lambda p: p['release_receipt'].update(tests_run=0), 'tests'),
            ('a malformed result hash', lambda p: p['release_receipt'].update(result_line_sha256='x'), 'SHA-256'),
            ('no log hash', lambda p: p['release_receipt'].pop('log_sha256'), 'SHA-256')):
        checks.append(refused('release receipt: %s' % name, lambda change=change: c.release_receipt_gate(
            variant(change)), pattern))
    setup, qualify = inspect.getsource(c.setup), inspect.getsource(c.qualify)
    dropped = [w for w in ("'unittest'", 'check_git_release', 'public-tests', 'ipc_gate', 'sampler_gate',
                           'sampling_gate') if w in qualify or w in setup]
    kept = [w for w in ('environment_gate(', 'reused_official_record(', "'S1.json'", 'compiler_gate(',
                        'serve_dry_run_gate(', 'stock_gate(', 'tbench_gate(', 'queue_gate(', 'login-timing.json')
            if w not in qualify]
    kept += [w for w in ('fresh_bundle_clone(', 'release_receipt_gate(', 'package-build') if w not in setup]
    clone = inspect.getsource(c.fresh_bundle_clone)
    kept += [w for w in ("'bundle-verify'", "'git-fsck'", "'--strict'") if w not in clone]
    receipt_before_claim = setup.index('release_receipt_gate(') < setup.index('claim_state(')
    print('LIGHT_LOGIN_GATES dropped_left=%s kept_missing=%s receipt_before_claim=%s' % (
        dropped, kept, receipt_before_claim))
    checks.append(not dropped and not kept and receipt_before_claim)
    return checks


if __name__ == '__main__':
    sys.exit(main())
