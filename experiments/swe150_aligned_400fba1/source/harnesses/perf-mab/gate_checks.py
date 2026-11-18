"""Local only (never delivered): the sealed plans' gates, printed for the validation log.

For every plan: the time fit (controller.validate_plan: phrase, ghx4 batch, servers, the workload plan, the time
within 1.2-1.35 x the expected wall time, the required servers' guaranteed minimums), every server's expected,
minimum and cap seconds, and the run-time timeline of the expected case (job.Budget's rules with every server taking
its expected time: which optional servers start). Then the refusals the gate must make: a time above 1.35 x expected
(a long limit hurts backfill), one below 1.2 x expected, a phrase with INTERACTIVE or the wrong minutes, the
ghx4-interactive partition, a required minimum that no longer fits, the probes on a pooled server, a G1c that would
replay a non-eager or non-full server, the tokenizer override outside synthetic stages, a release-check receipt of
another commit, failed, missing or malformed (light login gates), a sealed commit without its receipt, a cloned HEAD
other than the receipt's; and the queue gate not applied on ghx4 (no squeue, no record). integrate-1021: the Gemma
end-to-end plans' pools, switches and token cap, their refusals (16 sessions at 25 GiB, the decided KV, the batch
without a cap or above the overlap, fast path 3 without 2, the interactive variants' time / phrase, interactive_of on a
batch plan), the queue gate applied on ghx4-interactive (an empty queue passes, a job there refuses) and the billed
GPU-hours. Exit 0 and GATE_CHECKS_PASS iff every plan fits and every refusal happens.
"""
import copy
import json
import re
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402
import job  # noqa: E402
import mab_workload as w  # noqa: E402

PINS = c.load(ROOT / 'pins.json')


def variant(change):
    pins = copy.deepcopy(PINS)
    change(pins['config'])
    pins['workload_plan'] = w.build_plan(pins['config'], pins.get('real_data'))
    return pins


def refused(name, pins, plan, pattern=''):
    try:
        c.validate_plan(pins, plan)
    except RuntimeError as exc:
        ok = pattern in str(exc)
        print('REFUSED %s %s: %s' % (name, 'as expected' if ok else 'BUT WITH ANOTHER REASON', str(exc)[:300]))
        return ok
    print('NOT_REFUSED %s' % name)
    return False


def timeline(name):
    """The expected run: every server takes its expected time; the start rule of job.run_servers."""
    rows = c.plan_servers(PINS, name)
    b = c.budget_of(PINS, c.plan_model(PINS, name))
    t = c.fixed_seconds(PINS, name) - b['expected']['finish_seconds']
    deadline = c.time_seconds(c.cfg(PINS)['plans'][name]['time']) - b['finish_seconds'] - b['margin_seconds']
    out = []
    for index, row in enumerate(rows):
        server = c.server_entry(PINS, row['label'])
        reserve = job.later_reserve(PINS, rows, index)
        need, basis = job.start_need(PINS, server, row)
        starts = deadline - t >= need + reserve
        if starts:
            t += row['expected_seconds']
        out.append((row['label'], row['role'], row['expected_seconds'], row['minimum_seconds'], need, reserve, starts,
                    int(t)))
    return out, int(t + b['expected']['finish_seconds']), deadline


def main():
    ok = True
    print('RUNTIME commit=%s kv_session=%s release_idle_pages=%s decode_fastpath=%s plan_sha256=%s real_data=%s' % (
        PINS['commit'], bool(PINS['runtime_features'].get('kv_session')),
        bool(PINS['runtime_features'].get('release_idle_pages')), bool(PINS['runtime_features'].get('decode_fastpath')),
        PINS['workload_plan']['sha256'], (PINS.get('real_data') or {}).get('manifest_sha256')))
    gemma = PINS['runtime_features'].get('gemma4') or {}
    print('SWITCHES decode_fastpath2=%s decode_fastpath3=%s deferred_completion_io=%s goal_rewriter_batch=%s '
          'gemma_switches=%s gemma_batch_tokens=%s gemma_kv=%s' % (
              bool(PINS['runtime_features'].get('decode_fastpath2')), bool(PINS['runtime_features'].get('decode_fastpath3')),
              bool(PINS['runtime_features'].get('deferred_completion_io')),
              bool(PINS['runtime_features'].get('goal_rewriter_batch')), ','.join(gemma.get('switches') or ()),
              gemma.get('goal_rewriter_batch_tokens'), gemma.get('kv_cache_gib_choices')))
    for name in PINS['config']['plan_order']:
        fit = c.validate_plan(PINS, name)
        plan = PINS['config']['plans'][name]
        print('PLAN_FIT %s partition=%s time=%s (%d s) expected=%d factor=%.3f range=%s required_minimum=%d spare=%d '
              'required=%s optional=%s' % (name, plan['partition'], plan['time'], fit['time_seconds'],
                                           fit['expected_seconds'], fit['factor'], fit['factor_range'],
                                           fit['required_minimum_seconds'], fit['required_spare_seconds'],
                                           fit['required'], fit['optional']))
        for row in fit['servers']:
            server = c.server_entry(PINS, row['label'])
            start, agents, probes, stop = c.server_parts(PINS, server)
            print('  SERVER %-4s %-8s expected=%5d (start %d + agents %d + probes %d + stop %d) minimum=%5d cap=%5d '
                  'drain=%d' % (row['label'], row['role'], row['expected_seconds'], start, agents, probes, stop,
                                row['minimum_seconds'], row['cap_seconds'], c.drain_seconds(PINS, server)))
        rows, total, deadline = timeline(name)
        for label, role, expected, minimum, need, reserve, starts, end in rows:
            print('  EXPECTED_RUN %-4s %-8s start_need=%5d later_reserve=%5d starts=%s ends_at=%d' % (
                label, role, need, reserve, starts, end))
        print('  EXPECTED_TOTAL %s: %d s of %d s (deadline for server work %d s)' % (name, total, fit['time_seconds'],
                                                                                  deadline))
        ok &= all(starts for *_, starts, _ in rows)

    for name in PINS['config']['plan_order']:
        m = c.memory_fit(PINS, name)
        print('MEMORY_FIT %s --mem=%s need %d MiB (worst pinned host %.2f GiB + %d GiB overhead + %d GiB margin), '
              'spare %d MiB, max %dM' % (name, m['mem'], m['need_mib'], m['pinned_worst_gib'], m['overhead_gib'],
                                          m['margin_gib'], m['spare_mib'], m['max_mib']))
        ok &= m['fits']
    for name in PINS['config']['plan_order']:
        risky = c.startup_spike_risk(PINS, name)
        print('STARTUP_SPIKE %s %s' % (name, c.memory_fit(PINS, name)['startup_spike']))
        ok &= not risky
    for server in PINS['config']['servers']:
        a = c.arena_fit(PINS, server) if server['kind'] == 'runtime' else None
        if a:
            print('ARENA_FIT %s %d sessions at their largest cases, replies %d x: %.2f GiB of the %d-GiB arena'
                  % (server['label'], a['sessions'], a['reply_factor'], a['peak_gib'], a['arena_gib']))
            ok &= a['fits']
    # Gemma 4: the launcher's pool-0 rule, the pinned host of N shares, the time model's concurrency and re-prefill.
    for server in PINS['config']['servers']:
        if not c.is_gemma(server):
            continue
        if server['kind'] == 'runtime':
            s = c.runtime_settings(PINS['config'], server)
            fit, host = c.gemma_pool_fit(s), c.gemma_pinned_host(s)
            print('GEMMA_POOLS %s kv=%d K=%d rule=%s pool1=%d blocks (%.2f GiB) pool0=%d Pages usable=%d needed=%d '
                  'request_pages=%s fits=%s | pinned %.2f GiB of %.1f (%d share%s) fits=%s | decode_concurrency %.2f '
                  'drain %d s | switches fastpath2=%s fastpath3=%s deferred_io=%s batch=%s tokens=%s'
                  % (server['label'], s['kv_cache_gib'], s['max_running_sessions'], fit['rule'], fit['pool_one_blocks'],
                     fit['pool_one_gib'], fit['blocks'], fit['usable'], fit['needed'], fit['request_pages'], fit['fits'],
                     host['gib'], host['budget_gib'], host['shares'], '' if host['shares'] == 1 else 's', host['fits'],
                     c.decode_concurrency(PINS, server), c.drain_seconds(PINS, server), s[c.FASTPATH2_KEY],
                     s[c.FASTPATH3_KEY], s[c.DC_KEY], s.get(c.BATCH_KEY), c.batch_tokens_of(PINS, server)))
            ok &= fit['fits'] and host['fits']
        elif server.get('reprefill'):
            print('GEMMA_STOCK %s re-prefill per turn %.2f s (stage 0 TTFT curve) decode_concurrency %.0f drain %d s'
                  % (server['label'], c.gemma_reprefill_seconds(PINS, server), c.decode_concurrency(PINS, server),
                     c.drain_seconds(PINS, server)))

    def mem_small(config):
        config['plans']['g2']['mem'] = '70G'
    ok &= refused('g2-mem-below-the-pinned-host-need', variant(mem_small), 'g2', 'worst pinned host')

    def mem_large(config):
        config['plans']['g1']['mem'] = '113561M'
    ok &= refused('g1-mem-above-one-GH200-share', variant(mem_large), 'g1', 'worst pinned host')

    def arena_large(config):
        for server in config['servers']:
            if server['label'] == 'g2a':
                server['dram_arena_gib'] = 88
    ok &= refused('g2-arena-88-above-the-share', variant(arena_large), 'g2', 'worst pinned host')

    def chunks_long(config):
        config['workloads']['g2']['chunks'] = [12, 13, 14]
    ok &= refused('g2-12-14-chunks-overflow-the-arena', variant(chunks_long), 'g2', 'DRAM_ARENA_EXHAUSTED')

    def g4_small_arena(config):
        for server in config['servers']:
            if server['label'] == 'g4o':
                server.update(dram_arena_gib=100)
    ok &= refused('g4o-arena-100-below-16-sessions', variant(g4_small_arena), 'g4o', 'DRAM_ARENA_EXHAUSTED')

    def g4_one_share(config):
        config['plans']['g4o']['mem'] = '113560M'
    ok &= refused('g4o-on-one-memory-share', variant(g4_one_share), 'g4o', 'worst pinned host')

    def g1_two_shares(config):
        config['plans']['g1']['mem'] = '227120M'
    ok &= refused('g1-two-memory-shares', variant(g1_two_shares), 'g1', 'worst pinned host')

    def too_long(config):
        config['plans']['g1'].update(time='01:40:00', authorization=config['plans']['g1']['authorization']
                                     .replace('_70MIN_', '_100MIN_'))
    ok &= refused('g1-time-above-1.35x-expected', variant(too_long), 'g1', 'x the expected')

    def too_short(config):
        config['plans']['g1'].update(time='00:55:00', authorization=config['plans']['g1']['authorization']
                                     .replace('_70MIN_', '_55MIN_'))
    ok &= refused('g1-time-below-1.2x-expected', variant(too_short), 'g1', 'x the expected')

    def interactive(config):
        config['plans']['g2']['authorization'] = config['plans']['g2']['authorization'].replace('_BATCH_',
                                                                                                '_INTERACTIVE_')
    ok &= refused('g2-phrase-with-INTERACTIVE', variant(interactive), 'g2', 'phrase')

    def minutes(config):
        config['plans']['g2']['authorization'] = re.sub(r'_[0-9]+MIN_', '_120MIN_', config['plans']['g2']['authorization'])
    ok &= refused('g2-phrase-minutes', variant(minutes), 'g2', 'phrase')

    def partition(config):
        config['plans']['g2']['partition'] = 'ghx4-interactive'
    ok &= refused('g2-on-ghx4-interactive', variant(partition), 'g2', 'ghx4 batch partition')

    def minimum(config):
        for label in ('g2a', 'g2b'):
            next(s for s in config['servers'] if s['label'] == label)['min_agent_seconds'] = 2400
    ok &= refused('g2-required-minimum-does-not-fit', variant(minimum), 'g2', 'required minimum')

    def probes(config):
        next(s for s in config['servers'] if s['label'] == 'g2b')['probes'] = True
    ok &= refused('probes-on-a-pooled-server', variant(probes), 'g2', 'probes need one running session')

    def replay(config):
        next(s for s in config['servers'] if s['label'] == 'g1c')['replay_source'] = 'g1b'
    ok &= refused('g1c-replays-observation-query', variant(replay), 'g1', 'replays an earlier eager full')

    def override(config):
        config['runner']['tokenizer_override'] = dict(kind='char', config='x')
    ok &= refused('tokenizer-override-outside-synthetic', variant(override), 'g1', 'tokenizer override')

    def gm4o_two_shares(config):
        config['plans']['gm4o']['mem'] = '227120M'
    ok &= refused('gm4o-two-memory-shares-not-a-two-share-plan', variant(gm4o_two_shares), 'gm4o', 'worst pinned host')

    def gm4o_launcher_one_share(config):
        next(s for s in config['servers'] if s['label'] == 'gm4o')['dram_arena_gib'] = 100
    ok &= refused('gm4o-arena-100-above-one-share-launcher-budget', variant(gm4o_launcher_one_share), 'gm4o',
                  'Gemma launcher')

    def gm1_262144(config):
        next(s for s in config['servers'] if s['label'] == 'gm1a')['max_model_len'] = 262144
    ok &= refused('gm1a-at-262144-strict-rule', variant(gm1_262144), 'gm1', 'one turn at the context limit')

    def gm4o_arena(config):
        next(s for s in config['servers'] if s['label'] == 'gm4o')['dram_arena_gib'] = 59
    ok &= refused('gm4o-arena-59-below-8-gemma-sessions', variant(gm4o_arena), 'gm4o', 'DRAM_ARENA_EXHAUSTED')

    # integrate-1021 (the end-to-end dry runs): 25 GiB, the optimized switches, the rewriter batch, interactive.
    def gm4o_sixteen(config):
        next(s for s in config['servers'] if s['label'] == 'gm4o').update(max_running_sessions=16, workers=16)
    ok &= refused('gm4o-16-sessions-at-25-gib-32k-budget', variant(gm4o_sixteen), 'gm4o', 'usable blocks')

    def gm4g_kv28(config):
        next(s for s in config['servers'] if s['label'] == 'gm4g')['kv_cache_gib'] = 28
    ok &= refused('gm4g-kv-28-not-the-decided-25', variant(gm4g_kv28), 'gm4g', 'decided KV')

    def gm4g_kv32(config):
        next(s for s in config['servers'] if s['label'] == 'gm4g')['kv_cache_gib'] = 32
    ok &= refused('gm4g-batch-at-32-gib-without-a-token-cap', variant(gm4g_kv32), 'gm4g', 'no default token cap')

    def gm4g_batch8(config):
        next(s for s in config['servers'] if s['label'] == 'gm4g')['goal_rewriter_batch'] = 16
    ok &= refused('gm4g-batch-above-the-admission-overlap', variant(gm4g_batch8), 'gm4g', 'admission_overlap K >= N')

    def gm4r_fastpath3(config):
        next(s for s in config['servers'] if s['label'] == 'gm4r')['decode_fastpath2'] = False
    ok &= refused('gm4r-fastpath3-without-fastpath2', variant(gm4r_fastpath3), 'gm4r', 'requires decode_fastpath2')

    def gm4oi_time(config):
        config['plans']['gm4oi'].update(time='00:50:00', authorization=config['plans']['gm4oi']['authorization']
                                        .replace('_49MIN_', '_50MIN_'))
    ok &= refused('gm4oi-time-differs-from-its-batch-plan', variant(gm4oi_time), 'gm4oi', 'ghx4 batch partition')

    def gm4gi_phrase(config):
        config['plans']['gm4gi']['authorization'] = config['plans']['gm4gi']['authorization'].replace('_INTERACTIVE_',
                                                                                                    '_BATCH_')
    ok &= refused('gm4gi-phrase-with-BATCH', variant(gm4gi_phrase), 'gm4gi', 'phrase')

    def gm4o_batch_plan_named_interactive(config):
        config['plans']['gm4o']['interactive_of'] = 'gm4r'
    ok &= refused('interactive_of-on-a-batch-plan', variant(gm4o_batch_plan_named_interactive), 'gm4o',
                  'ghx4 batch partition')

    # integrate-1103 (the release smoke plans smq / smg): one share, the arena, the role's 2.25 x, the plan's async
    # switch (Qwen only), the vLLM Goal rewriter's refusals, Gemma at 16 sessions.
    def smq_one_share(config):
        config['plans']['smq']['mem'] = '113560M'
    ok &= refused('smq-16-sessions-on-one-memory-share', variant(smq_one_share), 'smq', 'worst pinned host')

    def smq_arena70(config):
        for label in ('smqo', 'smqr', 'smqg'):
            next(s for s in config['servers'] if s['label'] == label)['dram_arena_gib'] = 70
        config['plans']['smq']['mem'] = '113560M'
    ok &= refused('smq-arena-70-below-16-smoke-cases', variant(smq_arena70), 'smq', 'DRAM_ARENA_EXHAUSTED')

    def smg_above_role(config):
        config['plans']['smg'].update(time='03:40:00', authorization=config['plans']['smg']['authorization']
                                      .replace('_120MIN_', '_220MIN_'))
    ok &= refused('smg-time-above-2.25x-expected', variant(smg_above_role), 'smg', 'x the expected')

    def g1_role_release(config):   # the role is the smoke plans' (the 1.35 x rule of every other plan stays)
        config['plans']['g1'].update(time='01:40:00', authorization=config['plans']['g1']['authorization']
                                     .replace('_70MIN_', '_100MIN_'))
    ok &= refused('g1-time-above-1.35x-without-the-role', variant(g1_role_release), 'g1', 'x the expected')

    def smg_async(config):
        config['plans']['smg']['async_scheduling'] = True
    ok &= refused('smg-async-scheduling-on-gemma', variant(smg_async), 'smg', 'Qwen profile only')

    def smqg_batch(config):
        next(s for s in config['servers'] if s['label'] == 'smqg')['goal_rewriter_batch'] = 4
    ok &= refused('smqg-vllm-rewriter-with-the-batch', variant(smqg_batch), 'smq', 'excludes --goal-rewriter-batch')

    def smgg_no_overlap(config):
        next(s for s in config['servers'] if s['label'] == 'smgg').pop('admission_overlap')
    ok &= refused('smgg-vllm-rewriter-without-admission-overlap', variant(smgg_no_overlap), 'smg',
                  'needs --admission-overlap K >= 2')

    def smgo_sixteen(config):
        next(s for s in config['servers'] if s['label'] == 'smgo').update(max_running_sessions=16, workers=16)
    ok &= refused('smgo-16-sessions-at-25-gib-32k-budget', variant(smgo_sixteen), 'smg', 'usable blocks')

    def gemma_thinking(config):
        config['generation']['enable_thinking'] = True
    ok &= refused('gm1-thinking-on', variant(gemma_thinking), 'gm1', 'thinking off only')

    def gemma_override(config):
        config['gemma']['runner']['tokenizer_override'] = dict(kind='char', config='x')
    ok &= refused('gemma-tokenizer-override-outside-synthetic', variant(gemma_override), 'gm1', 'tokenizer override')

    def gm4v_long(config):
        config['plans']['gm4v'].update(time='08:00:00', authorization=config['plans']['gm4v']['authorization']
                                       .replace('_345MIN_', '_480MIN_'))
    ok &= refused('gm4v-time-above-1.35x-expected', variant(gm4v_long), 'gm4v', 'x the expected')

    # main ba4d1ba re-pin (2026-11-09): D-G the async fixes, D-B per-plan thinking, D-C Gemma async, D-H the GPU fit.
    for name, fit in ((n, c.gpu_fit(PINS, n)) for n in PINS['config']['plan_order']):
        for row in fit['servers']:
            if row['status'] == 'NOT_MODELLED':
                continue
            print('GPU_FIT %s %-6s %s peak %d MiB (anchor %s %s %d + vLLM child %d + async %d) free %d of %d (min %d)'
                  % (name, row['label'], row['status'], row['peak_mib'], row['anchor']['job'], row['anchor']['server'],
                     row['anchor']['peak_mib'], row['vllm_child_mib'], row['async_mib'], row['free_mib'],
                     row['total_mib'], row['min_free_mib']))
            ok &= row['status'] == 'LEAVES_MIN_FREE'

    def pins_variant(change):
        pins = copy.deepcopy(PINS)
        change(pins)
        return pins

    def no_async_fix(pins):
        pins['runtime_features'][c.ASYNC_FIX_KEY] = None
    for plan in ('smq', 'smqt', 'smga'):
        ok &= refused('%s-async-without-the-async-fixes' % plan, pins_variant(no_async_fix), plan, 'NOT_SUBMITTABLE')

    def qwen_only_async(pins):
        pins['runtime_features'][c.ASYNC_KEY]['profiles'] = [c.QWEN_MODEL]
    ok &= refused('smga-on-a-launcher-without-async-gemma', pins_variant(qwen_only_async), 'smga', 'Qwen profile only')

    def smqt_shares_smqo(config):
        config['plans']['smqt']['servers'] = ['smqo', 'smqrt', 'smqgt']
    ok &= refused('smqt-thinking-on-a-server-of-smq', variant(smqt_shares_smqo), 'smqt', 'disagree on kv_session_thinking')

    def smga_thinking(config):
        config['plans']['smga']['kv_session_thinking'] = 'on'
    ok &= refused('smga-kv-session-thinking-on-gemma', variant(smga_thinking), 'smga', 'thinking')

    def g1_thinking(config):
        config['plans']['g1']['kv_session_thinking'] = 'on'
    ok &= refused('g1-plan-thinking-with-stock-servers', variant(g1_thinking), 'g1', 'runtime servers only')

    def smgo_marked(config):
        next(s for s in config['servers'] if s['label'] == 'smgo')['async_gemma'] = True
    ok &= refused('smgo-async-gemma-outside-an-async-plan', variant(smgo_marked), 'smg', 'async_gemma')

    def child_tight(config):
        config['gpu']['vllm_child']['over_hf_mib'] = 5500
    ok &= refused('smqt-vllm-child-below-2-gib-free', variant(child_tight), 'smqt', '--goal-rewriter-vllm-kv-gib 2')

    # integrate-1112: the policies from the sealed registry (runtime_features.policy_registry): none without it, a
    # registered policy without a budget the checks judge (random50 'half', akv_rule 'rule') refused, the Goal switches
    # only on the Goal-engine policy.
    def no_registry(pins):
        pins['runtime_features'].pop(c.REGISTRY_KEY)
    ok &= refused('smq-without-the-policy-registry', pins_variant(no_registry), 'smq', 'no page-selection policy registry')
    for policy in ('random50', 'akv_rule'):
        def other_policy(config, policy=policy):
            next(s for s in config['servers'] if s['label'] == 'smqo')['policy'] = policy
        ok &= refused('smqo-%s-not-judged' % policy, variant(other_policy), 'smq', "smqo: policy '%s'" % policy)

    def overlap_on_obs(config):
        next(s for s in config['servers'] if s['label'] == 'smqo')[c.GOAL_OVERLAP_KEY] = True
    ok &= refused('smqo-goal-overlap-on-observation-query', variant(overlap_on_obs), 'smq',
                  '--goal-overlap needs a Goal-engine policy')

    # plan smqc (2026-11-15): the async-maxtok case cell. It needs --async-scheduling (no async step otherwise) and a
    # runtime with 13ee618's resolve; its counts and time are refused out of range.
    print('CUT_CELL smqc smqco estimate %d s, expected %d, max %d; plan %s'
          % (c.cut_cell_estimate(PINS, c.server_entry(PINS, 'smqco')),
             c.server_entry(PINS, 'smqco')[c.CUT_CELL_KEY]['expected_seconds'],
             c.server_entry(PINS, 'smqco')[c.CUT_CELL_KEY]['max_seconds'], json.dumps(c.plan_fit(PINS, 'smqc'))[:200]))

    def smqc_sync(config):
        config['plans']['smqc'].pop('async_scheduling')
    ok &= refused('smqc-cut-cell-without-async-scheduling', variant(smqc_sync), 'smqc', 'needs --async-scheduling')

    def no_resolve_fix(pins):
        pins['runtime_features'][c.ASYNC_FIX_KEY] = dict(pins['runtime_features'][c.ASYNC_FIX_KEY], resolve=None)
    ok &= refused('smqc-without-the-resolve-of-13ee618', pins_variant(no_resolve_fix), 'smqc',
                  'resolve of async-maxtok-1112')
    for name, change, pattern in (('min-hits-above-the-probe-turns', dict(min_hits=17), 'min_hits'),
                                  ('probe-bound-near-the-carrier', dict(carrier_max_tokens=400), 'carrier bound'),
                                  ('lead-below-the-admission', dict(lead_seconds=2.0), 'lead_seconds'),
                                  ('expected-below-the-estimate', dict(expected_seconds=60), 'estimate'),
                                  ('nine-probe-sessions', dict(probe_bounds=list(range(10, 100, 10))), 'probe_bounds')):
        def cell(config, change=change):
            next(s for s in config['servers'] if s['label'] == 'smqco')[c.CUT_CELL_KEY].update(change)
        ok &= refused('smqc-%s' % name, variant(cell), 'smqc', pattern)

    def smqc_long(config):
        config['plans']['smqc']['time'] = '02:00:00'
        config['plans']['smqc']['authorization'] = config['plans']['smqc']['authorization'].replace('_30MIN_',
                                                                                                   '_120MIN_')
    ok &= refused('smqc-time-above-2.25x-expected', variant(smqc_long), 'smqc', 'x the expected')

    # Light login gates: the pinned local release-check receipt of the sealed commit (no runtime suite on the login
    # node); every plan refuses another commit's receipt, a failed one, a missing one, a malformed digest.
    receipt = c.check_release_receipt(PINS)['receipt']
    print('RELEASE_RECEIPT commit=%s status=%s tests_run=%d skipped=%d result_line_sha256=%s log_sha256=%s' % (
        receipt['commit'], receipt['status'], receipt['tests_run'], receipt['skipped'], receipt['result_line_sha256'],
        receipt['log_sha256']))

    def receipt_variant(change):
        pins = copy.deepcopy(PINS)
        change(pins)
        return pins
    for name, change, pattern in (
            ('receipt-of-another-commit',   # the previous pin's commit (gemma-mab 9af4859)
             lambda p: p['release_receipt'].update(commit='9af4859e7dda2696f9fe40bff2fbacda6c73f677'), 'not the sealed'),
            ('receipt-failed', lambda p: p['release_receipt'].update(status='GIT_ARCHIVE_TESTS_FAIL'), 'did not pass'),
            ('receipt-missing', lambda p: p.pop('release_receipt'), 'no pinned release-check receipt'),
            ('receipt-malformed-digest', lambda p: p['release_receipt'].update(log_sha256='x' * 64), 'SHA-256'),
            ('sealed-commit-without-its-receipt', lambda p: p.update(commit='0' * 40), 'not the sealed')):
        for plan in ('g4o', 'g4g', 'gm4o', 'gm4g'):
            ok &= refused('%s-%s' % (plan, name), receipt_variant(change), plan, pattern)
    try:
        c.check_release_receipt(PINS, head='0' * 40)
        print('NOT_REFUSED cloned-head-differs-from-the-receipt')
        ok = False
    except RuntimeError as exc:
        print('REFUSED cloned-head-differs-from-the-receipt as expected: %s' % str(exc)[:200])
    with tempfile.TemporaryDirectory() as tmp:
        for name in PINS['config']['plan_order']:
            calls = []
            empty = lambda *a, **k: calls.append(a) or type('R', (), dict(returncode=0, stdout='', stderr=''))()
            plan = PINS['config']['plans'][name]
            value = c.queue_gate(Path(tmp), PINS, plan, 'queue-%s.json' % name, runner=empty, user='ycao11')
            applied = value is not None or bool(calls) or (Path(tmp) / ('queue-%s.json' % name)).exists()
            if plan['partition'] == 'ghx4':      # no per-user submit limit: no squeue, no record
                ok &= not applied
                print('QUEUE_GATE_%s %s partition=ghx4 squeue_calls=%d' % ('APPLIED_UNEXPECTEDLY' if applied else
                                                                          'NOT_APPLIED', name, len(calls)))
                continue
            # ghx4-interactive (one submitted job per user): squeue of that partition; an empty queue passes, a job
            # of the user there refuses the claim.
            passed = applied and value['status'] == 'QUEUE_GATE_PASS' and calls[0][0] == c.queue_command(
                plan['partition'], 'ycao11')
            busy = lambda *a, **k: type('R', (), dict(returncode=0, stdout='3430001|RUNNING|ghx4-interactive|x\n',
                                                      stderr=''))()
            try:
                c.queue_gate(Path(tmp), PINS, plan, 'queue-%s-busy.json' % name, runner=busy, user='ycao11')
                refused_busy = False
            except RuntimeError:
                refused_busy = True
            ok &= passed and refused_busy
            print('QUEUE_GATE_%s %s partition=%s empty=%s busy=%s' % ('APPLIED' if passed and refused_busy else
                                                                      'WRONG', name, plan['partition'],
                                                                      value and value['status'],
                                                                      'REFUSED' if refused_busy else 'PASSED'))
    for name in PINS['config']['plan_order']:
        cost = c.plan_cost(PINS, name)
        print('COST %s partition=%s billing_factor=%d gpu_hours_limit=%.2f gpu_hours_expected=%.2f' % (
            name, PINS['config']['plans'][name]['partition'], cost['billing_factor'], cost['gpu_hours_limit'],
            cost['gpu_hours_expected']))
    print('GATE_CHECKS_%s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
