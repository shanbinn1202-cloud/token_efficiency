"""Unit tests of perf-swe32 (local only): config and plans (the Table 1 five single-server ghx4 batch plans: full,
goalb4 = Ours (Goal), obs, rfill, stock; the optional goal; the ghx4-interactive variant of each; their time fit, limits
close to the expected time and billed GPU-hours), the per-policy flags with --decode-fastpath, --decode-fastpath2,
--decode-fastpath3, --deferred-completion-io (the switches, on), --connector-pool-gib 1 and --language-model-only, the
Goal rewriter batch with its token cap, capacity and memory at K = 16 within --mem=113560M (the 70-GiB arena), gates
(the queue gate over every harness on tbench with its afterany chain, the pinned release-check receipt of the light
login gates) and the sealed runtime's features, the job's budget (per-kind reserves), outcomes and archives, the tbench link's lock wait,
the failure classes, the model override and the pool spec, the server checks of the new switches, the Goal-text reader /
verifier (goal_text.py) and the statements.

Run: python -B -m unittest test_swe32 -v   (TEMP/TMP/TMPDIR on D:, localtemp.py)
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'postrun')]
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402
import goal_text  # noqa: E402
import job  # noqa: E402
import smoke_checks as sc  # noqa: E402
import swe_checks  # noqa: E402
import tbench_link  # noqa: E402

PINS = c.load(ROOT / 'pins.json')
SHARED = ['--kv-cache-gib', '32', '--page-tokens', '16', '--kv-layout', 'decoupled', '--cuda-graphs', 'full',
          '--max-running-sessions', '16', '--max-tokens', '4096', '--max-model-len', '131072', '--dram-arena-gib', '70',
          '--connector-pool-gib', '1', '--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3',
          '--deferred-completion-io', '--language-model-only']
REWRITER = '/projects/x/qwen3-0.6b'
PILOT = ['astropy__astropy-12907', 'astropy__astropy-13236', 'astropy__astropy-14096', 'astropy__astropy-14309',
         'astropy__astropy-14369', 'astropy__astropy-14508', 'astropy__astropy-14539', 'astropy__astropy-14995',
         'astropy__astropy-7336', 'astropy__astropy-7671']
# perf-decode2 job 3379206's text-only start (vLLM 0.24.0 --language-model-only; every server, stock and ours, logs
# these three): st-async's lines, verbatim from its server log.
JOB3379206_TEXT_ONLY_LOG = (
    '(APIServer pid=1230082) INFO 10-16 02:24:45 [registry.py:134] All limits of multimodal modalities supported by '
    'the model are set to 0, running in text-only mode.\n'
    '(EngineCore pid=1231971) INFO 10-16 02:25:09 [registry.py:134] All limits of multimodal modalities supported by '
    'the model are set to 0, running in text-only mode.\n'
    '(EngineCore pid=1231971) INFO 10-16 02:25:10 [mm_encoder_attention.py:373] Using AttentionBackendEnum.FLASH_ATTN '
    'for MMEncoderAttention.\n')
# The two multi-modal startup lines of a start without the flag (perf-decode2 fixtures, job 3364292's stock server).
ENCODER_CACHE_LINE = ('(EngineCore pid=3577418) INFO 10-12 00:40:10 [gpu_model_runner.py:6271] Encoder cache will be '
                      'initialized with a budget of 16384 tokens, and profiled with 1 image items of the maximum '
                      'feature size.\n')
WARMUP_LINE = '(APIServer pid=3572311) INFO 10-12 00:42:44 [base.py:223] Multi-modal warmup completed in 13.593s\n'
SWE16_TASKS = json.loads((ROOT.parent / 'perf-swe16' / 'tasks' / 'swe16-tasks.json').read_bytes())['instances'] \
    if (ROOT.parent / 'perf-swe16' / 'tasks' / 'swe16-tasks.json').is_file() else None
FIVE = ['full', 'goalb4', 'obs', 'rfill', 'stock']  # the Table 1 configurations (goalb4 = Ours (Goal))
BATCH = FIVE + ['goal']  # plus the optional goal_query unbatched
CTX = ['ctx24', 'ctx25']  # the group's new settings (2026-10-26): 262,144 tokens at 24 / 25 GiB, goal + obs in one job
VLLM = ['vllm25']  # integrate-1030 (2026-10-30): ctx25's goal server with the vLLM rewriter and --goal-overlap
STOCK25 = ['stock25']  # 2026-11-02: plan stock's server at the runtime servers' 262,144 tokens and 25 GiB of KV
STOCK25X = ['stock25x']  # 2026-11-02 (evening): stock25 with the agent cap 3,600 s (job 3506513 hit 1,800 s)
SMOKE = ['smoke']  # 2026-11-03: the release smoke (role release-smoke), three servers on integrate-1103
# 2026-11-06: the recent-window policies (required) and the thinking-on obs / goal comparisons, thinking on, one server each
THINK = ['recentt', 'rtoolst', 'obst', 'goalt', 'akvt']  # akvt: the AKV rule (akv-rule, 2026-11-07)
AKVS = ['akvs']  # 2026-11-07: AKV small-scale (the first 8 instances, 4 sessions), thinking on
SMOKET = ['smoket']  # 2026-11-09 (main ba4d1ba): the release smoke with thinking on (keep) and async, role release-smoke
# 2026-11-11: the ghx4-interactive variants of the thinking-on plans goalt / akvt / rtoolst (cap refitted to 02:00:00)
THINKI = ['goalti', 'akvsi', 'rtoolsti']
# 2026-11-11: the thinking-on full control (policy full, thinking on, obst's settings) and its interactive variant
FULLT = ['fullt', 'fullti']
# 2026-11-12: the 64-instance proxy (obs64 / obs64i / goal64 / full64) and the full baseline with idle release at 32
# (fulltr / fulltri); thinking on, idle release on the full plans
NEW64 = ['obs64', 'obs64i', 'goal64', 'full64', 'fulltr', 'fulltri']
# 2026-11-12 (later): the proxy's other policies (recent_fill, akv_rule with idle release, random_fill and its
# ghx4-interactive variant) and the stock control with thinking ON at 64 and 32 tasks
NEW1112 = ['recent64', 'akv64', 'rfill64', 'rfill64i', 'stock64t', 'stock32t', 'stock32ti']  # (stock32ti: the user's
# priority of 11-12: stock32t on ghx4-interactive)
# 2026-11-16: the stock keep-reasoning pair (thinking on, the earlier reasoning kept) and the async pair on recentt's
# recent_fill server (async_scheduling true / false)
KEEP1116 = ['stockkeep32t', 'stockkeepsync32t']
ASYNC1116 = ['rfillasync32t', 'rfillsync32t']
# 2026-11-16 (round 2): the async pair's ghx4-interactive variants and the second half (`tasks` "h2" = instances 33-64:
# our runtime with async on and its sync twin per policy, the stock control dropped and kept)
TI1116 = ['rfillasync32ti', 'rfillsync32ti']
H2RUNTIME = ['fullh2a', 'fullh2s', 'recenth2a', 'recenth2s', 'obsh2a', 'obsh2s', 'goalh2a', 'goalh2s']
H2STOCK = ['stockh2t', 'stockkeeph2t']
H2 = H2RUNTIME + H2STOCK
# 2026-11-16 (harness-1116): the scripted edge-case smoke of keep-rebuild-1116 (async on, the user's configuration) and
# its sync twin; blocked on 4ebff97
EDGE1116 = ['keepedge', 'keepedgesync']
NEW1116 = KEEP1116 + ASYNC1116 + TI1116 + H2 + EDGE1116
# 2026-11-14 (perf-swe32-ledger, 51b4085; folded in 2026-11-16): ledger_query at obst's (thinking on), ctx25's obs262
# (thinking off) and obs64's settings, and ledgert's ghx4-interactive variant; LEDGERT: the thinking-on ones
LEDGER1114 = ['ledgert', 'ledger', 'ledger64', 'ledgerti']
LEDGERT = ['ledgert', 'ledger64', 'ledgerti']
# 2026-11-16 (phase B, integrate-1116 a2f9e36): the 64-task plans with async ON (K = 16) and their K = 24 twins; the
# stock keep-reasoning control at 64 tasks (K = 16 / 24); the --gdn-inplace-crossing OFF / ON pairs (scripted client)
PB64 = ['full64a', 'full64ak24', 'recent64a', 'recent64ak24', 'obs64a', 'obs64ak24', 'goal64a', 'goal64ak24',
        'ledger64a', 'ledger64ak24']
PB64BASE = dict(full64a='full64', recent64a='recent64', obs64a='obs64', goal64a='goal64', ledger64a='ledger64')
PBSTOCK = ['stockkeep64t', 'stockkeep64k24']
GDN = ['gdnobs', 'gdnobson', 'gdngoal', 'gdngoalon', 'gdnidle', 'gdnidleon', 'gdnobs24', 'gdnobs24on']
GDN_ON = [n for n in GDN if n.endswith('on')]
# 2026-11-16 (the user's decision): the functional jobs short on ghx4-interactive: the *i variants, one job per
# OFF / ON pair (the ON server's own gdn_inplace_crossing), the 24-session test sess24, keepedgegoalp (blocked)
FUNC = ['keepedgei', 'keepedgesynci', 'gdnobsp', 'gdnobspi', 'gdngoalp', 'gdngoalpi', 'gdnidlep', 'gdnidlepi',
        'gdnobs24p', 'gdnobs24pi', 'sess24', 'sess24i', 'keepedgegoalp', 'keepedgegoalpi',
        'sess24r', 'sess24ri']  # (11-17: the K = 24 stable configuration's smoke: idle release, admission overlap 24)
PAIRS = ['gdnobsp', 'gdnobspi', 'gdngoalp', 'gdngoalpi', 'gdnidlep', 'gdnidlepi', 'gdnobs24p', 'gdnobs24pi']
GOALEDGE = ['keepedgegoal', 'keepedgegoalov', 'keepedgegoalp', 'keepedgegoalpi']  # goal-rollback-1116 (blocked)
PHASEB = PB64 + PBSTOCK + GDN + ['keepedgegoal', 'keepedgegoalov'] + FUNC
# 2026-11-17 (integrate-1117 2fb7f62): the group's 150-instance runs at 24 sessions (`tasks` 150): full and ledger_query
# (the K = 24 64-task plans' servers) and the stock keep-reasoning control at 24 sequences
P150 = ['full150k24', 'ledger150k24', 'stockkeep150k24']
P150RUNTIME = ['full150k24', 'ledger150k24']
# 2026-11-17: release gate 2 (full against stock, token for token; parity.py): one plan, our full server then the stock
# control with the probe client. The earlier tests' plan-wide properties hold for every other plan (PLAN_ORDER).
GATE2 = ['parity150']
K24 = [n for n in PB64 if n.endswith('k24')] + ['stockkeep64k24']
KEPT = KEEP1116 + ['stockkeeph2t'] + PBSTOCK + ['stockkeep150k24']  # every plan with the stock keep-reasoning history
ASYNC_ON = ['rfillasync32t', 'rfillasync32ti', 'fullh2a', 'recenth2a', 'obsh2a', 'goalh2a', 'keepedge'] + PB64 + GDN \
    + ['keepedgegoal', 'keepedgegoalov'] + [n for n in FUNC if n != 'keepedgesynci'] + P150RUNTIME
ASYNC_NAMED = ASYNC1116 + TI1116 + H2RUNTIME + EDGE1116 + PB64 + GDN + ['keepedgegoal', 'keepedgegoalov'] + FUNC \
    + P150RUNTIME
CLIENT_PLANS = EDGE1116 + GDN + ['keepedgegoal', 'keepedgegoalov'] + FUNC  # the scripted client's plans (bound 1536)
PLANS = (BATCH + [name + 'i' for name in BATCH] + CTX + VLLM + STOCK25 + STOCK25X + SMOKE + THINK + AKVS + SMOKET
         + THINKI + FULLT + NEW64 + NEW1112 + NEW1116 + LEDGER1114 + PHASEB + P150 + GATE2)
# the plan order before 2026-11-16 (the slices of the earlier tests count from its end)
ORDER1112 = [n for n in c.cfg(PINS)['plan_order'] if n not in NEW1116 + LEDGER1114 + PHASEB + P150 + GATE2]
# every plan but gate 2's (2026-11-17): the earlier tests' plan-wide properties
PLAN_ORDER = [n for n in c.cfg(PINS)['plan_order'] if n not in GATE2]
# 2026-11-12: describe() of policy-registry-1112 755550e's source-overlay/policy_registry.py (integrate-1112's is the
# same: the registry parity of 4ebff97 is byte-identical). 2026-11-14 (perf-swe32-ledger): the rows of ledger-int-1112
# 51b4085 as registry_rows reads them (describe(): the eight built-in policies, the release plug-in ledger_query and
# `protocols`; plus each policy's `hooked`)
# 2026-11-16 (phase B): the rows of integrate-1116 a2f9e36 (ledger_query v2, then the marker plug-ins
# goal_query_markers / observation_query_markers); 51b4085's: REGISTRY_ROWS_51B4085
REGISTRY_ROWS = json.loads((ROOT / 'fixtures' / 'policy-registry-a2f9e36.json').read_bytes())['rows']
REGISTRY_ROWS_51B4085 = json.loads((ROOT / 'fixtures' / 'policy-registry-51b4085.json').read_bytes())['rows']
REGISTRY_ROWS_755550E = json.loads((ROOT / 'fixtures' / 'policy-registry-755550e.json').read_bytes())['rows']
BUILTIN = ['full', 'random50', 'goal_query', 'observation_query', 'random_fill', 'recent_fill', 'recent_tools', 'akv_rule']
REGISTERED = BUILTIN + ['ledger_query', 'goal_query_markers', 'observation_query_markers']  # a2f9e36: the built-in
# policies, then the release plug-ins (51b4085: ledger_query only)
HOOKED = ('ledger_query', 'goal_query_markers', 'observation_query_markers')
RECEIPT_120BC3C = dict(commit='120bc3c0a6db281eeba21f541957c3ae6941542d', status='GIT_ARCHIVE_TESTS_PASS',
                       tests_run=1384, skipped=113,
                       result_line_sha256='c7e42eda95660ac4dc23cfa29ec8d3bd069d8fa91b38d242343e599521b03fbc',
                       log_sha256='0f547e90453f709a8316de916ca61a5cbc03e769d1bfb671475392df1f772d68')
# THE pin (one place for a re-pin): integrate-1103 ea18256 (c07d42a + async-c --async-scheduling + async-int +
# defaults-on + parallel-check; the release runtime of the release smoke matrix; --async-scheduling is used by plan
# smoke only, the vLLM Goal rewriter and --goal-overlap by vllm25 and smoke) and the parallel checker's release-check
# receipt (D:\Remote-RA-temp\scratch\parallel-check\release-receipt-ea18256.json, 2026-11-03; was rewriter-lfe-fix
# c07d42a, frozen in perf-swe32-03d3f747\). The cf2b28d state: perf-swe32-cf2b28d\; the 4f5f4e7 fallback:
# perf-swe32-4f5f4e7\.
# 2026-11-06: re-pinned to recent-fill-1103 60c5d21 (ea18256 + the recent-window policies recent_fill / recent_tools;
# the recent-window and thinking-on plans) with the parallel checker's receipt
# D:\Remote-RA-temp\scratch\recent-fill\release-receipt-60c5d21.json (was integrate-1103 ea18256, 1,698 tests);
# the same day re-pinned to recent-fill-1103 d72745c (recent_tools = tool calls, then tool responses, then reasoning)
# with D:\Remote-RA-temp\scratch\recent-fill\release-receipt-d72745c.json (60c5d21: 1,728 tests; its state, jobs
# 3539242 / 3539260: perf-swe32-fbd644ce\)
# 2026-11-07: re-pinned to recent-fill-1103 f4ccd4b (d72745c + akv_rule; plan akvt) with
# D:\Remote-RA-temp\scratch\recent-fill\release-receipt-f4ccd4b.json (d72745c's state, job 3540110: perf-swe32-20a5d68f\)
# 2026-11-09: re-pinned to main ba4d1ba (integrate-1107: the async fixes, tests/; plan smoket) with
# D:\Remote-RA-temp\scratch\main-ba4d1ba\release-receipt-ba4d1ba.json (f4ccd4b's state, jobs 3549065 / 3554749:
# perf-swe32-3ca7f1c2\)
# 2026-11-12: re-pinned to integrate-1112 4ebff97 (the policy registry, async-maxtok-1112) with
# D:\Remote-RA-temp\scratch\integrate-1112\receipt\release-receipt-4ebff97.json (ba4d1ba's state, payload 67c55f26,
# job 3583081: perf-swe32-67c55f26\)
RECEIPT_BA4D1BA = dict(commit='ba4d1ba4fc37ed9d3d593cdc6333b43628b36903', status='GIT_ARCHIVE_TESTS_PASS',
                       tests_run=1852, skipped=135,
                       result_line_sha256='887112869ed0fcd4f33f26d06c2b1f2b0edfa144e7ed0e649cd06c6ed16dcfac',
                       log_sha256='11319ba5426339ebeadce81d72f5a05e2941a9668b92481bc83abddfc397d36c')
# 2026-11-14 (perf-swe32-ledger): re-pinned to ledger-int-1112 51b4085 (integrate-1112 4ebff97 + ledger-policy-1112
# ed1cbf3) with D:\Remote-RA-temp\scratch\ledger-policy-1112\receipt\release-receipt-51b4085.json (4ebff97's receipt:
# RECEIPT_4EBFF97, the pin of perf-swe32 itself)
# 2026-11-16 (phase B): re-pinned to integrate-1116 a2f9e36 (a6d3ef7 + keep-rebuild-1116 + gdn-inplace-1116 +
# sessions-32-1116 + ledger-fixes; the coordinator's release check of 12:12-12:21Z) with
# D:\Remote-RA-temp\scratch\harness-1116\receipt\release-receipt-a2f9e36.json (51b4085's state, payload b25924bc,
# jobs 3611917 / 3612333: perf-swe32-b25924bc\)
# 2026-11-16 (phase B, later): re-pinned to integrate-1116 7a8c36e (a2f9e36 + goal-rollback-1116 43ff7b0) with
# D:\Remote-RA-temp\scratch\harness-1116\receipt\release-receipt-7a8c36e.json (a2f9e36's: RECEIPT_A2F9E36)
# 2026-11-17: re-pinned to integrate-1117 2fb7f62 (06c1e90: the session pool registers up to 256 tasks; docs;
# examples/swebench/verified-150.txt) with D:\Remote-RA-temp\scratch\harness-1117-150\receipt\release-receipt-2fb7f62.json
# (7a8c36e's: RECEIPT_7A8C36E; payload 9735ad9a, job 3616345: perf-swe32-9735ad9a\)
RECEIPT_PIN = dict(commit='2fb7f6241fe00517a472f8703125de153ec715d2', status='GIT_ARCHIVE_TESTS_PASS',
                   tests_run=2287, skipped=141,
                   result_line_sha256='0911337e39740917e0c1d296cc0df6069220eecf1852346e3411724069707ff1',
                   log_sha256='7aa59f00c91649e66901930d2762b98d484af1972fab81a4aefd7dd139fb939c')
RECEIPT_7A8C36E = dict(commit='7a8c36e8578feb500a787e5c260b1be6a3c8aa09', status='GIT_ARCHIVE_TESTS_PASS',
                   tests_run=2277, skipped=141,
                   result_line_sha256='797122f9dec4883b2d2137dbf6afbbdafc2d756e76385e4beb1a434f5eab352c',
                   log_sha256='b15e9b95f02bb17cfc9177b23e2f13ff272d631664fc3b4046e1808454636cce')
RECEIPT_A2F9E36 = dict(commit='a2f9e36db798036ff3adb5018474f14009025b61', status='GIT_ARCHIVE_TESTS_PASS',
                       tests_run=2263, skipped=141,
                       result_line_sha256='323200d4829b2662d126ab826c1ba48ef31472c3cfd342376cd47b75d98bd9ff',
                       log_sha256='500b6160324c0e7b79163410ec297ccb8a89fc93559ccadba43634edf368993e')
RECEIPT_51B4085 = dict(commit='51b4085409e3f8a8b58ccf9b8ef4465e9d496b7b', status='GIT_ARCHIVE_TESTS_PASS',
                   tests_run=2086, skipped=139,
                   result_line_sha256='83103d9a941208557f1b0c341420f27ebe56d1effb38c1c91e563aea6384b5a2',
                   log_sha256='1c5417d2140b14ec41a9d0a2b793ba58a6b6efb1f3239416d7209f2799c81701')
RECEIPT_4EBFF97 = dict(commit='4ebff978661f4c5e2e06da0ee68a5217f92dfa82', status='GIT_ARCHIVE_TESTS_PASS',
                       tests_run=2059, skipped=139,
                       result_line_sha256='308a6e79fc938fd2e2b62a87ebd7b7760703df31befac715fdb966459b30ea8f',
                       log_sha256='7bb8bcbd486c873fcab85964d9f59f4213c503fe4c8a2a785bbc424b54b82a6c')
RECEIPT_F4CCD4B = dict(commit='f4ccd4bce7af10f3e143a835d49b61b731ba0542', status='GIT_ARCHIVE_TESTS_PASS',
                       tests_run=1763, skipped=126,
                       result_line_sha256='130c15e41d1c22748101aa78507cc242716dd585e862965c77e1fa602e239b95',
                       log_sha256='d1257356fb9268d1a0c08f5c2e95ac4e390775d32affcbb5921cfc375c2fd9c9')
# The decided batch plans: (label, policy, tasks, step limit, cap, minimum), Slurm time, expected (goal: swe32 job
# 3404696's elapsed; goalb4: job 3404230's + 15 % of its batching saving for the 16384-token cap; the others: swe16's)
DECIDED = dict(full=(('full', 'full', 32, 120, 1920, 1500), '01:01:00', 2618),
               goalb4=(('goalb4', 'goal_query', 32, 120, 3540, 2650), '01:28:00', 4057),
               goal=(('goal', 'goal_query', 32, 120, 4380, 3150), '01:42:00', 4685),
               obs=(('obs', 'observation_query', 32, 120, 2280, 1800), '01:07:00', 3065),
               rfill=(('rfill', 'random_fill', 32, 120, 2400, 1900), '01:09:00', 3118),
               stock=(('stock', 'stock', 32, 120, 1800, 1400), '00:51:00', 2118))


def server(plan, index=0, pins=PINS):
    return c.plan_servers(pins, plan)[index]


def pins_51b4085_rows():
    """The pins with 51b4085's registry rows (2026-11-16, phase B): a2f9e36's rows add the marker plug-ins, and
    goal_query_markers (goal_engine) joins every runtime server expectation's rewriter / parking lists; with the old
    rows every earlier fingerprint is unchanged."""
    pins = copy.deepcopy(PINS)
    pins['runtime_features'][c.REGISTRY_KEY] = copy.deepcopy(REGISTRY_ROWS_51B4085)
    pins['runtime_features'].pop(c.GOAL_ROLLBACK_KEY, None)  # (7a8c36e: every runtime expectation gains goal_rollback)
    return pins


class Plans(unittest.TestCase):
    def test_every_plan_validates_and_fits_its_time(self):
        for name in c.cfg(PINS)['plan_order']:
            fit = c.validate_plan(PINS, name)
            self.assertTrue(fit['fits'], name)
            self.assertLessEqual(fit['worst_case_seconds'], fit['time_seconds'])
            self.assertLessEqual(fit['with_eval_wait_seconds'], fit['time_seconds'])
        self.assertEqual(c.cfg(PINS)['plan_order'], PLANS)  # (2026-11-17: gate 2's plan last)
        fits = {n: c.plan_fit(PINS, n) for n in BATCH}
        # every reserve + the cap = the limit exactly (spare 0); with the 600-s evaluation wait the minimum still fits
        self.assertEqual({n: f['worst_case_seconds'] for n, f in fits.items()},
                         dict(full=3660, goal=6120, obs=4020, rfill=4140, stock=3060, goalb4=5280))
        self.assertEqual({n: f['with_eval_wait_seconds'] for n, f in fits.items()},
                         dict(full=3420, goal=5070, obs=3720, rfill=3820, stock=2840, goalb4=4570))
        self.assertEqual({n: f['finish_seconds'] for n, f in fits.items()},
                         dict(full=480, goal=480, obs=480, rfill=480, stock=180, goalb4=480))
        self.assertTrue(all(f['optional'] == [] and f['all_caps_fit'] for f in fits.values()))
        for name in BATCH:  # each interactive variant: the batch plan's servers and fit on ghx4-interactive
            self.assertEqual(c.plan_fit(PINS, name + 'i'), fits[name], name)
            self.assertEqual(c.plan_servers(PINS, name + 'i'), c.plan_servers(PINS, name), name)
            self.assertLessEqual(fits[name]['time_seconds'], c.time_seconds('02:00:00'), name)

    def test_limits_stay_close_to_the_expected_time_and_cost(self):
        # about 1.3 x the swe16 elapsed, rounded up so the agents keep >= 1.05 x the swe16 agent wall after every
        # reserve (full and stock: their short jobs carry the fixed reserves); refused above 1.5 x
        # walls: swe16's agent walls; goal swe32 job 3404696's; goalb4 job 3404230's 3,020 s + 15 % of 705 s
        walls = dict(full=1800.0, goal=3725.1, obs=2135.0, rfill=2255.0, stock=1700.0, goalb4=3126.0)
        total = [0.0, 0.0]
        for name in BATCH:
            cost = c.plan_cost(PINS, name)
            self.assertEqual((cost['expected_seconds'], cost['gpus'], cost['billing_factor']), (DECIDED[name][2], 1, 1))
            self.assertLessEqual(cost['limit_over_expected'], c.MAX_LIMIT_OVER_EXPECTED, name)
            self.assertGreaterEqual(cost['limit_over_expected'], 1.3, name)
            self.assertGreaterEqual(server(name)['agent_cap_seconds'], 1.05 * walls[name], name)
            self.assertLessEqual(server(name)['agent_min_seconds'], 0.85 * walls[name], name)
            if name in FIVE:
                total[0] += cost['gpu_hours_limit']
                total[1] += cost['gpu_hours_expected']
            # the interactive variant bills double (the team usage log: GPU-hours, cores and memory x 2)
            twice = c.plan_cost(PINS, name + 'i')
            self.assertEqual((twice['billing_factor'], twice['gpu_hours_limit'], twice['gpu_hours_expected']),
                             (2, round(2 * cost['limit_seconds'] / 3600.0, 3), round(2 * cost['expected_seconds'] / 3600.0, 3)))
        self.assertEqual((round(total[0], 2), round(total[1], 2)), (5.6, 4.16))  # the Table 1 five
        self.assertEqual((c.plan_cost(PINS, 'goal')['gpu_hours_limit'], c.plan_cost(PINS, 'goal')['gpu_hours_expected']),
                         (1.7, 1.301))  # the optional goal
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['full']['expected_seconds'] = 2000  # 3660 s > 1.5 x 2000: too loose
        with self.assertRaisesRegex(RuntimeError, 'tight limits'):
            c.validate_plan(pins, 'full')
        pins['config']['plans']['full'].pop('expected_seconds')
        with self.assertRaisesRegex(RuntimeError, 'tight limits'):
            c.validate_plan(pins, 'full')

    def test_the_decisions(self):
        cfg = c.cfg(PINS)
        self.assertEqual({k: cfg['server'][k] for k in (
            'kv_cache_gib', 'page_tokens', 'kv_layout', 'max_model_len', 'max_tokens', 'dram_arena_gib', 'cuda_graphs',
            'max_running_sessions', 'decode_fastpath', 'decode_fastpath2', 'decode_fastpath3', 'deferred_completion_io',
            'language_model_only', 'connector_pool_gib')},
            dict(kv_cache_gib=32, page_tokens=16, kv_layout='decoupled', max_model_len=131072, max_tokens=4096,
                 dram_arena_gib=70, cuda_graphs='full', max_running_sessions=16, decode_fastpath=True,
                 decode_fastpath2=True, decode_fastpath3=True, deferred_completion_io=True, language_model_only=True,
                 connector_pool_gib=1))
        self.assertNotIn('release_idle_pages', cfg['server'])  # idle release off, as the SWE doc says
        self.assertEqual((cfg['agent']['step_limit'], cfg['agent']['max_tokens'], cfg['agent']['max_running']),
                         (120, 4096, 16))
        self.assertEqual((cfg['stock']['max_num_seqs'], cfg['stock']['language_model_only'],
                          cfg['stock']['async_scheduling']), (16, True, True))
        self.assertIs(cfg['stock']['stock_model_kwargs']['extra_body']['chat_template_kwargs']['enable_thinking'], False)
        for name, (row, time, expected) in DECIDED.items():
            servers = c.plan_servers(PINS, name)
            self.assertEqual([(s['label'], s['policy'], s['tasks'], s['step_limit'], s['agent_cap_seconds'],
                               s['agent_min_seconds']) for s in servers], [row], name)
            self.assertFalse(servers[0]['optional'])
            plan = cfg['plans'][name]
            self.assertEqual((plan['partition'], plan['time'], plan['expected_seconds'], plan['eval_wait_seconds']),
                             ('ghx4', time, expected, 600), name)
            self.assertTrue(plan['authorization'].startswith('AUTHORIZE_SWE32_%s_32TASKS_16SESSIONS_' % name.upper()))
            self.assertIn('_BATCH_REUSED_V9_ENV_ONE_GH200_', plan['authorization'])
            self.assertEqual(plan['role'], 'optional' if name == 'goal' else 'table1')
            variant = cfg['plans'][name + 'i']
            self.assertEqual((variant['partition'], variant['time'], variant['role'], variant['variant_of'],
                              variant['servers']), ('ghx4-interactive', time, 'interactive', name, plan['servers']))
            self.assertEqual(variant['authorization'], plan['authorization'].replace(
                'AUTHORIZE_SWE32_%s_' % name.upper(), 'AUTHORIZE_SWE32_%sI_' % name.upper()).replace('_BATCH_', '_INTERACTIVE_'))
        self.assertEqual({k: cfg['budget'][k] for k in ('job_checks_seconds', 'server_start_seconds', 'tunnel_seconds',
                                                         'stop_reserve_seconds', 'check_seconds', 'finish_seconds',
                                                         'stock_check_seconds', 'stock_finish_seconds',
                                                         'margin_seconds')},
                         dict(job_checks_seconds=60, server_start_seconds=360, tunnel_seconds=60, stop_reserve_seconds=180,
                              check_seconds=300, finish_seconds=480, stock_check_seconds=120, stock_finish_seconds=180,
                              margin_seconds=300))
        self.assertEqual(cfg['policies']['goal_query']['admission'], ['admission_overlap=16'])
        self.assertEqual(PINS['slurm']['mail_user'], 'cao58@illinois.edu')
        self.assertEqual(PINS['slurm']['mail_type'], 'BEGIN,END,FAIL')
        self.assertEqual(PINS['slurm']['job_name_prefix'], 'swe32-')
        self.assertEqual(PINS['slurm']['mem'], '113560M')
        self.assertEqual(PINS['clean_env']['job_id'], '3270721')
        self.assertEqual(PINS['commit'], RECEIPT_PIN['commit'])  # main 4f5f4e7 (was integrate-1019-tbgoal 120bc3c)
        self.assertEqual({k: PINS['release_receipt'][k] for k in RECEIPT_PIN}, RECEIPT_PIN)

    def test_the_task_list_is_swe16s_first_32(self):
        ids = c.instance_ids(PINS)
        self.assertEqual(len(ids), 32)
        if SWE16_TASKS is not None:
            self.assertEqual(ids, SWE16_TASKS)  # the same instances in the same order as perf-swe16's formal runs
        self.assertEqual(ids[:2], ['astropy__astropy-12907', 'astropy__astropy-13033'])
        self.assertEqual(ids[-1], 'django__django-11099')
        self.assertFalse(PINS['tasks']['provisional'])
        pool = PINS['tasks']['pool']
        self.assertEqual(sorted(pool['statements']), sorted(ids))
        self.assertEqual(sorted(pool['pilot_pins_checked']), sorted(PILOT))  # the ten pilot statements = their pins
        import build
        self.assertEqual({i: pool['statements'][i] for i in PILOT}, build.PILOT_STATEMENT_SHA256)

    def test_one_concurrency_everywhere(self):
        for key, path in (('agent', ('agent', 'max_running')), ('stock', ('stock', 'max_num_seqs'))):
            pins = copy.deepcopy(PINS)
            pins['config'][path[0]][path[1]] = 10
            with self.assertRaises(RuntimeError, msg=key):
                c.validate_plan(pins, 'goal')
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['full']['servers'][0]['tasks'] = 12  # fewer registered tasks than K = 16: refused
        with self.assertRaises(RuntimeError):
            c.validate_plan(pins, 'full')

    def test_optional_servers_come_after_the_required_ones(self):
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['full']['servers'][0]['optional'] = True  # no required server
        with self.assertRaises(RuntimeError):
            c.validate_plan(pins, 'full')
        pins['config']['plans']['full']['servers'][0]['optional'] = 'yes'
        with self.assertRaises(RuntimeError):
            c.validate_plan(pins, 'full')

    def test_a_server_without_tasks_registers_the_whole_list(self):
        self.assertEqual(server('goal')['tasks'], 32)
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['goal']['servers'][0]['tasks'] = 18
        self.assertEqual(server('goal', pins=pins)['tasks'], 18)
        self.assertEqual(c.server_instances(pins, server('goal', pins=pins)), c.instance_ids(PINS)[:18])


class PolicyFlags(unittest.TestCase):
    def args(self, plan, index=0, pins=PINS):
        return c.launch_arguments(pins, server(plan, index, pins), REWRITER)

    def test_each_policy_gets_exactly_the_decided_flags(self):
        self.assertEqual(self.args('full'), SHARED)
        self.assertEqual(self.args('obs'), SHARED + ['--active-token-budget', '8192'])
        self.assertEqual(self.args('rfill'), SHARED + ['--active-token-budget', '8192'])
        self.assertEqual(self.args('goal'), SHARED + ['--active-token-budget', '8192', '--admission-overlap', '16',
                                                      '--goal-rewriter-model', REWRITER])
        # Ours (Goal): the batch of 4 with the runtime's default token cap (16384, not passed; pinned by the server)
        self.assertEqual(self.args('goalb4'), SHARED + ['--active-token-budget', '8192', '--admission-overlap', '16',
                                                        '--goal-rewriter-model', REWRITER, '--goal-rewriter-batch', '4'])
        for plan in ('full', 'obs', 'rfill', 'goal', 'goalb4'):
            args = self.args(plan)
            self.assertEqual(self.args(plan + 'i'), args)  # the interactive variant runs the same server
            for flag in ('--gdn-device-handoff', '--phase1-in-step', '--release-idle-pages', '--goal-rewriter-batch-tokens'):
                self.assertNotIn(flag, args)
            for flag in ('--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3', '--deferred-completion-io',
                         '--language-model-only', '--connector-pool-gib'):
                self.assertEqual(args.count(flag), 1, (plan, flag))
            self.assertEqual(args[args.index('--dram-arena-gib') + 1], '70')
            self.assertEqual(args[args.index('--connector-pool-gib') + 1], '1')

    def test_the_fast_path_is_one_config_switch(self):
        pins = copy.deepcopy(PINS)
        pins['config']['server']['decode_fastpath'] = False
        pins['config']['server']['decode_fastpath2'] = False  # --decode-fastpath2 needs --decode-fastpath
        pins['config']['server']['decode_fastpath3'] = False  # --decode-fastpath3 needs --decode-fastpath2
        self.assertNotIn('--decode-fastpath', self.args('full', 0, pins))
        self.assertIsNone(c.server_expectation(pins, server('full', 0, pins), {}, 'm', [])['decode_fastpath'])
        self.assertTrue(c.server_expectation(PINS, server('full', 0), {}, 'm', [])['decode_fastpath'])

    def test_fastpath2_is_one_config_switch_on(self):
        # the user, 2026-10-16 (spd2 job 3379206: o16-fast2 PASS, W2 b16 0.914x fastpath1): on for every runtime plan
        self.assertIs(c.decode_fastpath2_on(PINS), True)
        off = copy.deepcopy(PINS)
        off['config']['server']['decode_fastpath2'] = False  # THE switch: one entry; off drops the flag and the echo
        off['config']['server']['decode_fastpath3'] = False  # (fastpath3 needs it: off with it)
        for plan in ('full', 'obs', 'rfill', 'goal', 'goalb4'):
            self.assertNotIn('--decode-fastpath2', self.args(plan, 0, off))
            self.assertIsNone(c.server_expectation(off, server(plan, 0, off), {}, 'm', [])['decode_fastpath2'])
            c.validate_plan(off, plan)
        pins = copy.deepcopy(PINS)
        for plan in ('full', 'obs', 'rfill', 'goal', 'goalb4'):
            args = self.args(plan, 0, pins)
            self.assertEqual(args.count('--decode-fastpath2'), 1)
            self.assertEqual(args.index('--decode-fastpath2'), args.index('--decode-fastpath') + 1)
            self.assertTrue(c.server_expectation(pins, server(plan, 0, pins), {}, 'm', [])['decode_fastpath2'])
            c.validate_plan(pins, plan)
        self.assertNotIn('--decode-fastpath2', c.stock_argv(pins, 8000))
        c.check_features(pins['runtime_features'], pins['config'])
        broken = copy.deepcopy(pins)
        broken['config']['server']['decode_fastpath'] = False  # serve.py refuses --decode-fastpath2 alone
        broken['config']['server']['decode_fastpath3'] = False
        with self.assertRaises(RuntimeError):
            c.check_features(broken['runtime_features'], broken['config'])
        with self.assertRaises(RuntimeError):
            c.launch_arguments(broken, server('full', 0, broken), REWRITER)
        with self.assertRaises(RuntimeError):  # a runtime without the switch
            c.check_features(dict(pins['runtime_features'], decode_fastpath2=None), pins['config'])
        c.check_features(dict(off['runtime_features'], decode_fastpath2=None), off['config'])  # off: not needed
        pins['config']['server']['decode_fastpath2'] = 'yes'
        with self.assertRaises(RuntimeError):
            c.check_features(pins['runtime_features'], pins['config'])

    def test_fastpath3_and_deferred_completion_are_config_switches_on(self):
        # the optimized runtime of record (the user, 2026-10-19/21): both on for every runtime plan, never stock
        self.assertIs(c.decode_fastpath3_on(PINS), True)
        self.assertIs(c.deferred_completion_io_on(PINS), True)
        for plan in ('full', 'obs', 'rfill', 'goal', 'goalb4'):
            args = self.args(plan)
            self.assertEqual(args.index('--decode-fastpath3'), args.index('--decode-fastpath2') + 1)
            self.assertEqual(args.index('--deferred-completion-io'), args.index('--decode-fastpath3') + 1)
            self.assertEqual(args.index('--language-model-only'), args.index('--deferred-completion-io') + 1)
            expect = c.server_expectation(PINS, server(plan), {}, 'm', [])
            self.assertEqual((expect['decode_fastpath3'], expect['deferred_completion_io']), (True, True))
        for flag in ('--decode-fastpath3', '--deferred-completion-io'):
            self.assertNotIn(flag, c.stock_argv(PINS, 8000))
        for key, flag in (('decode_fastpath3', '--decode-fastpath3'), ('deferred_completion_io', '--deferred-completion-io')):
            off = copy.deepcopy(PINS)
            off['config']['server'][key] = False  # THE switch: one entry
            self.assertNotIn(flag, self.args('goalb4', 0, off))
            self.assertIsNone(c.server_expectation(off, server('full', 0, off), {}, 'm', [])[key])
            c.check_features(dict(off['runtime_features'], **{key: None}), off['config'])  # off: not needed
            with self.assertRaises(RuntimeError, msg=key):  # on: a runtime without the switch is refused
                c.check_features(dict(PINS['runtime_features'], **{key: None}), PINS['config'])
            off['config']['server'][key] = 'yes'
            with self.assertRaises(RuntimeError, msg=key):
                c.check_features(off['runtime_features'], off['config'])
        alone = copy.deepcopy(PINS)
        alone['config']['server']['decode_fastpath2'] = False  # serve.py refuses --decode-fastpath3 alone
        with self.assertRaisesRegex(RuntimeError, 'decode_fastpath3 requires decode_fastpath2'):
            c.check_features(alone['runtime_features'], alone['config'])
        with self.assertRaises(RuntimeError):
            c.launch_arguments(alone, server('full', 0, alone), REWRITER)
        self.assertEqual(PINS['runtime_features']['decode_fastpath3'],
                         dict(serve_arg='--decode-fastpath3', config_key='decode_fastpath3', module=c.FASTPATH3_MODULE,
                              requires='--decode-fastpath2'))
        self.assertEqual(PINS['runtime_features']['deferred_completion_io'], c.DC_FEATURE)

    def test_the_connector_pool_follows_the_config(self):
        self.assertEqual((c.connector_pool_gib(PINS), c.connector_bytes(PINS)), (1, c.GIB))
        self.assertEqual(c.server_expectation(PINS, server('obs'), {}, 'm', [])['connector_pool_bytes'], c.GIB)
        pins = copy.deepcopy(PINS)
        pins['config']['server'].pop('connector_pool_gib')  # the 32-GiB default: no switch, 31 GiB more pinned
        self.assertNotIn('--connector-pool-gib', self.args('full', 0, pins))
        self.assertEqual(c.connector_bytes(pins), 32 * c.GIB)
        self.assertIsNone(c.server_expectation(pins, server('full', 0, pins), {}, 'm', [])['connector_pool_bytes'])
        self.assertAlmostEqual(c.pinned_host(pins, c.LAYOUT, 16)['gib'], 104.29, places=2)
        with self.assertRaisesRegex(RuntimeError, 'stay spare'):
            c.validate_plan(pins, 'full')
        for bad in (0, 32, '1', 1.0):
            config = copy.deepcopy(PINS['config'])
            config['server']['connector_pool_gib'] = bad
            with self.assertRaises(RuntimeError, msg=bad):
                c.check_features(PINS['runtime_features'], config)
        with self.assertRaises(RuntimeError):
            c.check_features(dict(PINS['runtime_features'], connector_pool=None), PINS['config'])

    def test_switching_admission_flags_is_one_config_entry(self):
        pins = copy.deepcopy(PINS)
        pins['config']['policies']['full']['admission'] = ['phase1_in_step', 'gdn_device_handoff']
        self.assertEqual(self.args('full', 0, pins), SHARED + ['--gdn-device-handoff', '--phase1-in-step'])
        self.assertEqual(c.server_expectation(pins, server('full', 0, pins), {}, 'm', [])['admission'],
                         dict(gdn_device_handoff=True, phase1_in_step=True))
        # (e) pins a second GDN carry per session: 75.59 GiB pinned + 32 leaves 3.3 GiB, inside the 5-GiB margin
        self.assertEqual(c.plan_carries(pins, 'full'), 2)
        with self.assertRaisesRegex(RuntimeError, 'stay spare'):
            c.validate_plan(pins, 'full')
        pins['config']['policies']['goal_query']['admission'] = []
        self.assertEqual(self.args('goal', pins=pins), SHARED + ['--active-token-budget', '8192', '--goal-rewriter-model',
                                                                 REWRITER])
        c.validate_plan(pins, 'goal')

    def test_bad_admission_flags_are_refused(self):
        for bad in (['admission_overlap=17'], ['admission_overlap=1'], ['phase1_in_step', 'phase1_in_step'], ['x']):
            pins = copy.deepcopy(PINS)
            pins['config']['policies']['goal_query']['admission'] = bad
            with self.assertRaises(Exception, msg=bad):
                c.validate_plan(pins, 'goal')
        pins = copy.deepcopy(PINS)
        pins['config']['policies']['goal_query']['admission'] = ['admission_overlap=8']
        c.validate_plan(pins, 'goal')

    def test_expectation_of_goal_query(self):
        expect = c.server_expectation(PINS, server('goal'), dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m',
                                                                 config_sha256='c' * 64), 'f' * 64, [{'instance_id': 'a'}])
        self.assertEqual(expect['admission'], dict(admission_overlap=dict(permits=16, admission_seconds=2100,
                                                                          lease_seconds=2160)))
        self.assertEqual((expect['running'], expect['max_num_seqs'], expect['active_token_budget']), (16, 16, 8192))
        self.assertEqual(expect['layout']['record'], dict(page_tokens=16, mamba_tokens=784, mamba_slots=33))
        self.assertEqual(expect['layout']['pools'], [27874, 100])  # swe16's pools at K = 16
        self.assertEqual(expect['layout']['dram_arena_pages'], 71680)  # --dram-arena-gib 70: 71,680 1-MiB Pages
        self.assertEqual(expect['layout']['arena_pages'], 71680)
        self.assertAlmostEqual(expect['layout']['pinned_host']['gib'], 73.29, places=2)
        self.assertEqual(expect['capture_sizes'], c.CAPTURE_SIZES_16)
        self.assertEqual(expect['parking_policies'], ['goal_query', 'goal_query_markers'])  # (a2f9e36's registry)
        self.assertTrue(expect['decode_fastpath'])
        self.assertIs(expect['decode_fastpath2'], True)
        self.assertEqual((expect['decode_fastpath3'], expect['deferred_completion_io']), (True, True))
        self.assertEqual((expect['goal_rewriter_batch'], expect['goal_rewriter_batch_tokens']), (None, None))
        self.assertEqual(expect['connector_pool_bytes'], c.GIB)
        self.assertEqual(expect['goal_text'], PINS['runtime_features'].get('goal_text'))
        self.assertEqual(expect['rewriter'], dict(model='/m', config_sha256='c' * 64))

    def test_capture_sizes_follow_vllm(self):
        self.assertEqual(c.capture_sizes(16), c.CAPTURE_SIZES_16)
        self.assertEqual(c.capture_sizes(8), [1, 2, 4, 8, 16])  # job 3338893
        self.assertEqual(c.capture_sizes(10), [1, 2, 4, 8, 16])
        import smoke_checks
        for k in range(1, 17):
            self.assertEqual(c.capture_sizes(k), smoke_checks.capture_sizes(k))


class Capacity(unittest.TestCase):
    def test_pinned_host_and_attention_pool_at_the_decided_settings(self):
        host = c.pinned_host(PINS, c.LAYOUT, 16)
        self.assertTrue(host['fits'])
        self.assertEqual(host['arena_pages'], 71680)  # --dram-arena-gib 70: 280 chunks of 256 Pages
        self.assertEqual(host['arena_bytes'], 70 << 30)
        self.assertEqual(host['connector_bytes'], 1 << 30)  # --connector-pool-gib 1
        self.assertAlmostEqual(host['gib'], 73.29, places=2)  # 1 + 16 x 0.1434 + 70
        self.assertEqual(host['budget_gib'], 128)
        self.assertAlmostEqual(c.pinned_host(PINS, c.LAYOUT, 16, 2)['gib'], 75.59, places=2)
        attention = c.attention_fit(PINS, c.LAYOUT, 16)
        self.assertEqual((attention['fa_blocks'], attention['mamba_blocks']), (27874, 100))
        self.assertTrue(attention['fits'])
        self.assertEqual([c.plan_carries(PINS, name) for name in PLANS], [1] * len(PLANS))

    def test_the_memory_request_holds_the_worst_case(self):
        # --mem=113560M (one GH200's share); worst case: the 1-GiB pool + 16 carries + the 70-GiB arena (73.29 GiB
        # pinned) + the 32-GiB reserve = 105.29 GiB of 110.90: 5.61 GiB spare (at least 5 required)
        self.assertEqual(PINS['slurm']['mem'], '113560M')
        self.assertEqual((c.mem_mib('113560M'), c.mem_mib('144g'), c.mem_mib('144G')), (113560, 147456, 147456))
        for name in PLANS:
            value = c.memory_fit(PINS, name)
            self.assertTrue(value['fits'], name)
            c.validate_plan(PINS, name)
        for name in ('full', 'goal', 'obs', 'rfill'):
            self.assertEqual({k: c.memory_fit(PINS, name)[k] for k in (
                'mem_mib', 'mem_gib', 'within_share', 'pinned_host_gib', 'connector_pool_gib', 'carries_gib',
                'overhead_gib', 'need_gib', 'headroom_gib', 'min_spare_gib', 'spare_fits', 'arena_gib',
                'arena_demand_gib', 'arena_bound_gib', 'arena_fits')},
                dict(mem_mib=113560, mem_gib=110.9, within_share=True, pinned_host_gib=73.29, connector_pool_gib=1.0,
                     carries_gib=2.29, overhead_gib=32, need_gib=105.29, headroom_gib=5.61, min_spare_gib=5,
                     spare_fits=True, arena_gib=70, arena_demand_gib=27.71, arena_bound_gib=58.32, arena_fits=True),
                name)
        stock = c.memory_fit(PINS, 'stock')
        self.assertEqual((stock['pinned_host_gib'], stock['need_gib'], stock['headroom_gib']), (0.0, 32.0, 78.9))
        self.assertIsNone(stock['arena_gib'])
        args = c.sbatch_args(Path('/u/ycao11/swe32-x'), PINS, dict(plan='full', mail_user='cao58@illinois.edu',
                                                                   run_label='r1', after_job=None), 'o')
        self.assertIn('--mem=113560M', args)
        for mem in ('113561M', '144g', '200g'):  # above one GH200's share: refused by every plan, stock too
            pins = copy.deepcopy(PINS)
            pins['slurm']['mem'] = mem
            for name in ('full', 'stock'):
                self.assertFalse(c.memory_fit(pins, name)['within_share'], (mem, name))
                with self.assertRaisesRegex(RuntimeError, 'share'):
                    c.validate_plan(pins, name)
        pins = copy.deepcopy(PINS)
        pins['slurm']['mem'] = '110000M'  # 107.42 GiB: 2.1 GiB beside 105.29
        with self.assertRaisesRegex(RuntimeError, 'stay spare'):
            c.validate_plan(pins, 'goal')
        self.assertTrue(c.memory_fit(pins, 'stock')['fits'])
        pins = copy.deepcopy(PINS)
        pins['config']['memory']['process_overhead_gib'] = 34  # a larger reserve leaves 3.6 GiB: refused
        with self.assertRaisesRegex(RuntimeError, 'stay spare'):
            c.validate_plan(pins, 'full')
        pins['config']['memory']['process_overhead_gib'] = 30  # the lower end of the user's 30-32: 7.61 spare
        self.assertEqual(c.memory_fit(pins, 'full')['headroom_gib'], 7.61)
        for bad in ('113560', '110.9g', '113560K', ''):
            pins['slurm']['mem'] = bad
            with self.assertRaises(RuntimeError):
                c.memory_fit(pins, 'full')

    def test_the_arena_is_sized_from_the_measured_histories(self):
        pins = copy.deepcopy(PINS)
        pins['config']['server']['dram_arena_gib'] = 41  # below 1.5 x the measured 27.71 GiB peak (41.57)
        self.assertTrue(c.pinned_host(pins, c.LAYOUT, 16)['fits'])  # serve.py would take it
        self.assertFalse(c.memory_fit(pins, 'full')['arena_fits'])
        with self.assertRaisesRegex(RuntimeError, 'measured arena demand'):
            c.validate_plan(pins, 'goal')
        self.assertTrue(c.memory_fit(pins, 'stock')['fits'])  # no runtime server, no arena
        pins['config']['server']['dram_arena_gib'] = 58  # below swe16's 16 largest sessions at once (58.32 GiB)
        with self.assertRaisesRegex(RuntimeError, 'measured arena demand'):
            c.validate_plan(pins, 'full')
        pins['config']['server']['dram_arena_gib'] = 71  # 1 + 2.29 + 71 + 32 = 106.29: 4.61 spare < 5
        self.assertTrue(c.memory_fit(pins, 'full')['arena_fits'])
        with self.assertRaisesRegex(RuntimeError, 'stay spare'):
            c.validate_plan(pins, 'full')
        pins['config']['server']['dram_arena_gib'] = 70  # the largest arena that keeps 5 GiB spare
        c.validate_plan(pins, 'full')
        pins['config']['memory']['min_spare_gib'] = 6
        with self.assertRaisesRegex(RuntimeError, 'stay spare'):
            c.validate_plan(pins, 'full')

    def test_the_formal_arena_or_every_session_is_refused(self):
        pins = copy.deepcopy(PINS)
        pins['config']['server']['dram_arena_gib'] = 88  # swe16's arena: fits serve.py's 128 GiB, not the share
        self.assertTrue(c.pinned_host(pins, c.LAYOUT, 16)['fits'])
        with self.assertRaisesRegex(RuntimeError, 'stay spare'):
            c.validate_plan(pins, 'goal')
        pins['config']['server']['dram_arena_gib'] = 4  # less than one session at the limit
        self.assertFalse(c.pinned_host(pins, c.LAYOUT, 16)['fits'])
        pins['config']['server']['dram_arena_gib'] = 129  # holds every session at the limit: serve.py refuses it
        self.assertFalse(c.pinned_host(pins, c.LAYOUT, 16)['fits'])
        pins['config']['server']['dram_arena_gib'] = None  # room for every session: 131.3 GiB > 128
        self.assertFalse(c.pinned_host(pins, c.LAYOUT, 16)['fits'])
        with self.assertRaises(RuntimeError):
            c.validate_plan(pins, 'full')

    def test_language_model_only_follows_the_runtime_feature(self):
        # 227dedb: serve.py --language-model-only on every runtime server, vLLM's flag on the stock control
        self.assertEqual(PINS['runtime_features'].get('language_model_only'), c.LMO_FEATURE)
        self.assertTrue(c.cfg(PINS)['server']['language_model_only'] and c.cfg(PINS)['stock']['language_model_only'])
        self.assertEqual(c.cfg(PINS)['memory']['multimodal_warmup']['on_risk'], 'refuse')
        for plan in PLANS:
            for s in c.plan_servers(PINS, plan):
                args = c.launch_arguments(PINS, s, REWRITER) if s['kind'] == 'runtime' else c.stock_argv(PINS, 8000)
                self.assertEqual(args.count('--language-model-only'), 1, (plan, s['label']))
            self.assertEqual(c.memory_fit(PINS, plan)['startup'],
                             dict(status='ELIMINATED', need_gib=None, without_flag=[], on_risk='refuse', feature=True))
            c.validate_plan(PINS, plan)
        self.assertIs(c.stock_expected(PINS, 8000)['language_model_only'], True)
        self.assertEqual(c.server_expectation(PINS, server('full', 0), {}, 'm', [])['language_model_only'],
                         dict(mm_limits=dict(image=0, video=0), vision_tower_tensors=0))
        # the stock control without it: the stock plan refused (the warmup spike), runtime plans fine
        pins = copy.deepcopy(PINS)
        pins['config']['stock']['language_model_only'] = False
        self.assertEqual(c.memory_fit(pins, 'stock')['startup']['without_flag'], ['stock'])
        self.assertEqual(c.memory_fit(pins, 'goal')['startup']['status'], 'ELIMINATED')
        self.assertNotIn('--language-model-only', c.stock_argv(pins, 8000))
        self.assertNotIn('language_model_only', c.stock_expected(pins, 8000))
        with self.assertRaisesRegex(RuntimeError, 'multi-modal warmup'):
            c.validate_plan(pins, 'stock')
        c.validate_plan(pins, 'goal')
        # a runtime without the switch: refused by the features check and by every plan's memory fit
        pins = copy.deepcopy(PINS)
        pins['runtime_features']['language_model_only'] = None
        with self.assertRaises(RuntimeError):
            c.check_features(pins['runtime_features'], pins['config'])
        for plan in PLANS:
            for s in c.plan_servers(pins, plan):
                args = c.launch_arguments(pins, s, REWRITER) if s['kind'] == 'runtime' else c.stock_argv(pins, 8000)
                self.assertNotIn('--language-model-only', args)
            startup = c.memory_fit(pins, plan)['startup']
            self.assertEqual((startup['status'], startup['need_gib'], startup['feature']), ('RISK', 152.4, False), plan)
            with self.assertRaisesRegex(RuntimeError, 'multi-modal warmup'):
                c.validate_plan(pins, plan)
        self.assertIsNone(c.server_expectation(pins, server('full', 0, pins), {}, 'm', [])['language_model_only'])

    def test_language_model_only_feature_detection(self):
        serve = "text_only=[]\np.add_argument('--language-model-only',action='store_true')\n"
        module = ("CONFIG_KEY = 'language_model_only'   # x\nARGUMENT = '--language-model-only'\n"
                  "TOWER_MODALITIES = ('image', 'video')\n"
                  "RECORD = {'mm_limits': {m: 0 for m in TOWER_MODALITIES}, 'vision_tower_tensors': 0}\n"
                  "def worker_check(runner, startup):\n    pass\n")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.assertIsNone(c.language_model_only_feature(repo, serve))  # no module
            path = repo / c.LMO_MODULE
            path.parent.mkdir(parents=True)
            path.write_text(module, encoding='utf-8')
            self.assertEqual(c.language_model_only_feature(repo, serve), c.LMO_FEATURE)
            self.assertIsNone(c.language_model_only_feature(repo, "p.add_argument('--decode-fastpath')"))
            path.write_text(module.replace("'video'", "'audio'"), encoding='utf-8')
            self.assertIsNone(c.language_model_only_feature(repo, serve))

    def test_the_dry_run_command_carries_the_flag_last_and_the_pool(self):
        transfer = json.dumps(dict(kv_connector='OffloadingConnector', kv_role='kv_both',
                                   kv_connector_extra_config=dict(cpu_bytes_to_use=c.GIB)), separators=(',', ':'))
        command = ['python', '-c', 'x', '--model', 'm', '--kv-transfer-config', transfer, '--gdn-prefill-backend',
                   'triton', '--language-model-only']
        self.assertTrue(c.check_text_only_command(command, True))
        for bad in (command[:-1], command[:-1] + ['--language-model-only', '--x'],
                    command + ['--language-model-only']):
            with self.assertRaises(RuntimeError):
                c.check_text_only_command(bad, True)
        self.assertFalse(c.check_text_only_command(command[:-1], False))
        with self.assertRaises(RuntimeError):
            c.check_text_only_command(command, False)
        self.assertEqual(c.check_pool_command(command, c.GIB), c.GIB)
        with self.assertRaises(RuntimeError):  # the default 32-GiB pool where 1 GiB was asked for
            c.check_pool_command(command, 32 * c.GIB)
        with self.assertRaises(RuntimeError):
            c.check_pool_command(command[:5] + command[7:], c.GIB)

    def test_the_server_check_reads_the_text_only_records(self):
        record = dict(mm_limits=dict(image=0, video=0), vision_tower_tensors=0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            log = out / 'server.log'

            def write(on, ready_record=record, launch_last='--language-model-only', log_text=None):
                startup = dict(effective_server_config=dict(max_num_seqs=16, language_model_only=True) if on else
                               dict(max_num_seqs=16))
                ready, receipt = {}, {}
                if on:
                    startup['language_model_only'] = True
                    ready['language_model_only'], receipt['language_model_only'] = ready_record, record
                for name, value in (('startup.json', startup), ('ready.json', ready),
                                    ('worker-attached.json', dict(receipt=receipt)),
                                    ('launch.json', dict(command=['python', '--model', 'm', launch_last]))):
                    (out / name).write_text(json.dumps(value), encoding='utf-8')
                log.write_text(log_text if log_text is not None else
                               ('running in text-only mode\n' if on else 'Multi-modal warmup completed in 14s\n'),
                               encoding='utf-8')
            write(True)
            value = swe_checks.check_language_model_only(out, log, dict(language_model_only=record))
            self.assertEqual((value['language_model_only'], value['record'], value['launch_checked']),
                             (True, record, True))
            for kwargs in (dict(ready_record=dict(record, vision_tower_tensors=3)), dict(launch_last='--x'),
                           dict(log_text='Multi-modal warmup completed in 14s\n')):
                write(True, **kwargs)
                with self.assertRaises(swe_checks.CheckFailed, msg=kwargs):
                    swe_checks.check_language_model_only(out, log, dict(language_model_only=record))
            write(False, launch_last='--gdn-prefill-backend')
            self.assertFalse(swe_checks.check_language_model_only(out, log, dict(language_model_only=None))[
                'language_model_only'])
            write(True)
            with self.assertRaises(swe_checks.CheckFailed):  # records without the decision
                swe_checks.check_language_model_only(out, log, dict(language_model_only=None))
            self.assertTrue(swe_checks.check_text_only_log(log, True)['language_model_only'])
            log.write_text('x for MMEncoderAttention.\n', encoding='utf-8')  # the backend line alone: no text-only line
            with self.assertRaisesRegex(swe_checks.CheckFailed, 'text-only line'):
                swe_checks.check_text_only_log(log, True)

    def test_text_only_log_of_job_3379206(self):
        """vLLM 0.24.0's text-only start (perf-decode2 job 3379206, every server with --language-model-only): the
        text-only line from the API server and the engine core and the encoder backend line from the engine core. The
        backend line is counted only; the text-only line is required; an encoder cache or warmup line fails; without
        the flag nothing is required. The fakes log the same lines."""
        import fake_server
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'server.log'
            log.write_text(JOB3379206_TEXT_ONLY_LOG, encoding='utf-8')
            for on in (True, False):
                value = swe_checks.check_text_only_log(log, on)
                self.assertEqual(value['encoder_backend_lines'], 1)
                self.assertEqual(value['log'], {'running in text-only mode': 2, 'for MMEncoderAttention': 1,
                                                'Encoder cache will be initialized': 0,
                                                'Multi-modal warmup completed': 0})
            for line, message in ((ENCODER_CACHE_LINE, 'Encoder cache will be initialized'),
                                  (WARMUP_LINE, 'Multi-modal warmup completed')):
                log.write_text(JOB3379206_TEXT_ONLY_LOG + line, encoding='utf-8')
                with self.assertRaisesRegex(swe_checks.CheckFailed, 'multi-modal startup lines: .*%s' % message):
                    swe_checks.check_text_only_log(log, True)
                self.assertEqual(swe_checks.check_text_only_log(log, False)['log'][message], 1)
            for raw, on in ((fake_server.TEXT_ONLY_LINES, True), (fake_server.MULTIMODAL_LINES, False)):
                log.write_bytes(raw)
                self.assertEqual(swe_checks.check_text_only_log(log, on)['encoder_backend_lines'], 1)
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_text_only_log(log, True)  # the fakes' multi-modal start


class NewSwitchChecks(unittest.TestCase):
    """The server checks of --decode-fastpath2 and --connector-pool-gib (perf-decode2's checks on swe32's records)."""

    def write(self, out, fast2=False, pool=c.GIB, launch_pool=None, echo=('startup', 'ready', 'worker'), launch=True):
        startup, ready, receipt = dict(decode_fastpath=True), dict(decode_fastpath=True), dict(decode_fastpath=True)
        for name, value in (('startup', startup), ('ready', ready), ('worker', receipt)):
            if fast2 and name in echo:
                value['decode_fastpath2'] = True
        if pool is not None:
            startup['connector_pool_bytes'] = pool
        out.mkdir(parents=True, exist_ok=True)
        for name, value in (('startup.json', startup), ('ready.json', ready),
                            ('worker-attached.json', dict(receipt=receipt))):
            (out / name).write_text(json.dumps(value), encoding='utf-8')
        if launch:
            transfer = dict(kv_connector='OffloadingConnector', kv_role='kv_both',
                            kv_connector_extra_config=dict(cpu_bytes_to_use=launch_pool or pool or 32 * c.GIB))
            command = ['python', '--model', 'm', '--kv-transfer-config', json.dumps(transfer), '--language-model-only']
            (out / 'launch.json').write_text(json.dumps(dict(command=command)), encoding='utf-8')
        elif (out / 'launch.json').exists():
            (out / 'launch.json').unlink()

    def test_fastpath2_echo_exactly_when_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.write(out)
            self.assertEqual(swe_checks.check_decode_fastpath2(out, dict(decode_fastpath2=None)),
                             dict(decode_fastpath2=None))
            with self.assertRaises(swe_checks.CheckFailed):  # the switch is on, the records are not
                swe_checks.check_decode_fastpath2(out, dict(decode_fastpath2=True))
            self.write(out, fast2=True)
            self.assertEqual(swe_checks.check_decode_fastpath2(out, dict(decode_fastpath2=True)),
                             dict(decode_fastpath2=True))
            with self.assertRaises(swe_checks.CheckFailed):  # records without the switch
                swe_checks.check_decode_fastpath2(out, dict(decode_fastpath2=None))
            for missing in ('startup', 'ready', 'worker'):
                self.write(out, fast2=True, echo=[e for e in ('startup', 'ready', 'worker') if e != missing])
                with self.assertRaises(swe_checks.CheckFailed, msg=missing):
                    swe_checks.check_decode_fastpath2(out, dict(decode_fastpath2=True))

    def test_connector_pool_records_and_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.write(out)
            value = swe_checks.check_connector_pool(out, dict(connector_pool_bytes=c.GIB))
            self.assertEqual((value['connector_pool_bytes'], value['cpu_bytes_to_use'], value['launch_checked']),
                             (c.GIB, c.GIB, True))
            self.write(out, launch_pool=32 * c.GIB)  # vLLM got the default pool
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_connector_pool(out, dict(connector_pool_bytes=c.GIB))
            self.write(out, pool=None)  # startup without the pin
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_connector_pool(out, dict(connector_pool_bytes=c.GIB))
            self.assertEqual(swe_checks.check_connector_pool(out, dict(connector_pool_bytes=None))['cpu_bytes_to_use'],
                             32 * c.GIB)
            self.write(out, launch=False)
            self.assertFalse(swe_checks.check_connector_pool(out, dict(connector_pool_bytes=c.GIB))['launch_checked'])

    def test_fastpath3_and_deferred_echo_exactly_when_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            def write(startup, ready, worker):
                for name, value in (('startup.json', startup), ('ready.json', ready),
                                    ('worker-attached.json', dict(receipt=worker))):
                    (out / name).write_text(json.dumps(value), encoding='utf-8')
            on3 = dict(decode_fastpath2=True, decode_fastpath3=True)
            write(on3, on3, on3)
            self.assertEqual(swe_checks.check_decode_fastpath3(out, dict(decode_fastpath3=True)),
                             dict(decode_fastpath3=True))
            with self.assertRaises(swe_checks.CheckFailed):  # records without the switch
                swe_checks.check_decode_fastpath3(out, dict(decode_fastpath3=None))
            for missing in range(3):
                values = [dict(on3) for _ in range(3)]
                values[missing].pop('decode_fastpath3')
                write(*values)
                with self.assertRaises(swe_checks.CheckFailed, msg=missing):
                    swe_checks.check_decode_fastpath3(out, dict(decode_fastpath3=True))
            write({}, {}, {})
            self.assertEqual(swe_checks.check_decode_fastpath3(out, {}), dict(decode_fastpath3=None))
            with self.assertRaises(swe_checks.CheckFailed):  # the switch is on, the records are not
                swe_checks.check_decode_fastpath3(out, dict(decode_fastpath3=True))
            dc = dict(deferred_completion_io=True)
            write(dc, dc, {})  # engine only: startup and READY, never the worker receipt
            self.assertEqual(swe_checks.check_deferred_completion_io(out, dict(deferred_completion_io=True)),
                             dict(deferred_completion_io=True))
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_deferred_completion_io(out, {})
            for startup, ready, worker in ((dc, {}, {}), ({}, dc, {}), (dc, dc, dc)):
                write(startup, ready, worker)
                with self.assertRaises(swe_checks.CheckFailed, msg=(startup, ready, worker)):
                    swe_checks.check_deferred_completion_io(out, dict(deferred_completion_io=True))
            write({}, {}, {})
            self.assertEqual(swe_checks.check_deferred_completion_io(out, {}), dict(deferred_completion_io=None))

    def test_feature_detection_of_fastpath3_deferred_and_the_batch_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / 'source-overlay' / 'remote_ra_live_v2').mkdir(parents=True)
            serve3 = ("p.add_argument('--decode-fastpath3',action='store_true')\n"
                      "need(decode_fastpath2 or not decode_fastpath3,'--decode-fastpath3 requires --decode-fastpath2')\n")
            self.assertIsNone(c.decode_fastpath3_feature(repo, serve3))  # no module
            (repo / c.FASTPATH3_MODULE).write_text("CONFIG_KEY = 'decode_fastpath3'\n", encoding='utf-8')
            self.assertEqual(c.decode_fastpath3_feature(repo, serve3)['requires'], '--decode-fastpath2')
            self.assertIsNone(c.decode_fastpath3_feature(repo, serve3.splitlines()[0]))  # no refusal without fast2
            serve_dc = ("p.add_argument('--deferred-completion-io',action='store_true')\n"
                        "    config.update(deferred_completion_io=True)\n")
            self.assertIsNone(c.deferred_completion_feature(repo, serve_dc))
            (repo / c.DC_MODULE).write_text("CONFIG_KEY = 'deferred_completion_io'      # startup config\n",
                                            encoding='utf-8')
            self.assertEqual(c.deferred_completion_feature(repo, serve_dc), c.DC_FEATURE)
            (repo / c.DC_MODULE).write_text("CONFIG_KEY = 'other'\n", encoding='utf-8')
            with self.assertRaises(RuntimeError):
                c.deferred_completion_feature(repo, serve_dc)
            batch = repo / c.BATCH_MODULE
            batch.write_text("CONFIG_KEY = 'goal_rewriter_batch'\nLIMITS = (2, 16)\n", encoding='utf-8')
            old = ("p.add_argument('--goal-rewriter-batch',type=int)\nneed(goal_rewriter_batch<=admission_overlap)\n"
                   "        config.update(goal_rewriter_batch=goal_rewriter_batch)\n")
            self.assertEqual(c.goal_rewriter_batch_feature(repo, old),  # 5b15b2f: no cap
                             dict(serve_arg='--goal-rewriter-batch', config_key='goal_rewriter_batch', min=2, max=16))
            new = ("GOAL_REWRITER_BATCH_TOKENS=16384;GOAL_REWRITER_BATCH_TOKENS_LIMIT=16*262144\n"
                   "p.add_argument('--goal-rewriter-batch',type=int)\np.add_argument('--goal-rewriter-batch-tokens')\n"
                   "need(goal_rewriter_batch<=admission_overlap)\n"
                   "        config.update(goal_rewriter_batch=goal_rewriter_batch,goal_rewriter_batch_tokens=\n"
                   "                      GOAL_REWRITER_BATCH_TOKENS if goal_rewriter_batch_tokens is None else "
                   "goal_rewriter_batch_tokens)\n")
            self.assertEqual(c.goal_rewriter_batch_feature(repo, new)['serve_arg'], '--goal-rewriter-batch')
            self.assertNotIn('tokens_arg', c.goal_rewriter_batch_feature(repo, new))  # the module lacks TOKENS_KEY
            batch.write_text("CONFIG_KEY = 'goal_rewriter_batch'\nTOKENS_KEY = 'goal_rewriter_batch_tokens'\n"
                             "LIMITS = (2, 16)\nTOKENS_LIMIT = 16 * 262144\n", encoding='utf-8')
            self.assertEqual(c.goal_rewriter_batch_feature(repo, new), PINS['runtime_features']['goal_rewriter_batch'])
            self.assertEqual(c.goal_rewriter_batch_feature(repo, new.replace('16384', '5000'))['tokens_default'], 5000)

    def test_feature_detection_of_the_switches(self):
        serve_pool = ("CONNECTOR_POOL_GIB=range(1,32)\np.add_argument('--connector-pool-gib',type=int)\n"
                      "config.update(connector_pool_bytes=pool)\nx['cpu_bytes_to_use']=pool\n"
                      "def pinned_host_bytes(a,connector=CONNECTOR_BYTES):pass\n")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.assertIsNone(c.connector_pool_feature(repo, serve_pool))  # no connector_guard.py
            guard = repo / c.POOL_MODULE
            guard.parent.mkdir(parents=True)
            guard.write_text("POOL_KEY = 'connector_pool_bytes'\n", encoding='utf-8')
            self.assertEqual(c.connector_pool_feature(repo, serve_pool), c.POOL_FEATURE)
            self.assertIsNone(c.connector_pool_feature(repo, serve_pool.replace('range(1,32)', 'range(1,64)')))
            serve2 = "p.add_argument('--decode-fastpath2',action='store_true')\n"
            self.assertIsNone(c.decode_fastpath2_feature(repo, serve2))  # no module
            module = repo / c.FASTPATH2_MODULE
            module.write_text("CONFIG_KEY = 'decode_fastpath2'     # startup config\n", encoding='utf-8')
            self.assertEqual(c.decode_fastpath2_feature(repo, serve2)['serve_arg'], '--decode-fastpath2')
            self.assertIsNone(c.decode_fastpath2_feature(repo, 'no switch'))
            module.write_text("CONFIG_KEY = 'other'\n", encoding='utf-8')
            with self.assertRaises(RuntimeError):
                c.decode_fastpath2_feature(repo, serve2)


class GoalRewriterBatch(unittest.TestCase):
    """Runtime 5b15b2f: --goal-rewriter-batch N (plan goalb4 only; off everywhere else) and 2867301's per-call timing."""

    def test_goalb4_is_goal_plus_the_batch(self):
        goal, b4 = server('goal'), server('goalb4')
        self.assertEqual(c.launch_arguments(PINS, b4, REWRITER),
                         c.launch_arguments(PINS, goal, REWRITER) + ['--goal-rewriter-batch', '4'])
        for name in ('full', 'obs', 'rfill', 'stock', 'goal'):  # batch OFF in every other plan
            self.assertNotIn('--goal-rewriter-batch', c.launch_arguments(PINS, server(name), REWRITER)
                             if server(name)['kind'] == 'runtime' else c.stock_argv(PINS, 8000))
            self.assertIsNone(server(name)[c.BATCH_KEY])
            self.assertIsNone(server(name)[c.BATCH_TOKENS_KEY])
        self.assertNotIn('--goal-rewriter-batch', c.launch_arguments(PINS, b4, REWRITER, rewriter=False))
        plans = c.cfg(PINS)['plans']
        self.assertLessEqual(c.time_seconds(plans['goalb4']['time']), c.time_seconds(plans['goal']['time']))
        self.assertEqual(plans['goalb4']['authorization'],
                         'AUTHORIZE_SWE32_GOALB4_32TASKS_16SESSIONS_REWRITER_B4_BATCH_REUSED_V9_ENV_ONE_GH200_88MIN_NO_RETRY')
        expect = c.server_expectation(PINS, b4, {}, 'm', [])
        self.assertEqual((expect['goal_rewriter_batch'], expect['goal_rewriter_batch_tokens'],
                          expect['admission']['admission_overlap']['permits']), (4, 16384, 16))
        self.assertEqual(expect['goal_rewriter_timing'], PINS['runtime_features']['goal_rewriter_timing'])
        self.assertIsNone(c.server_expectation(PINS, goal, {}, 'm', [])['goal_rewriter_batch'])
        self.assertIsNone(c.server_expectation(PINS, server('full'), {}, 'm', [])['goal_rewriter_timing'])
        self.assertEqual(PINS['runtime_features']['goal_rewriter_batch'],
                         dict(serve_arg='--goal-rewriter-batch', config_key='goal_rewriter_batch', min=2, max=16,
                              tokens_arg='--goal-rewriter-batch-tokens', tokens_key='goal_rewriter_batch_tokens',
                              tokens_default=16384, tokens_limit=16 * 262144))
        # another cap is passed explicitly (and pinned); the sealed default is not
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['goalb4']['servers'][0]['goal_rewriter_batch_tokens'] = 8192
        self.assertEqual(c.launch_arguments(pins, server('goalb4', pins=pins), REWRITER)[-2:],
                         ['--goal-rewriter-batch-tokens', '8192'])
        self.assertEqual(c.server_expectation(pins, server('goalb4', pins=pins), {}, 'm', [])['goal_rewriter_batch_tokens'],
                         8192)
        c.validate_plan(pins, 'goalb4')
        self.assertEqual(PINS['runtime_features']['goal_rewriter_timing']['native'],
                         ['goal_rewrite_wait_ms', 'goal_rewrite_call_ms'])

    def test_the_batch_is_refused_where_serve_refuses_it(self):
        def refused(change, plan='goalb4', pattern='goal-rewriter-batch'):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, plan)
        b4 = lambda pins: pins['config']['plans']['goalb4']['servers'][0]
        for bad in (1, 17, 0, '4', 4.0):
            refused(lambda pins, bad=bad: b4(pins).update(goal_rewriter_batch=bad))
        for bad in (-1, 16 * 262144 + 1, '16384', 16384.0, None):  # the token cap (0 = no cap is allowed)
            refused(lambda pins, bad=bad: b4(pins).update(goal_rewriter_batch_tokens=bad), pattern='batch_tokens')
        refused(lambda pins: pins['config']['policies']['goal_query'].update(admission=['admission_overlap=2']))
        refused(lambda pins: pins['config']['plans']['full']['servers'][0].update(goal_rewriter_batch=4), 'full')
        refused(lambda pins: pins['config']['plans']['stock']['servers'][0].update(goal_rewriter_batch=4), 'stock')
        refused(lambda pins: pins['config']['plans']['goal']['servers'][0].update(goal_rewriter_batch_tokens=16384),
                'goal', 'batch_tokens')  # a cap without the batch
        refused(lambda pins: pins['runtime_features'].update(goal_rewriter_batch=None))
        pins = copy.deepcopy(PINS)
        pins['config']['policies']['goal_query']['admission'] = ['admission_overlap=4']  # N = K: allowed
        c.validate_plan(pins, 'goalb4')
        b4(pins)['goal_rewriter_batch_tokens'] = 0
        c.validate_plan(pins, 'goalb4')
        with self.assertRaisesRegex(RuntimeError, 'goal-rewriter-batch'):  # a pin without the switch: build refuses
            c.check_features(dict(PINS['runtime_features'], goal_rewriter_batch=None), PINS['config'])
        with self.assertRaisesRegex(RuntimeError, 'goal-rewriter-batch-tokens'):  # 5b15b2f: no cap
            c.check_features(dict(PINS['runtime_features'], goal_rewriter_batch=dict(
                serve_arg='--goal-rewriter-batch', config_key='goal_rewriter_batch', min=2, max=16)), PINS['config'])
        config = copy.deepcopy(PINS['config'])
        for name in ('goalb4', 'goalb4i') + tuple(CTX):  # (ctx24 / ctx25's goal262 runs goalb4's batch too)
            del config['plans'][name]
            config['plan_order'].remove(name)
        c.check_features(dict(PINS['runtime_features'], goal_rewriter_batch=None), config)  # no plan asks for it

    def test_the_server_check_of_the_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            def write(startup, ready):
                (out / 'startup.json').write_text(json.dumps(startup), encoding='utf-8')
                (out / 'ready.json').write_text(json.dumps(ready), encoding='utf-8')
            capped = dict(goal_rewriter_batch=4, goal_rewriter_batch_tokens=16384)
            want = dict(goal_rewriter_batch=4, goal_rewriter_batch_tokens=16384)
            write(capped, capped)
            self.assertEqual(swe_checks.check_goal_rewriter_batch(out, want), want)
            write(dict(goal_rewriter_batch=4), dict(goal_rewriter_batch=4))  # 5b15b2f: no cap pinned
            self.assertEqual(swe_checks.check_goal_rewriter_batch(out, dict(goal_rewriter_batch=4)),
                             dict(goal_rewriter_batch=4, goal_rewriter_batch_tokens=None))
            for startup, ready, wanted in ((dict(goal_rewriter_batch=4), {}, want), ({}, {}, want),
                                           (dict(goal_rewriter_batch=2), dict(goal_rewriter_batch=2), want),
                                           (capped, capped, dict(goal_rewriter_batch=None)),
                                           (capped, dict(goal_rewriter_batch=4), want),
                                           (dict(capped, goal_rewriter_batch_tokens=5000), capped, want),
                                           (dict(goal_rewriter_batch=4), dict(goal_rewriter_batch=4), want),
                                           (capped, capped, dict(goal_rewriter_batch=4))):
                write(startup, ready)
                with self.assertRaises(swe_checks.CheckFailed, msg=(startup, ready, wanted)):
                    swe_checks.check_goal_rewriter_batch(out, wanted)
            write({}, {})
            self.assertEqual(swe_checks.check_goal_rewriter_batch(out, {}),
                             dict(goal_rewriter_batch=None, goal_rewriter_batch_tokens=None))
            write(dict(goal_rewriter_batch_tokens=16384), {})  # a cap without a batch
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_goal_rewriter_batch(out, {})

    @staticmethod
    def exchange(wait, call, **extra):
        return dict(dict(request='{}', response='{"result": {}}', error=None, wait_ms=wait, start_ms=0.0, call_ms=call,
                         input_tokens=900, output_tokens=40, generation_latency_ms=round(call * 0.9, 3)), **extra)

    def case(self, rewrites_by_row, natives, batch=None):
        timing = PINS['runtime_features']['goal_rewriter_timing']
        records = [('goals.jsonl', i, dict(request_id='r%d' % i, rewrites=r)) for i, r in enumerate(rewrites_by_row)]
        pairs = [(dict(request_id='r%d' % i, session_index=i, turn=1), [records[i]]) for i in range(len(records))]
        return goal_text.verify_timing(records, pairs, timing, batch, natives.get)

    def test_the_rewriter_timing_and_the_native_totals(self):
        rows = [[self.exchange(12.5, 40.25), self.exchange(0.125, 39.0)], [], [self.exchange(None, 41.0)]]
        natives = {'r0': dict(goal_construct_ms=95.0, goal_rewrite_wait_ms=round(12.5 + 0.125, 3),
                              goal_rewrite_call_ms=round(40.25 + 39.0, 3)),
                   'r1': dict(goal_construct_ms=1.0),
                   'r2': dict(goal_construct_ms=60.0, goal_rewrite_wait_ms=None, goal_rewrite_call_ms=41.0)}
        summary, reasons, findings = self.case(rows, natives)
        self.assertEqual((reasons, summary['calls'], summary['natives_checked'], summary['nulls']), ([], 3, 3, 1))
        self.assertEqual(findings, [dict(finding='GOAL_REWRITE_TIMING_NULL', times=1)])
        self.assertEqual(goal_text.rewrite_totals(rows[0]), dict(goal_rewrite_wait_ms=12.625, goal_rewrite_call_ms=79.25))
        bad_natives = dict(natives, r0=dict(natives['r0'], goal_rewrite_wait_ms=13.0))
        self.assertTrue(self.case(rows, bad_natives)[1])  # a total that is not the entries' sum
        self.assertTrue(self.case(rows, dict(natives, r1=dict(goal_construct_ms=1.0, goal_rewrite_wait_ms=0.0,
                                                              goal_rewrite_call_ms=0.0)))[1])  # totals without a call
        self.assertTrue(self.case(rows, dict(natives, r0=dict(goal_construct_ms=95.0)))[1])  # totals missing
        self.assertFalse(self.case(rows, dict(natives, r0=dict(goal_async_ms=3.0)))[1])  # not on a Goal thread
        missing = [[dict(rows[0][0])]]
        missing[0][0].pop('generation_latency_ms')
        self.assertTrue(self.case(missing, {})[1])
        self.assertTrue(self.case([[self.exchange(-1.0, 40.0)]], {})[1])
        self.assertTrue(self.case([[self.exchange(1.0, 40.0, batch_id=0, batch_size=1)]], {})[1])  # unbatched server
        batched = [[self.exchange(1.0, 40.0, batch_id=0, batch_size=2)], [self.exchange(2.0, 40.0, batch_id=0, batch_size=2)],
                   [self.exchange(0.5, 38.0, batch_id=1, batch_size=1)]]
        summary, reasons, _ = self.case(batched, {}, batch=4)
        self.assertEqual((reasons, summary['batches'], summary['batch_sizes']), ([], 2, {'1': 1, '2': 2}))
        self.assertTrue(self.case(batched, {}, batch=1)[1])  # a size above N
        self.assertTrue(self.case([[self.exchange(1.0, 40.0)]], {}, batch=4)[1])  # the batch fields missing
        crowded = [[self.exchange(1.0, 40.0, batch_id=7, batch_size=1)], [self.exchange(1.0, 40.0, batch_id=7, batch_size=1)]]
        self.assertTrue(self.case(crowded, {}, batch=4)[1])  # more calls than the batch's size


class HeldAcks(unittest.TestCase):
    """--deferred-completion-io (completion-offload ad66aca, 120bc3c): a completion's Cancel whose request's files are
    still being written is acknowledged later by the engine loop (dispatcher Held -> lifecycle.tick), outside its
    dispatch and outside any resume piece. Job 3428631 (goalb4 r2) failed goal-async on exactly one such ack; the
    excerpt fixtures/held-cancel-ack-job3428631.log is its server log around call 8004923403447454992 (verbatim, the
    /metrics access lines left out)."""
    CALL = 8004923403447454992
    TEXT = (ROOT / 'fixtures' / 'held-cancel-ack-job3428631.log').read_text(encoding='utf-8')

    def test_the_held_cancel_ack_of_job_3428631(self):
        import measure
        events, unreadable = measure.admission_events(self.TEXT)
        self.assertEqual(unreadable, 0)
        permits, timeouts = measure.log_permits(events)
        self.assertEqual((permits, timeouts), (16, [2100]))
        engine = measure.engine_pieces(events, several=True, admission_timeouts=timeouts)
        self.assertEqual((engine['errors'], engine['overlaps']), ([], []))
        self.assertEqual(sorted(engine['held']), [self.CALL])
        self.assertIsNotNone(engine['held'][self.CALL]['ack_pos'])
        self.assertEqual(engine['held'][self.CALL]['dispatch']['ack'], 'held')
        self.assertNotIn(self.CALL, engine['parked'])  # a held Cancel is never a parked admission
        resumed = [call for call, entry in engine['parked'].items() if entry['resume'] is not None]
        self.assertEqual(sorted(resumed), [8812813660767145013, 11539188251730331762])  # the two Goals: resumed
        self.assertEqual(engine['calls'][10111496416222790387][0]['ack'], 'immediate')  # an ordinary Cancel
        # the job's parser (fc1087b5) read the same lines as a parked call never resumed
        old = measure.engine_pieces(events, several=True, admission_timeouts=timeouts + [60])
        self.assertEqual(old['errors'], [
            'ack of call %d at event %d is neither right after its dispatch nor inside a piece'
            % (self.CALL, engine['held'][self.CALL]['ack_pos']), 'parked call %d was never resumed' % self.CALL])
        # an admission ADD (its 2100-s wait) acknowledged that way stays an error
        text = self.TEXT.replace('{"call_id": %d, "phase": "frontend_command_sent", "pid": 82536, "timeout_seconds": 60}'
                                 % self.CALL, '{"call_id": %d, "phase": "frontend_command_sent", "pid": 82536, '
                                 '"timeout_seconds": 2100}' % self.CALL)
        self.assertNotEqual(text, self.TEXT)
        events = measure.admission_events(text)[0]
        self.assertTrue(measure.engine_pieces(events, several=True, admission_timeouts=[2100])['errors'])
        found = measure.admissions(self.TEXT)
        self.assertEqual(sorted(found['engine']['held']), [self.CALL])

    def test_goal_async_accepts_held_acks_only_with_the_switch(self):
        import smoke_checks as sc
        from unittest import mock
        engine = dict(errors=[], overlaps=[], parked={}, pieces=[], unattributed=[],
                      held={self.CALL: dict(dispatch={}, ack_pos=7)})
        found = dict(requests={}, engine=engine, permits=1, admission_timeouts=[300])
        expect = dict(admission=None, parking_policies=[], deferred_completion_io=True)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sc.measure, 'admissions', return_value=found):
            log = Path(tmp) / 'server.log'
            log.write_text('', encoding='utf-8')
            self.assertEqual(sc.check_goal_async(Path(tmp), log, [], 'full', expect)['held_acks'], 1)
            with self.assertRaisesRegex(sc.CheckFailed, 'without --deferred-completion-io'):
                sc.check_goal_async(Path(tmp), log, [], 'full', dict(expect, deferred_completion_io=None))


class StockAndAgent(unittest.TestCase):
    def test_stock_argv_default_mode(self):
        argv = c.stock_argv(PINS, 8123)
        self.assertNotIn('--enforce-eager', argv)
        self.assertNotIn('--no-async-scheduling', argv)
        value = lambda flag: argv[argv.index(flag) + 1]
        self.assertEqual((value('--served-model-name'), value('--max-num-seqs'), value('--max-model-len'),
                          value('--kv-cache-memory-bytes'), value('--port')),
                         ('selective-prefix', '16', '131072', str(32 * 2 ** 30), '8123'))
        self.assertEqual(argv.count('--language-model-only'), 1)
        self.assertEqual(c.stock_expected(PINS, 8123)['enforce_eager'], False)
        self.assertIsNone(c.stock_expected(PINS, 8123)['async_scheduling'])  # no flag: vLLM's own default (ON)
        self.assertNotIn('--connector-pool-gib', argv)

    def test_stock_async_is_one_switch(self):
        pins = copy.deepcopy(PINS)
        pins['config']['stock']['async_scheduling'] = False
        self.assertIn('--no-async-scheduling', c.stock_argv(pins, 8123))
        self.assertIs(c.stock_expected(pins, 8123)['async_scheduling'], False)
        pins['config']['stock'].update(mode='eager', async_scheduling=True)
        with self.assertRaises(Exception):
            c.stock_argv(pins, 8123)
        pins['config']['stock'].update(async_scheduling='yes')
        with self.assertRaises(Exception):
            c.stock_argv(pins, 8123)

    def test_stock_checks_follow_the_switch(self):
        import stock
        lines = ('INFO [vllm.py:1006] Asynchronous scheduling is enabled.\n' * 2 + 'enforce_eager=False\n')
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'server.log'
            log.write_text(lines)
            on = stock.check_server(tmp, log, dict(PINS['config']['stock']), {})
            off = stock.check_server(tmp, log, dict(PINS['config']['stock'], async_scheduling=False), {})
        status = lambda report: {r['check']: r['status'] for r in report['checks']}['stock-async-scheduling']
        self.assertEqual((status(on), status(off)), ('PASS', 'FAIL'))

    def test_model_override_runtime_and_stock(self):
        text = c.model_override_text(PINS, server('full', 0), 23004)
        for line in ('    api_base: "http://127.0.0.1:23004/v1"', '    max_tokens: 4096', '    timeout: 3600',
                     '    max_retries: 0', '    num_retries: 0', '  step_limit: 120', '  cost_limit: 0',
                     '  model_class: litellm_textbased', '  environment_class: docker'):
            self.assertIn(line, text.split('\n'))
        self.assertNotIn('stop:', text)
        stock_text = c.model_override_text(PINS, server('stock', 0), 23004)
        self.assertIn('    stop: ["</mswea_bash_command>"]', stock_text.split('\n'))
        self.assertIn('include_stop_str_in_output', stock_text)
        self.assertIn('"enable_thinking": false', stock_text)
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['goal']['servers'][0]['step_limit'] = 25
        self.assertIn('  step_limit: 25', c.model_override_text(pins, server('goal', 0, pins), 23004).split('\n'))
        try:
            import yaml
        except ImportError:
            return
        value = yaml.safe_load(stock_text)
        self.assertEqual(value['model']['model_kwargs']['extra_body'], dict(
            chat_template_kwargs=dict(enable_thinking=False), include_stop_str_in_output=True))
        self.assertEqual(yaml.safe_load(text)['agent'], dict(step_limit=120, cost_limit=0))

    def test_pool_spec_round_trip(self):
        spec, value = c.pool_spec(PINS, server('goal'), 23005, 1800000000)
        self.assertRegex(spec, r'^[A-Za-z0-9_-]+$')
        self.assertEqual(c.decode_spec(spec), value)
        self.assertEqual((value['schema'], value['max_running'], value['step_limit'], value['mini_retry_attempts'],
                          value['close']), ('swe32-pool-spec-v1', 16, 120, 1, True))
        self.assertFalse(c.pool_spec(PINS, server('stock', 0), 23005, 1)[1]['close'])
        self.assertEqual((value['prompt_file'], value['prompt_sha256']),
                         ('swebench_prompt_override.yaml', PINS['config']['agent']['prompt']['sha256']))

    def test_the_prompt_is_one_entry_checked_against_the_sealed_commit(self):
        import build
        agent = PINS['config']['agent']
        self.assertEqual(agent['prompt']['source'], 'runtime:examples/swebench/upstream/swebench_prompt_override.yaml')
        self.assertEqual(PINS['upstream']['prompt'], agent['prompt']['sha256'])
        sealed = {'examples/swebench/upstream/swebench_prompt_override.yaml': agent['prompt']['sha256']}
        config = copy.deepcopy(PINS['config'])
        self.assertEqual(build.upstream_pins(config, sealed)['prompt_file'], 'swebench_prompt_override.yaml')
        with self.assertRaises(ValueError):  # the sealed commit's prompt differs
            build.upstream_pins(config, {'examples/swebench/upstream/swebench_prompt_override.yaml': '0' * 64})
        config['agent']['prompt']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):  # the file in tbench/upstream differs from the entry
            build.upstream_pins(config, sealed)
        config = copy.deepcopy(PINS['config'])
        config['agent']['prompt']['source'] = 'somewhere'
        with self.assertRaises(ValueError):
            build.upstream_pins(config, sealed)

    def test_run_id_ports_sockets_and_names_never_collide_with_swe16_or_swe10(self):
        rid = c.run_id('3351234', 'r1')
        self.assertEqual(rid, 'swe32-3351234-r1')
        port = tbench_link.tunnel_port('3351234', 0, PINS['tbench'])
        self.assertTrue(23000 <= port < 23800)  # swe16 used 21000+, swe10 22000+
        pins = copy.deepcopy(PINS)
        pins['tbench']['port_stride'] = 0
        with self.assertRaises(RuntimeError):  # a plan with more servers than ports per job is refused
            c.validate_plan(pins, 'full')
        path = tbench_link.socket_path(PINS['tbench'], rid, 'stock')
        self.assertLessEqual(len(path.encode()), 100)
        self.assertTrue(path.startswith(PINS['tbench']['socket_dir'] + '/'))
        self.assertEqual((c.STAGE_PREFIX, c.CLAIM_PREFIX, c.PHRASE_PREFIX, c.LAYOUT), ('swe32-', '.swe32-',
                                                                                       'AUTHORIZE_SWE32_', 'swe32'))
        self.assertEqual(PINS['tbench']['root'], '/home/ubuntu/swe32')
        self.assertTrue(all(PINS['tbench'][k].startswith('/home/ubuntu/swe32/') for k in (
            'bin', 'lock_dir', 'mswea_config', 'runs_dir', 'socket_dir')))
        self.assertEqual(job.PASS, 'SWE32_PASS')
        self.assertEqual((job.CORE_ARCHIVE, job.BULK_ARCHIVE), ('swe32-core.tar.xz', 'swe32-bulk.tar.xz'))
        self.assertTrue(c.claim_glob(PINS, 'r1', 'goal').startswith('.swe32-' + PINS['commit']))


class FakeRun:
    def __init__(self, stdout):
        self.returncode, self.stdout, self.stderr = 0, stdout, ''


def interactive_pins():
    """PINS as sealed: every plan has its ghx4-interactive variant (<plan>i; 2x billing, one submitted job per user)."""
    return copy.deepcopy(PINS)


class Gates(unittest.TestCase):
    def test_queue_gate_batch_refuses_another_swe_job_unless_named(self):
        plan = c.cfg(PINS)['plans']['goal']
        gate = lambda out, after, name: c.queue_gate(Path(tmp), PINS, plan, dict(plan='goal', after_job=after), name,
                                                     runner=lambda *a, **k: FakeRun(out), user='ycao11')
        with tempfile.TemporaryDirectory() as tmp:
            for i, other in enumerate(('777|RUNNING|ghx4|swe32-full|(null)', '777|PENDING|ghx4|swe10-goal|(null)',
                                       '777|RUNNING|ghx4|swe16-stock|(null)', '777|PENDING|ghx4|tb16-dryo|(null)',
                                       '777|RUNNING|ghx4-interactive|gswe32-goal|(null)',
                                       '777|PENDING|ghx4|gem0-x|(null)')):
                out = other + '\n888|PENDING|ghx4|other-job|(null)\n999|RUNNING|ghx4-interactive|sctx2-x|(null)\n'
                with self.assertRaises(RuntimeError, msg=other):  # tbench must not see two agent runs at once
                    gate(out, None, 'q%d.json' % i)
                value = gate(out, '777', 'p%d.json' % i)
                self.assertEqual((value['status'], value['blocking']), ('QUEUE_GATE_PASS', []))
            with self.assertRaises(RuntimeError):  # -AfterJob names one; a second tbench job outside its chain blocks
                gate('777|RUNNING|ghx4|swe32-full|(null)\n666|PENDING|ghx4|swe10-goal|(null)\n', '777', 'r.json')
            # a chain queued at once: 666 waits for 777 (afterany), so a job after 666 waits for both
            chain = ('777|RUNNING|ghx4|swe32-full|(null)\n666|PENDING|ghx4|swe32-goalb4|afterany:777(unfulfilled)\n'
                     '555|PENDING|ghx4|tb16-x|afterany:666(unfulfilled)\n')
            value = gate(chain, '555', 'c1.json')
            self.assertEqual((value['status'], value['chain']), ('QUEUE_GATE_PASS', ['555', '666', '777']))
            with self.assertRaises(RuntimeError):  # after the middle job: 555 would start beside it
                gate(chain, '666', 'c2.json')
            with self.assertRaises(RuntimeError):  # 'after:' (start) is no end dependency: 666 may still run
                gate('777|RUNNING|ghx4|swe32-full|(null)\n666|PENDING|ghx4|swe32-goalb4|after:777(unfulfilled)\n',
                     '666', 'c3.json')
            with self.assertRaisesRegex(RuntimeError, 'names no job of yours'):  # an ended or mistyped job
                gate('777|RUNNING|ghx4|swe32-full|(null)\n', '778', 'c4.json')
            value = gate('888|PENDING|ghx4|other-job|(null)\n', None, 's.json')
            self.assertEqual(value['status'], 'QUEUE_GATE_PASS')  # other work of the user does not block
            value = gate('888|PENDING|ghx4|other-job|(null)\n', '888', 's2.json')
            self.assertEqual(value['status'], 'QUEUE_GATE_PASS')  # (a dependency on other work is the user's choice)
            self.assertEqual(PINS['slurm']['tbench_job_prefixes'],
                             ['swe32-', 'swe10-', 'swe16-', 'gswe32-', 'tb16-', 'gem0-'])
            self.assertEqual(c.queue_command('ghx4', 'ycao11')[-1], '%i|%T|%P|%j|%E|%r')  # %r: 2026-11-11

            class Failing(FakeRun):
                def __init__(self):
                    super().__init__('')
                    self.returncode, self.stderr = 1, 'slurm down'
            with self.assertRaises(RuntimeError):
                c.queue_gate(Path(tmp), PINS, plan, dict(plan='goal', after_job=None), 'qfail.json',
                             runner=lambda *a, **k: Failing(), user='ycao11')

    def test_queue_gate_interactive_refuses_a_queued_job(self):
        pins = interactive_pins()
        with tempfile.TemporaryDirectory() as tmp:
            plan = c.cfg(pins)['plans']['fulli']
            options = dict(plan='fulli', after_job=None)

            def gate(out, name):
                return c.queue_gate(Path(tmp), pins, plan, options, name, runner=lambda *a, **k: FakeRun(out),
                                    user='ycao11')
            with self.assertRaises(RuntimeError):  # one submitted ghx4-interactive job per user
                gate('123|RUNNING|ghx4-interactive|other|(null)\n', 'q1.json')
            with self.assertRaises(RuntimeError):  # a queued tbench job: an interactive plan goes first in a chain
                gate('124|PENDING|ghx4|swe32-goalb4|(null)\n', 'q3.json')
            value = gate('125|PENDING|ghx4|other-job|(null)\n', 'q4.json')  # other batch work does not block
            self.assertEqual(value['status'], 'QUEUE_GATE_PASS')
            value = gate('', 'q2.json')
            self.assertEqual(value['status'], 'QUEUE_GATE_PASS')
            self.assertEqual(value['partition'], 'ghx4-interactive')
            self.assertEqual(len(value['calls']), 2)  # both partitions are listed

    def test_sbatch_mail_partition_and_dependency(self):
        for name, (row, time, expected) in DECIDED.items():
            for plan, partition in ((name, 'ghx4'), (name + 'i', 'ghx4-interactive')):
                options = dict(plan=plan, mail_user='cao58@illinois.edu', run_label='r1', after_job=None)
                args = c.sbatch_args(Path('/u/ycao11/swe32-x'), PINS, options, '/u/ycao11/swe32-x/slurm-%j.out')
                for item in ('--mail-type=BEGIN,END,FAIL', '--mail-user=cao58@illinois.edu', '--partition=' + partition,
                             '--time=' + time, '--job-name=swe32-' + plan, '--gpus=1', '--nodes=1', '--mem=113560M'):
                    self.assertIn(item, args, plan)
                self.assertFalse([a for a in args if a.startswith('--dependency')])
        args = c.sbatch_args(Path('/u/ycao11/swe32-x'), PINS, dict(options, after_job='777'), 'o')
        self.assertIn('--dependency=afterany:777', args)

    def test_tbench_gate_requires_the_deployed_manifest(self):
        class Link:
            def __init__(self, manifest, status='TBENCH_PREFLIGHT_PASS'):
                self.manifest, self.status, self.seen = manifest, status, None

            def preflight(self, instances, statements, agent):
                self.seen = instances
                return dict(status=self.status, preflight=dict(manifest_sha256=self.manifest, failures=[]))
        options = dict(plan='full')
        with tempfile.TemporaryDirectory() as tmp:
            link = Link(PINS['tbench_half']['manifest_sha256'])
            c.tbench_gate(Path(tmp), PINS, options, link)
            self.assertEqual(link.seen, c.instance_ids(PINS))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                c.tbench_gate(Path(tmp), PINS, options, Link('0' * 64))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                c.tbench_gate(Path(tmp), PINS, options, Link(PINS['tbench_half']['manifest_sha256'], 'TBENCH_PREFLIGHT_FAIL'))

    def test_features_of_older_runtimes_are_refused(self):
        features = copy.deepcopy(PINS['runtime_features'])
        c.check_features(features, c.cfg(PINS))
        for name, value in (('context_limits', None), ('session_pool', dict(max_tasks=24, close_route='/v1/sessions/close')),
                            ('admission_flags', None), ('max_sessions', 8), ('decode_fastpath', None),
                            ('language_model_only', None), ('connector_pool', None), ('decode_fastpath2', None),
                            ('decode_fastpath3', None), ('deferred_completion_io', None), ('goal_rewriter_batch', None)):
            broken = dict(features, **{name: value})
            with self.assertRaises(RuntimeError, msg=name):
                c.check_features(broken, c.cfg(PINS))
        config = copy.deepcopy(c.cfg(PINS))
        config['server']['decode_fastpath'] = False  # e877051 serves without the switch when it is off
        config['server']['decode_fastpath2'] = False
        config['server']['decode_fastpath3'] = False
        config['server']['deferred_completion_io'] = False
        c.check_features(dict(features, decode_fastpath=None, decode_fastpath2=None, decode_fastpath3=None,
                              deferred_completion_io=None), config)

    def test_the_sealed_runtime(self):
        features = PINS['runtime_features']
        self.assertEqual(features['decode_fastpath'], dict(serve_arg='--decode-fastpath', config_key='decode_fastpath'))
        self.assertEqual(features['decode_fastpath2']['serve_arg'], '--decode-fastpath2')
        self.assertEqual(features['decode_fastpath3']['serve_arg'], '--decode-fastpath3')
        self.assertEqual(features['deferred_completion_io'], c.DC_FEATURE)
        self.assertEqual(features['goal_rewriter_batch']['tokens_default'], 16384)
        self.assertIn(32, features['kv_cache_gib_choices'])  # 120bc3c adds 25 / 28 for Gemma; Qwen runs 32
        self.assertEqual(features['connector_pool'], c.POOL_FEATURE)
        self.assertEqual(features['language_model_only'], c.LMO_FEATURE)
        self.assertEqual((features['goal_text'] or {}).get('files'), ['goals.jsonl'])
        self.assertIn(131072, features['context_limits']['max_model_len_choices'])
        self.assertEqual(features['session_pool']['max_tasks'], 256)  # 2026-11-17: integrate-1117 06c1e90 (was 64)
        self.assertEqual(features['max_sessions'], 32)  # 2026-11-16: sessions-32-1116 (a2f9e36)

    def test_the_release_receipt_is_pinned(self):
        """The pinned local release-check receipt of the pin (RECEIPT_PIN: 4f5f4e7, D:\\Remote-RA-temp\\scratch\\
        docs-main-1023\\release-receipt-4f5f4e7.json): exactly the known values; any other, or none, is refused (light
        login gates: setup, the job and the verifier check it; no runtime test suite on the login node)."""
        import inspect
        value = c.check_receipt(PINS)
        self.assertEqual(value, dict(RECEIPT_PIN, pinned='RELEASE_RECEIPT_PINNED'))
        # the earlier pins' receipts stay known (re-pin rehearsals, frozen copies' jobs)
        self.assertEqual(c.RELEASE_RECEIPTS['120bc3c0a6db281eeba21f541957c3ae6941542d'],
                         {k: v for k, v in RECEIPT_120BC3C.items() if k != 'commit'})
        for key, wrong in (('status', 'FAIL'), ('tests_run', 1383), ('skipped', 0), ('result_line_sha256', '0' * 64),
                           ('log_sha256', '1' * 64), ('commit', '0' * 40)):
            bad = copy.deepcopy(PINS)
            bad['release_receipt'][key] = wrong
            with self.assertRaisesRegex(RuntimeError, 'release-check receipt', msg=key):
                c.check_receipt(bad)
        bad = copy.deepcopy(PINS)
        bad.pop('release_receipt')
        with self.assertRaisesRegex(RuntimeError, 'release-check receipt'):
            c.check_receipt(bad)
        other = dict(copy.deepcopy(PINS), commit='5b15b2ff2d7497717dffdb4a77641781a8c1090e')
        with self.assertRaisesRegex(RuntimeError, 'no pinned release-check receipt'):
            c.check_receipt(other)
        # the light gates: no suite, no check_git_release on the login node; setup and the job check the receipt
        source = inspect.getsource(c.qualify)
        self.assertNotIn("'unittest', 'discover'", source)
        self.assertNotIn('check_git_release.py', source)
        self.assertIn('check_receipt(pins)', inspect.getsource(c.setup))
        self.assertIn('check_receipt(pins)', source)
        self.assertIn('check_receipt(pins)', inspect.getsource(job.preflight))
        import build
        self.assertEqual(build.receipt_of(build.DEFAULT_RECEIPT, PINS['commit'])['result_line_sha256'],
                         RECEIPT_PIN['result_line_sha256'])

    def test_phrases_name_exactly_their_partition(self):
        for name in BATCH:
            for plan, wrong in ((name, '_INTERACTIVE_'), (name + 'i', '_BATCH_')):
                self.assertNotIn(wrong, c.cfg(PINS)['plans'][plan]['authorization'])
                pins = copy.deepcopy(PINS)
                entry = pins['config']['plans'][plan]
                other = 'ghx4-interactive' if entry['partition'] == 'ghx4' else 'ghx4'
                entry['partition'] = other  # the plan moved to the other partition with its old phrase: refused
                with self.assertRaisesRegex(RuntimeError, 'phrase', msg=plan):
                    c.validate_plan(pins, plan)
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['fulli']['servers'][0]['agent_cap_seconds'] = 1800  # a variant differs from its plan
        with self.assertRaisesRegex(RuntimeError, 'variant_of'):
            c.validate_plan(pins, 'fulli')
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['goali']['time'] = '02:01:00'
        pins['config']['plans']['goali']['authorization'] = pins['config']['plans']['goali']['authorization'].replace(
            '_102MIN_', '_121MIN_')
        with self.assertRaisesRegex(RuntimeError, 'maximum'):  # above ghx4-interactive's 02:00:00
            c.validate_plan(pins, 'goali')
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['full']['role'] = 'interactive'
        with self.assertRaisesRegex(RuntimeError, 'role'):
            c.validate_plan(pins, 'full')

    def test_feature_detection_on_synthetic_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            overlay = repo / 'source-overlay' / 'remote_ra_live_v2'
            overlay.mkdir(parents=True)
            (repo / 'source-overlay' / 'remote_ra_http').mkdir()
            serve = "parser.add_argument('--decode-fastpath',action='store_true')\n"
            (repo / 'serve.py').write_text(serve)
            (overlay / 'decode_fastpath.py').write_text("CONFIG_KEY = 'decode_fastpath'\nclass ClaimTable(dict):\n    pass\n"
                                                        "def attach_worker(owner, bridge, qwen):\n    pass\n")
            evidence = repo / 'source-overlay' / 'remote_ra_http' / 'evidence.py'
            evidence.write_text("value.update(decode_fastpath=True)\nwith (self.output/'runtime.jsonl').open('ab'):pass\n")
            self.assertEqual(c.decode_fastpath_feature(repo, serve), dict(serve_arg='--decode-fastpath',
                                                                          config_key='decode_fastpath'))
            self.assertIsNone(c.decode_fastpath_feature(repo, 'no switch'))
            self.assertIsNone(c.goal_text_feature(repo))
            (repo / 'source-overlay' / 'vllm').mkdir()
            (repo / 'source-overlay' / 'vllm' / 'bridge.py').write_text('f"{tag.goal_hash}.ranking.json"\n')
            self.assertIsNone(c.goal_text_feature(repo))  # 00e173d's only goal-ish json: an f-string, not a file name
            (overlay / 'test_goal.py').write_text("NAME = 'goal-text.jsonl'\n")
            self.assertIsNone(c.goal_text_feature(repo))  # tests do not count
            (overlay / 'goal_text.py').write_text("GOAL_TEXT_FILE = 'goal-text.jsonl'\n")
            self.assertEqual(c.goal_text_feature(repo), dict(files=['goal-text.jsonl'],
                                                             modules=['source-overlay/remote_ra_live_v2/goal_text.py']))


class Budget(unittest.TestCase):
    def budget(self, plan, now, pins=PINS):
        clock = [0.0]
        value = job.Budget(0.0, pins, plan, clock=lambda: clock[0])
        clock[0] = now
        return value

    def test_every_plan_gets_its_cap_without_a_wait(self):
        # the server starts after the 60-s job checks (fits_server); with server start 360 + tunnel 60 the agents
        # start at 480 s at the latest without a wait and get their full cap
        for name, (row, time, expected) in DECIDED.items():
            self.assertTrue(self.budget(name, 60).fits_server(0), name)
            budget = self.budget(name, 480)
            self.assertEqual(budget.later(0), 0)
            self.assertEqual(budget.agent_timeout(0), row[4], name)

    def test_agent_cap_follows_the_deadline(self):
        # full, 01:01:00: deadline 3,660 - 480 - 300 = 2,880 s; agents get min(1,920, deadline - now - 180 - 300)
        self.assertEqual(self.budget('full', 0).deadline, 2880.0)
        self.assertEqual(self.budget('full', 600).agent_timeout(0), 2880 - 600 - 480)  # after the full 600-s wait
        self.assertEqual(self.budget('full', 900).agent_timeout(0), 1500)  # exactly the minimum
        self.assertIsNone(self.budget('full', 901).agent_timeout(0))  # below 1,500 s: refused before any agent
        # stock, 00:51:00: its own finish (180 s) and checks (120 s)
        self.assertEqual(self.budget('stock', 0).deadline, 3060.0 - 180 - 300)
        self.assertEqual(self.budget('stock', 600).agent_timeout(0), 2580 - 600 - 180 - 120)
        self.assertEqual(self.budget('stock', 0).ready_deadline(0), 2580 - 180 - 120 - 1400 - 60)
        # goal, 01:42:00: after the full wait it still gets 4,260 s of its 4,380-s cap; goalb4, 01:28:00: 3,420 of 3,540
        self.assertEqual(self.budget('goal', 600).agent_timeout(0), 6120 - 780 - 600 - 480)
        self.assertEqual(self.budget('goalb4', 600).agent_timeout(0), 5280 - 780 - 600 - 480)
        self.assertEqual(self.budget('goalb4i', 600).agent_timeout(0), 5280 - 780 - 600 - 480)
        self.assertEqual(self.budget('full', 0).ready_deadline(0), 2880 - 180 - 300 - 1500 - 60)

    def test_optional_servers_run_while_time_remains(self):
        # perf-swe10-m110's optional servers (no swe32 plan has one): a synthetic plan full + an optional stock
        pins = copy.deepcopy(PINS)
        plan = copy.deepcopy(pins['config']['plans']['full'])
        plan['time'] = '02:00:00'
        plan['servers'].append(dict(pins['config']['plans']['stock']['servers'][0], optional=True))
        pins['config']['plans']['pair'] = plan
        early = self.budget('pair', 480, pins)
        self.assertEqual(early.later(0), 0)  # the optional server reserves nothing in full's budget
        self.assertEqual(early.agent_timeout(0), 1920)
        deadline = 7200 - 480 - 300
        late = self.budget('pair', deadline - 2000, pins)  # stock needs 360 + 60 + 1400 + 180 + 120 = 2,120 s
        self.assertFalse(late.fits_server(1))
        self.assertTrue(self.budget('pair', deadline - 2200, pins).fits_server(1))

    def test_eval_deadline(self):
        for name in PLANS:  # the job start + the plan's eval_wait_seconds (600; smoke 900: PLAN 3.4, after a SWE job;
            # smoket 1,800: last in the tbench chain, after the SWE-Gemma smoke's 96 evaluations, D-E)
            # (2026-11-12: the 64-task plans wait up to 1,200 s for a previous 64-instance evaluation)
            self.assertEqual(job.Budget(100.0, PINS, name, clock=lambda: 100.0).eval_deadline,
                             dict(smoke=1000.0, smoket=1900.0, obs64=1300.0, obs64i=1300.0, goal64=1300.0,
                                  full64=1300.0, recent64=1300.0, akv64=1300.0, rfill64=1300.0, rfill64i=1300.0,
                                  stock64t=1300.0, ledger64=1300.0, **dict.fromkeys(PB64 + PBSTOCK, 1300.0),
                                  **dict.fromkeys(P150, 2860.0), **dict.fromkeys(GATE2, 100.0))
                             .get(name, 700.0), name)  # (11-14: ledger64; 11-16: the phase-B 64-task plans;
            # 11-17: the 150-task plans wait up to 2,760 s for the previous 150-instance evaluation; gate 2 waits for
            # nothing: its probes never take the tbench lock)


class Outcomes(unittest.TestCase):
    CHECKS = dict.fromkeys(job.END_CHECKS, True)

    def classify(self, statuses, labels=('full',), abort=None, egress='EGRESS_PASS'):
        return job.classify([dict(label=l, status=s) for l, s in zip(labels, statuses)], self.CHECKS, abort, None,
                            dict(status=egress), list(labels))[0]

    def test_outcomes(self):
        self.assertEqual(self.classify(['PASS']), job.PASS)
        self.assertEqual(self.classify(['PASS_TRUNCATED']), job.PARTIAL)
        self.assertEqual(self.classify(['REFUSED_TBENCH_BUSY']), job.REFUSED)
        self.assertEqual(self.classify(['REFUSED_TIME']), job.REFUSED)
        self.assertEqual(self.classify(['FAIL']), job.FAIL)
        self.assertEqual(self.classify(['PASS'], egress='EGRESS_FAIL'), job.FAIL)
        self.assertEqual(self.classify(['PASS', 'PASS'], ('full', 'obs')), job.PASS)
        self.assertEqual(self.classify(['PASS', 'SKIPPED_TIME_BUDGET'], ('full', 'obs')), job.PARTIAL)
        self.assertEqual(self.classify(['REFUSED_TBENCH_BUSY', 'NOT_RUN_AFTER_REFUSED_TBENCH_BUSY'], ('full', 'obs')),
                         job.REFUSED)
        self.assertEqual(self.classify(['PASS'], abort='x'), job.FAIL)

    def test_optional_skips_are_findings_not_partial_runs(self):
        labels, optional = ('goal', 'full', 'obs', 'rfill', 'stock'), ('obs', 'rfill', 'stock')
        run = lambda statuses: job.classify([dict(label=l, status=s) for l, s in zip(labels, statuses)], self.CHECKS,
                                            None, None, dict(status='EGRESS_PASS'), list(labels), optional)[0]
        self.assertEqual(run(['PASS'] * 5), job.PASS)
        self.assertEqual(run(['PASS', 'PASS', 'SKIPPED_TIME_BUDGET', 'PASS', 'PASS']), job.PASS)
        self.assertEqual(run(['PASS', 'PASS', 'SKIPPED_TIME_BUDGET', 'SKIPPED_TIME_BUDGET', 'REFUSED_TIME']), job.PASS)
        self.assertEqual(run(['PASS', 'SKIPPED_TIME_BUDGET', 'PASS', 'PASS', 'PASS']), job.PARTIAL)  # full is required
        self.assertEqual(run(['PASS', 'REFUSED_TIME', 'PASS', 'PASS', 'PASS']), job.REFUSED)
        self.assertEqual(run(['PASS', 'PASS', 'FAIL', 'NOT_RUN_AFTER_FAIL', 'NOT_RUN_AFTER_FAIL']), job.FAIL)
        self.assertEqual(run(['PASS', 'PASS', 'PASS_TRUNCATED', 'PASS', 'PASS']), job.PARTIAL)
        self.assertTrue(job.optional_skip(dict(label='obs', status='SKIPPED_TIME_BUDGET', optional=True)))
        self.assertFalse(job.optional_skip(dict(label='full', status='SKIPPED_TIME_BUDGET', optional=False)))
        self.assertEqual(job.result_row(dict(label='obs', optional=True, status='SKIPPED_TIME_BUDGET'))['optional'], True)

    def test_server_status_and_pool_outcome(self):
        self.assertEqual(job.server_status(dict(agent=dict(outcome='completed'), errors=[])), 'PASS')
        self.assertEqual(job.server_status(dict(agent=dict(outcome='agent_timeout'), errors=[])), 'PASS_TRUNCATED')
        self.assertEqual(job.server_status(dict(agent=dict(outcome='refused_tbench_busy'), errors=[])),
                         'REFUSED_TBENCH_BUSY')
        self.assertEqual(job.server_status(dict(agent=dict(outcome='completed'), errors=[], failed=['ready'])), 'FAIL')
        done = dict(pool_exit=0, instances=dict(a=dict(trajectory=True)))
        self.assertEqual(job.pool_outcome(dict(returncode=0, done=done), ['a']), 'completed')
        self.assertEqual(job.pool_outcome(dict(returncode=0, done=done, tunnel_dropped_s=3.0), ['a']), 'tunnel_dropped')
        self.assertEqual(job.pool_outcome(dict(returncode=4, refused=dict(reason='lock')), ['a']), 'refused_tbench_busy')
        self.assertEqual(job.pool_outcome(dict(returncode=2, refused=dict(reason='spec rejected')), ['a']),
                         'pool_refused')
        outcomes = job.instance_outcomes(['a', 'b'], dict(timed_out=True, done=dict(instances=dict(a=dict(trajectory=True)))))
        self.assertEqual((outcomes['a']['status'], outcomes['b']['status']), ('done', 'truncated'))

    def test_result_row_carries_the_goal_text_presence(self):
        row = job.result_row(dict(label='goal', goal_text=dict(status='PRESENT', records=3, declared=['g.jsonl'],
                                                               undeclared=[], files={'g.jsonl': dict(records=3)})))
        self.assertEqual(row['goal_text'], dict(status='PRESENT', records=3, files={'g.jsonl': 3}, declared=['g.jsonl'],
                                                undeclared=[]))
        self.assertIsNone(job.result_row(dict(label='stock'))['goal_text'])

    def test_server_handoff_waits_for_the_engine_child(self):
        """perf-hbm job 3428610: the 2-s LISTEN check aborted a complete job while the EngineCore child that inherited
        the listening fd was still tearing down (~7-8 s). Now the port and the GPU are waited for up to 120 s; a
        5-s linger passes with the finding HANDOFF_WAIT, a linger past the limit still fails."""
        from qualification_ports import Ports, HANDOFF_LIMIT_S

        class Clock:
            def __init__(self):
                self.t = 0.0

            def __call__(self):
                return self.t

            def sleep(self, seconds):
                self.t += seconds

        class Exited:
            def poll(self):
                return 0
        self.assertEqual(HANDOFF_LIMIT_S, 120)
        for linger, ok in ((5.0, True), (0.0, True), (121.0, False)):
            with tempfile.TemporaryDirectory() as tmp:
                clock = Clock()
                ports = Ports(Path(tmp), listener=lambda: set())
                port, _ = ports.allocate('goalb4')
                ports.listener = lambda: {port} if clock() < linger else set()
                if ok:
                    waited = ports.stopped('goalb4', Exited(), clock=clock, sleep=clock.sleep)
                    self.assertAlmostEqual(waited, linger, delta=0.05)
                    row = json.loads((Path(tmp) / 'ports.jsonl').read_text().splitlines()[-1])
                    self.assertEqual((row['event'], row['limit_s']), ('handoff_complete', 120))
                else:
                    with self.assertRaisesRegex(RuntimeError, 'LISTEN'):
                        ports.stopped('goalb4', Exited(), clock=clock, sleep=clock.sleep)
                    row = json.loads((Path(tmp) / 'ports.jsonl').read_text().splitlines()[-1])
                    self.assertEqual(row['reason'], 'tcp_listen_remains')
        # the GPU idle check: a compute process lingering 5 s passes (idle_wait_s), past 120 s it fails
        for linger, ok in ((5.0, True), (130.0, False)):
            clock = Clock()
            state = lambda: dict(compute_pids=['123'] if clock() < linger else [], memory_mib=0)
            if ok:
                value = job.idle([], None, state_of=state, shared_of=lambda: [], clock=clock, sleep=clock.sleep)
                self.assertAlmostEqual(value['idle_wait_s'], 5.0, delta=0.3)
            else:
                with self.assertRaisesRegex(RuntimeError, 'next server prohibited'):
                    job.idle([], None, state_of=state, shared_of=lambda: [], clock=clock, sleep=clock.sleep)
        # the job's record and finding: HANDOFF_WAIT above 2 s, none for a prompt handoff
        with tempfile.TemporaryDirectory() as tmp:
            for wait, finding in ((7.5, 'HANDOFF_WAIT'), (0.4, None)):
                class FakePorts:
                    def stopped(self, label, process):
                        return wait
                record = job.handoff(Path(tmp), FakePorts(), 'goalb4-%s' % wait, Exited(),
                                     lambda shm, limit: dict(compute_pids=[], idle_wait_s=0.0), [], None)
                self.assertEqual((record['listen_wait_s'], record['limit_s'], record['finding']), (wait, 120, finding))
            self.assertIn('handoff', job.RESULT_KEYS)

    def test_close_leftovers_retries_busy_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [dict(schema='x', event='open', session_index=0), dict(schema='x', event='open', session_index=1),
                    dict(schema='x', event='close', session_index=1)]
            (Path(tmp) / 'sessions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            answers = [(409, b'busy'), (200, b'{}')]
            closed = job.close_leftovers('http://x', tmp, [dict(instance_id='a'), dict(instance_id='b')],
                                         post=lambda url, body: answers.pop(0), sleep=lambda s: None)
            self.assertEqual(closed, [dict(session=0, instance_id='a', http=200, attempts=2, body='{}')])


class Archives(unittest.TestCase):
    def stage(self, tmp):
        stage = Path(tmp) / 'stage'
        out = stage / 'job-output' / 'goal'
        out.mkdir(parents=True)
        rows = []
        for session in (1, 0):
            for turn in (1, 0):
                rid = 'req-%d-%d' % (session, turn)
                rows.append(dict(request_id=rid, session_index=session, turn=turn))
                (out / ('native-%s.json' % hashlib.sha256(rid.encode()).hexdigest())).write_text(
                    json.dumps(dict(prompt=[session] * 1000 + [turn])))
        (out / 'runtime.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (out / 'ready.json').write_text('{}')
        (out / 'goal-text.jsonl').write_text('{"request_id": "req-0-0"}\n')
        (stage / 'result.json').write_text('{}')
        return stage

    def test_core_and_bulk_split_and_order(self):
        import verify_evidence
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.stage(tmp)
            value = job.archive(stage, xz='')  # Python's lzma
            self.assertEqual(value['bulk']['members'], 5)
            with tarfile.open(stage / 'fetch' / job.BULK_ARCHIVE, 'r:xz') as stream:
                names = stream.getnames()
            self.assertEqual(names[0], 'job-output/goal/runtime.jsonl')
            index = json.loads((stage / 'evidence-sha256.json').read_text())
            self.assertEqual(sorted(index['bulk']), sorted(names))
            self.assertIn('job-output/goal/ready.json', index['core'])
            self.assertIn('job-output/goal/goal-text.jsonl', index['core'])  # the Goal text travels in the core archive
            self.assertIn('result.json', index['core'])
            digests = verify_evidence.safe_extract(stage / 'fetch' / job.CORE_ARCHIVE, Path(tmp) / 'x')
            self.assertIn('evidence-sha256.json', digests)

    def test_system_xz_when_present(self):
        import shutil as sh
        if not sh.which('xz'):
            self.skipTest('no xz')
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.stage(tmp)
            value = job.archive(stage)
            self.assertNotEqual(value['core']['compressor'], 'python-lzma')
            with tarfile.open(stage / 'fetch' / job.CORE_ARCHIVE, 'r:xz') as stream:
                self.assertIn('result.json', stream.getnames())


class LockWait(unittest.TestCase):
    class Link(tbench_link.Link):
        def __init__(self, answers):
            self.answers, self.t = list(answers), [0.0]
            self.clock = lambda: self.t[0]
            self.sleep = lambda s: self.t.__setitem__(0, self.t[0] + s)

        def lock_status(self):
            value = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            return dict(status='LOCK_READ', lock=value)

    def test_waits_for_an_evaluation_then_runs(self):
        held = dict(held=True, alive=True, kind='eval', run_id='swe32-1-r1')
        link = self.Link([held, held, dict(held=False, alive=False)])
        value = link.wait_for_lock(100.0, 30, log=lambda line: None)
        self.assertEqual((value['status'], value['waited_s'], len(value['polls'])), ('LOCK_FREE', 60.0, 3))

    def test_refuses_at_the_bound_or_beside_a_live_pool(self):
        held = dict(held=True, alive=True, kind='eval', run_id='swe32-1-r1')
        self.assertEqual(self.Link([held]).wait_for_lock(50.0, 30, log=lambda line: None)['status'], 'LOCK_HELD_TIMEOUT')
        pool = dict(held=True, alive=True, kind='pool', run_id='swe32-2-r1')
        self.assertEqual(self.Link([pool]).wait_for_lock(5000.0, 30, log=lambda line: None)['status'],
                         'LOCK_HELD_BY_LIVE_RUN')
        stale = dict(held=True, alive=False, kind='pool', run_id='swe32-2-r1')
        self.assertEqual(self.Link([stale]).wait_for_lock(0.0, 30, log=lambda line: None)['status'], 'LOCK_FREE')

    def test_pool_command_validation(self):
        link = tbench_link.Link('/u/ycao11/swe32-x', PINS['tbench'])
        with self.assertRaises(RuntimeError):
            link.pool_command('swe32-1-r1', 'BAD', 'runtime', 22000, '/s', 'abc', ['a'])
        with self.assertRaises(RuntimeError):
            link.pool_command('swe32-1-r1', 'full', 'runtime', 22000, '/s', 'a b', ['a'])
        argv = link.pool_command('swe32-1-r1', 'full', 'runtime', 22000, '/s', 'abc', ['a'])
        self.assertEqual(argv[-8:], [PINS['tbench']['bin'] + '/run_pool.sh', 'swe32-1-r1', 'full', 'runtime', '22000',
                                     '/s', 'abc', 'a'])


class Failures(unittest.TestCase):
    TASKS = [dict(instance_id='a', problem_statement='x'), dict(instance_id='b', problem_statement='y')]

    def test_context_refusal_and_cancellation_are_allowed(self):
        refusal = dict(session_index=0, turn=2, request_id=None, error='VLLMValidationError',
                       detail="This model's maximum context length is 131072 tokens.")
        cancel = dict(session_index=1, turn=1, request_id='r9', error='CancelledError', detail='client went away')
        allowed, disallowed = swe_checks.classify_failures([refusal, cancel], {0: [1, 2], 1: [1]},
                                                           dict(a=dict(status='done', exit_status='ContextWindowExceededError'),
                                                                b=dict(status='truncated')), self.TASKS)
        self.assertEqual([f['kind'] for f in allowed], ['context_limit', 'cancelled_at_stop'])
        self.assertEqual(disallowed, [])

    def test_other_failures_fail(self):
        other = dict(session_index=0, turn=0, request_id='r1', error='RuntimeError', detail='engine rejected')
        allowed, disallowed = swe_checks.classify_failures([other], {0: []}, dict(a=dict(status='done')), self.TASKS)
        self.assertEqual((allowed, len(disallowed)), ([], 1))
        late = dict(session_index=1, turn=5, request_id='r1', error='CancelledError')
        allowed, disallowed = swe_checks.classify_failures([late], {1: [1]}, dict(b=dict(status='truncated')), self.TASKS)
        self.assertEqual(len(disallowed), 1)


class DecodeFastpath(unittest.TestCase):
    def files(self, tmp, startup, ready, worker):
        out = Path(tmp)
        (out / 'startup.json').write_text(json.dumps(startup))
        (out / 'ready.json').write_text(json.dumps(ready))
        (out / 'worker-attached.json').write_text(json.dumps(dict(receipt=worker)))
        return out

    def test_on_everywhere_or_nowhere(self):
        on = dict(decode_fastpath=True)
        with tempfile.TemporaryDirectory() as tmp:
            out = self.files(tmp, on, on, on)
            self.assertEqual(swe_checks.check_decode_fastpath(out, dict(decode_fastpath=True)), dict(decode_fastpath=True))
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_decode_fastpath(out, dict(decode_fastpath=None))
        with tempfile.TemporaryDirectory() as tmp:
            out = self.files(tmp, on, on, {})  # the worker did not echo it
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_decode_fastpath(out, dict(decode_fastpath=True))
        with tempfile.TemporaryDirectory() as tmp:
            out = self.files(tmp, {}, {}, {})
            self.assertEqual(swe_checks.check_decode_fastpath(out, dict(decode_fastpath=None)), dict(decode_fastpath=None))


def goal_row(session, turn, request_id=None, selector=None, text=None):
    selector = selector or dict(schema_version='goal-v2', root_goal='fix it', policy_hint=None,
                                retrieval_goal=dict(description='goal %d' % turn, files=['a.py'], symbols=[], tests=[],
                                                    errors=[], constraints=[]))
    text = text if text is not None else 'Goal of turn %d: fix a.py' % turn
    row = dict(request_id=request_id or 'req-%d-%d' % (session, turn), response_id='resp-%d-%d' % (session, turn),
               session_index=session, turn=turn,
               query=dict(goal_hash=goal_text.sha(goal_text.canonical(selector)),
                          goal=dict(payload_sha256=goal_text.sha(text), applied_decision='PATCH')))
    record = dict(request_id=row['request_id'], session_index=session, turn=turn, applied_goal=dict(
        selector, turn_id=turn + 1, diagnostics={}), goal_text=text, goal_text_sha256=goal_text.sha(text),
        rewriter=dict(raw_output='{"x": 1}', raw_output_sha256=goal_text.sha('{"x": 1}'), error=None,
                      request=dict(messages=[dict(role='user', content='u')])))
    return row, record


class GoalText(unittest.TestCase):
    def rows_records(self, n=3):
        pairs = [goal_row(k, t) for k in range(2) for t in range(n)]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def test_verified_when_every_turn_has_its_texts(self):
        rows, records = self.rows_records()
        value = goal_text.verify_rows({'goal-text.jsonl': records}, rows + [dict(request_id='x', query=None)],
                                      'goal_query', ('goal-text.jsonl',))
        self.assertEqual((value['status'], value['goal_turns'], value['matched'], value['self_pairs']),
                         ('VERIFIED', 6, 6, 12))
        self.assertEqual(value['joined_by'], dict(request_id=6))

    def test_a_changed_text_a_missing_turn_or_a_wrong_copy_fail(self):
        rows, records = self.rows_records()
        changed = copy.deepcopy(records)
        changed[1]['goal_text'] += ' (edited)'
        value = goal_text.verify_rows({'g.jsonl': changed}, rows, 'goal_query', ('g.jsonl',))
        self.assertEqual(value['status'], 'FAIL')
        self.assertTrue(any('payload_sha256' in r for r in value['reasons']))
        self.assertEqual(len(value['self_pair_failures']), 1)
        value = goal_text.verify_rows({'g.jsonl': records[:-1]}, rows, 'goal_query', ('g.jsonl',))
        self.assertEqual((value['status'], len(value['missing'])), ('FAIL', 1))
        wrong = copy.deepcopy(records)
        wrong[0]['goal_hash'] = '0' * 64
        self.assertEqual(goal_text.verify_rows({'g.jsonl': wrong}, rows, 'goal_query')['status'], 'FAIL')

    def test_not_available_without_files_and_fail_when_declared(self):
        rows, _ = self.rows_records()
        self.assertEqual(goal_text.verify_rows({}, rows, 'goal_query')['status'], 'NOT_AVAILABLE')
        self.assertEqual(goal_text.verify_rows({}, rows, 'goal_query', ('g.jsonl',))['status'], 'FAIL')
        self.assertEqual(goal_text.verify_rows({}, rows, 'observation_query', ('g.jsonl',))['status'], 'NOT_AVAILABLE')

    def test_other_formats_and_joins(self):
        rows, records = self.rows_records(2)
        by_response = [dict({k: v for k, v in r.items() if k not in ('request_id', 'session_index', 'turn')},
                            response_id='resp-%d-%d' % (r['session_index'], r['turn'])) for r in records]
        value = goal_text.verify_rows({'g.json': by_response}, rows, 'goal_query')
        self.assertEqual((value['status'], value['joined_by']), ('VERIFIED', dict(response_id=4)))
        self.assertEqual(value['findings'][0]['finding'], 'GOAL_TEXT_UNDECLARED')
        unkeyed = [{k: v for k, v in r.items() if k not in ('request_id', 'session_index', 'turn')} for r in records]
        value = goal_text.verify_rows({'g.json': unkeyed}, rows, 'goal_query', ('g.json',))
        self.assertEqual((value['status'], value['joined_by']), ('VERIFIED', dict(hash=4)))
        # the selector payload as its canonical string, the Goal as one JSON list file
        row, record = goal_row(0, 0)
        flat = dict(request_id=record['request_id'], goal_json=goal_text.canonical(dict(
            schema_version='goal-v2', root_goal='fix it', policy_hint=None, retrieval_goal=record['applied_goal'][
                'retrieval_goal'])), payload=record['goal_text'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'goal_texts.json'
            path.write_text(json.dumps([flat]))
            self.assertEqual([p.name for p in goal_text.candidate_files(tmp)], ['goal_texts.json'])
            value = goal_text.verify_server(tmp, [row], 'goal_query', ('goal_texts.json',))
            self.assertEqual(value['status'], 'VERIFIED')
            (Path(tmp) / 'goal_texts.json').write_text('{"broken": ')
            self.assertEqual(goal_text.verify_server(tmp, [row], 'goal_query')['status'], 'FAIL')

    def test_candidate_files_ignore_the_known_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('startup.json', 'ready.json', 'runtime.jsonl', 'measurements.jsonl', 'native-ab.json',
                         'sessions.jsonl', 'notes.txt'):
                (Path(tmp) / name).write_text('{}')
            self.assertEqual(goal_text.candidate_files(tmp), [])
            (Path(tmp) / 'rewriter-io.jsonl').write_text('{}\n')
            self.assertEqual([p.name for p in goal_text.candidate_files(tmp)], ['rewriter-io.jsonl'])

    def test_presence_check_of_the_job(self):
        rows, records = self.rows_records(1)
        with tempfile.TemporaryDirectory() as tmp:
            expect = dict(goal_text=dict(files=['goal-text.jsonl']))
            with self.assertRaises(swe_checks.CheckFailed):  # declared, goal_query, nothing written
                swe_checks.check_goal_text_present(tmp, 'goal_query', expect)
            (Path(tmp) / 'goal-text.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            value = swe_checks.check_goal_text_present(tmp, 'goal_query', expect)
            self.assertEqual((value['records'], value['undeclared']), (2, []))
            with self.assertRaises(swe_checks.Finding):  # the runtime wrote a file it does not declare
                swe_checks.check_goal_text_present(tmp, 'goal_query', dict(goal_text=None))

    def test_the_goals_jsonl_rows_of_the_goal_text_runtime(self):
        """227e32c's format: canonical rows {request_id, ..., goal {sha256, text}, query {variant, sha256, text},
        decision, rewrites}, named by the runtime row's query.goal_text {file, sha256 of the row}."""
        lines, rows = [], []
        for turn in range(3):
            row, _ = goal_row(0, turn)
            selector = dict(schema_version='goal-v2', root_goal='fix it', policy_hint=None,
                            retrieval_goal=dict(description='goal %d' % turn, files=['a.py'], symbols=[], tests=[],
                                                errors=[], constraints=[]))
            text = 'Goal of turn %d: fix a.py' % turn
            record = dict(request_id=row['request_id'], session_id='http-x', session_index=0, epoch='e', turn=turn,
                          local_turn=turn, terminal_state='complete', schema='remote-ra-goal-text-v1',
                          goal=dict(sha256=row['query']['goal_hash'], text=goal_text.canonical(selector)),
                          query=dict(variant='full_goal', sha256=row['query']['goal']['payload_sha256'], text=text),
                          decision=dict(applied_decision='PATCH'),
                          rewrites=[dict(request='{"r":1}', response=None, error='TimeoutError: 120 s')])
            raw = goal_text.canonical(record).encode()
            row['query']['goal_text'] = dict(file='goals.jsonl', sha256=hashlib.sha256(raw).hexdigest())
            lines.append(raw)
            rows.append(row)
        cancelled = dict(json.loads(lines[0]), request_id='req-cancel', terminal_state='cancelled')
        lines.append(goal_text.canonical(cancelled).encode())
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'goals.jsonl').write_bytes(b''.join(line + b'\n' for line in lines))
            value = goal_text.verify_server(tmp, rows, 'goal_query', ('goals.jsonl',))
            self.assertEqual((value['status'], value['matched'], value['joined_by'], value['self_pairs'],
                              value['unjoined_records'], value['rewriter_errors'], value['terminal_states']),
                             ('VERIFIED', 3, dict(row_sha256=3), 8, 1, 4, dict(complete=3, cancelled=1)))
            rows[1]['query']['goal_text']['sha256'] = '0' * 64  # a reference to no row
            value = goal_text.verify_server(tmp, rows, 'goal_query', ('goals.jsonl',))
            self.assertEqual((value['status'], len(value['bad_references'])), ('FAIL', 1))
            rows[1]['query']['goal_text']['sha256'] = hashlib.sha256(lines[1]).hexdigest()
            wrong = json.loads(lines[2])
            wrong['goal']['sha256'] = '1' * 64  # the pair no longer holds
            (Path(tmp) / 'goals.jsonl').write_bytes(b''.join(line + b'\n' for line in lines[:2] + [
                goal_text.canonical(wrong).encode()]))
            value = goal_text.verify_server(tmp, rows, 'goal_query', ('goals.jsonl',))
            self.assertEqual(value['status'], 'FAIL')
            self.assertTrue(value['self_pair_failures'] and value['bad_references'])

    def test_rewriter_timing_fields_are_tolerated(self):
        """A coming runtime commit (from 227dedb) adds per-call wait / call times to goals.jsonl's rewrites[]: extra
        fields of a rewrite (numbers, not *_sha256 pairs) change no verification; the record is still VERIFIED and the
        calls are counted (re-pin: build.py --commit <hash>, then local_validation.ps1)."""
        row, _ = goal_row(0, 0)
        selector = dict(schema_version='goal-v2', root_goal='fix it', policy_hint=None,
                        retrieval_goal=dict(description='goal 0', files=['a.py'], symbols=[], tests=[], errors=[],
                                            constraints=[]))
        record = dict(request_id=row['request_id'], session_index=0, turn=0, terminal_state='complete',
                      schema='remote-ra-goal-text-v1',
                      goal=dict(sha256=row['query']['goal_hash'], text=goal_text.canonical(selector)),
                      query=dict(variant='full_goal', sha256=row['query']['goal']['payload_sha256'],
                                 text='Goal of turn 0: fix a.py'),
                      decision=dict(applied_decision='REWRITE'),
                      rewrites=[dict(request='{"r":1}', response='{"g":1}', error=None, wait_ms=81234.5, call_ms=2875.25,
                                     started_wall=1791777047.17, queue_depth=7)])
        raw = goal_text.canonical(record).encode()
        row['query']['goal_text'] = dict(file='goals.jsonl', sha256=hashlib.sha256(raw).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'goals.jsonl').write_bytes(raw + b'\n')
            value = goal_text.verify_server(tmp, [row], 'goal_query', ('goals.jsonl',))
        self.assertEqual((value['status'], value['matched'], value['rewriter_calls'], value['rewriter_errors']),
                         ('VERIFIED', 1, 1, 0))

    def test_per_turn_join_for_the_package(self):
        rows, records = self.rows_records(2)
        joined = goal_text.per_turn({'g.jsonl': records}, rows)
        self.assertEqual(sorted(joined), [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(joined[(1, 1)][0]['_file'], 'g.jsonl')


class Statements(unittest.TestCase):
    def test_subset_of_a_sealed_statements_file(self):
        import build
        tasks = [dict(instance_id='i%d' % k, problem_statement='statement %d' % k) for k in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pool.json'
            path.write_bytes(c.tasks_bytes(tasks))
            raw = build.subset_pool(['i3', 'i1'], path)
            self.assertEqual(json.loads(raw), [tasks[3], tasks[1]])
            self.assertEqual(raw, c.tasks_bytes([tasks[3], tasks[1]]))
            with self.assertRaises(ValueError):
                build.subset_pool(['i9'], path)
            path.write_bytes((json.dumps(tasks, indent=2) + '\n').encode())
            with self.assertRaises(ValueError):  # another encoding
                build.subset_pool(['i1'], path)

    def test_the_sealed_statements_are_swe16s(self):
        swe16 = ROOT.parent / 'perf-swe16' / 'tasks' / 'tasks-pool.json'
        if not swe16.is_file():
            self.skipTest('perf-swe16 statements absent')
        mine = (ROOT / 'tasks' / 'tasks-pool.json').read_bytes()
        self.assertEqual(mine, swe16.read_bytes())  # byte-identical: swe16's tbench-built file
        self.assertEqual(hashlib.sha256(mine).hexdigest(), PINS['tasks']['pool']['sha256'])
        self.assertEqual(PINS['tasks']['pool']['sha256'],
                         'e483592486260d23f205833c3771e103e6ae300078c2f2dbe710cb43b5c71ade')  # swe16's result.json pin
        self.assertEqual([t['instance_id'] for t in json.loads(mine)], c.instance_ids(PINS))


REF = Path(r'D:\Remote-RA-temp\scratch\perf-swe16\ref')  # local extracts of real evidence (never shared; perf-swe16's)


@unittest.skipUnless(REF.is_dir(), 'real evidence extracts absent')
class RealEvidence(unittest.TestCase):
    """The evidence-format functions on real int-1010-era evidence: job 3343069's session pool (24 tasks, K = 8) and
    job 3348705's frontend HTTP 400 at the 131072 context limit; and the kernel lines of real server logs."""

    @staticmethod
    def by_session(rows):
        out = {}
        for row in sorted(rows, key=lambda r: (r['session_index'], r['turn'])):
            out.setdefault(row['session_index'], []).append(row)
        return out

    def test_session_pool_of_job_3343069(self):
        out = REF / 'sint1' / 'job-output' / 'i-P8-obs-fd'
        ready = json.loads((out / 'ready.json').read_text())
        rows = swe_checks.runtime_rows(out)
        names = {e['session_index']: e['instance_id'] for e in swe_checks.session_events(out)}
        T = ready['session_pool']['registered']
        expect = dict(running=ready['session_pool']['max_running_sessions'],
                      tasks=[dict(instance_id=names[k]) for k in range(T)])
        value = swe_checks.check_pool(out, expect, rows, swe_checks.ids_of(ready, T), ready, self.by_session(rows))
        self.assertEqual((value['registered'], value['opened'], value['closed'], value['peak_open'], value['poisoned']),
                         (24, 24, 24, 8, 0))
        self.assertEqual(value['open_waits'], 16)

    def test_context_refusal_of_job_3348705(self):
        out = REF / 'sctx1' / 'job-output' / 'c-L1-full'
        rows = swe_checks.runtime_rows(out)
        failures = [json.loads(line) for line in (out / 'failures.jsonl').read_text().splitlines() if line.strip()]
        tasks = [dict(instance_id='t0')]
        allowed, disallowed = swe_checks.classify_failures(failures, self.by_session(rows), dict(
            t0=dict(status='done', exit_status='ContextWindowExceededError')), tasks)
        self.assertEqual([(a['kind'], a['turn'], a['request_id']) for a in allowed], [('context_limit', 11, None)])
        self.assertEqual(disallowed, [])

    def test_kernel_lines_of_real_server_logs(self):
        for log in (REF / 'sctx1' / 'server-c-B16-obs-full.log', REF / 'sint1' / 'server-i-S16-obs.log'):
            self.assertEqual(swe_checks.check_kernels(log)['status'], 'KERNELS_ALIGNED')


SWE16_GOAL = Path(r'D:\Remote-RA-temp\scratch\perf-swe16\fetch\job3363871')


@unittest.skipUnless(SWE16_GOAL.is_dir(), 'swe16 goal_query evidence absent')
class RealGoalRows(unittest.TestCase):
    """The Goal-text reader on swe16's real goal_query rows (job 3363871, 00e173d's predecessor e877051): the hashes are
    there, no Goal-text file exists (NOT_AVAILABLE), and a record built the way the goal-text runtime describes it
    (the payload text and the selector payload) verifies only when its texts are the right ones."""

    def test_real_rows(self):
        out = sorted(SWE16_GOAL.glob('evidence-*'))[-1] / 'files' / 'job-output' / 'goal'
        if not (out / 'runtime.jsonl').is_file():
            self.skipTest('bulk not extracted')
        rows = []
        with (out / 'runtime.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                rows.append(json.loads(line))
                if len(rows) >= 20:
                    break
        for row in rows:
            row.pop('prompt_token_ids', None)
        self.assertEqual(goal_text.candidate_files(out), [])
        value = goal_text.verify_rows({}, rows, 'goal_query')
        self.assertEqual((value['status'], value['goal_turns']), ('NOT_AVAILABLE', 20))
        forged = [dict(request_id=r['request_id'], goal_text='not the Goal') for r in rows]
        self.assertEqual(goal_text.verify_rows({'g.jsonl': forged}, rows, 'goal_query')['status'], 'FAIL')


class Kernels(unittest.TestCase):
    """kernel-alignment: vLLM 0.24.0's own startup lines (as in job 3348705's and job 3321538's server logs)."""
    SAMPLER = ('(EngineCore pid=7) INFO 10-09 11:57:18 [topk_topp_sampler.py:39] FlashInfer top-p/top-k sampling '
               'disabled via VLLM_USE_FLASHINFER_SAMPLER=0.\n')
    GDN = ('(EngineCore pid=7) INFO 10-09 11:57:23 [qwen_gdn_linear_attn.py:228] Using Triton/FLA GDN prefill kernel '
           '(requested=triton, head_k_dim=128).\n')
    ATTN = ("(EngineCore pid=7) INFO 10-09 11:57:24 [cuda.py:480] Using FLASH_ATTN attention backend out of potential "
            "backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].\n")
    FA = '(EngineCore pid=7) INFO 10-09 11:57:24 [flash_attn.py:670] Using FlashAttention version 3\n'

    def check(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'server.log'
            path.write_text(text)
            return swe_checks.check_kernels(path)

    def test_the_aligned_set_passes(self):
        value = self.check(self.SAMPLER + self.GDN + self.ATTN + self.FA)
        self.assertEqual(value['status'], 'KERNELS_ALIGNED')
        self.assertEqual(value['gdn_prefill'], [['Triton/FLA', 'triton']])

    def test_any_other_choice_fails(self):
        good = self.SAMPLER + self.GDN + self.ATTN + self.FA
        for bad in (good.replace('Using FLASH_ATTN attention', 'Using FLASHINFER attention'),
                    good + self.ATTN.replace('FLASH_ATTN attention', 'TRITON_ATTN attention'),
                    good.replace('version 3', 'version 2'),
                    good.replace('requested=triton', 'requested=flashinfer'),
                    good.replace('Using Triton/FLA GDN', 'Using FlashInfer GDN'),
                    good + '(EngineCore pid=7) INFO [topk_topp_sampler.py:30] Using FlashInfer for top-p & top-k sampling.\n',
                    self.GDN + self.ATTN + self.FA, self.SAMPLER + self.ATTN + self.FA, self.SAMPLER + self.GDN, ''):
            with self.assertRaises(swe_checks.CheckFailed):
                self.check(bad)

    def test_both_launches_carry_the_aligned_settings_and_no_attention_pin(self):
        import stock
        argv = c.stock_argv(PINS, 8123)
        self.assertEqual(argv[argv.index('--gdn-prefill-backend') + 1], 'triton')
        self.assertNotIn('--attention-backend', argv)
        self.assertEqual(stock.server_env({}, Path('/x'))['VLLM_USE_FLASHINFER_SAMPLER'], '0')
        self.assertEqual(job.runtime_env(Path('/x'), {})['VLLM_USE_FLASHINFER_SAMPLER'], '0')
        for plan in ('full', 'obs', 'rfill', 'goal'):
            self.assertNotIn('--attention-backend', c.launch_arguments(PINS, server(plan), REWRITER))


class HostCheck(unittest.TestCase):
    """tbench/bin/host_check.py: refuses only for leftovers of our runs and too little memory / disk."""

    def check(self, containers, mode='strict', lock=None, available_kb=25000000):
        import importlib.util
        import io
        import contextlib
        spec = importlib.util.spec_from_file_location('host_check', ROOT / 'tbench' / 'bin' / 'host_check.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / 'runs' / 'swe32-1-r1' / 'full' / 'logs'
            logs.mkdir(parents=True)
            (logs / 'a.log').write_text('Started container minisweagent-0badc0de\n')
            proc = root / 'proc'
            proc.mkdir()
            (proc / 'loadavg').write_text('9.5 3.0 1.0 1/1 1\n')
            (proc / 'meminfo').write_text('MemTotal: 30000000 kB\nMemAvailable: %d kB\n' % available_kb)
            (root / 'swe32.env').write_text('SWE32_MIN_MEM_GB=12\nSWE32_MIN_FREE_GB=0\nSWE32_HOST_PROC=%s\n' % proc)
            table = ''.join('%s\t%s\timg\tUp\n' % (n[:12].ljust(12, '0'), n) for n in containers)
            module.run = lambda argv, timeout=60: (0, table, '') if 'ps' in argv[:3] else (1, '', '')
            module.lock_state = lambda root: lock
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = module.main(str(root), mode)
            return code, json.loads(out.getvalue())

    def test_other_work_and_load_are_reported_not_refused(self):
        code, value = self.check(['harbor-task-demo', 'jobs-tb-1', 'sweb.eval.astropy__astropy-12907.swe16-1-r1-full'])
        self.assertEqual((code, value['status'], value['other_containers']), (0, 'HOST_FREE', 3))  # swe16's are not ours
        self.assertEqual(value['load'][0], 9.5)

    def test_our_leftovers_and_low_memory_refuse(self):
        code, value = self.check(['harbor-task-demo', 'minisweagent-0badc0de'])
        self.assertEqual((code, value['status'], value['our_leftovers']), (5, 'HOST_BUSY', 1))
        self.assertEqual(self.check([], available_kb=5000000)[1]['status'], 'HOST_BUSY')
        eval_lock = dict(held=True, alive=True, kind='eval', run_id='swe32-1-r1')
        name = 'sweb.eval.astropy__astropy-12907.swe32-1-r1-full'
        self.assertEqual(self.check([name], 'preflight', eval_lock, 5000000)[1]['status'], 'HOST_FREE')
        self.assertEqual(self.check([name], 'strict', None)[1]['status'], 'HOST_BUSY')
        self.assertEqual(self.check([name], 'preflight', None)[1]['status'], 'HOST_BUSY')


class Ctx262(unittest.TestCase):
    """The group's new settings (2026-10-26): ctx24 / ctx25 run goal_query (goal262 = goalb4's server) then
    observation_query (obs262 = obs's server) in one ghx4 job at --max-model-len 262144 with --kv-cache-gib 24 / 25 and
    the 70-GiB arena (config.plans.<name>.server over config.server); ctx25 is blocked while the sealed Qwen launcher's
    KV_CACHE_GIB lacks 25 (controller.kv_block); every other plan is unchanged."""

    def own(self, args, gib):
        """goalb4's / obs's serve.py arguments with the plan's three own values."""
        out = list(args)
        for flag, value in (('--kv-cache-gib', str(gib)), ('--max-model-len', '262144'), ('--dram-arena-gib', '70')):
            out[out.index(flag) + 1] = value
        return out

    def test_the_plans_run_goalb4s_and_obss_servers_with_their_own_settings(self):
        for name, gib in (('ctx24', 24), ('ctx25', 25)):
            goal, obs = c.plan_servers(PINS, name)
            self.assertEqual((goal['label'], goal['policy'], obs['label'], obs['policy']),
                             ('goal262', 'goal_query', 'obs262', 'observation_query'))
            self.assertEqual(goal['settings'], dict(kv_cache_gib=gib, max_model_len=262144, dram_arena_gib=70))
            self.assertEqual(c.server_cfg(PINS, goal), dict(c.cfg(PINS)['server'], kv_cache_gib=gib,
                                                            max_model_len=262144, dram_arena_gib=70))
            self.assertEqual(c.launch_arguments(PINS, goal, REWRITER),
                             self.own(c.launch_arguments(PINS, server('goalb4'), REWRITER), gib))
            self.assertEqual(c.launch_arguments(PINS, obs, REWRITER),
                             self.own(c.launch_arguments(PINS, server('obs'), REWRITER), gib))
            self.assertEqual((goal[c.BATCH_KEY], goal[c.BATCH_TOKENS_KEY], obs.get(c.BATCH_KEY)), (4, 16384, None))
            self.assertEqual(c.cfg(PINS)['plans'][name]['role'], 'validation')
            self.assertNotIn('settings', server('goalb4'))  # every other plan: config.server itself
            self.assertIs(c.server_cfg(PINS, server('goalb4')), c.cfg(PINS)['server'])

    def test_expectation_pools_arena_and_capacity(self):
        """The checks' expectation at the plan's settings; the launcher's lone-turn rule (README 4.9) at K = 16 and
        262,144: 17,862 of 19,681 usable attention blocks at 24 GiB, 17,873 of 20,705 at 25 GiB (fits); the arena 70 GiB
        = 71,680 Pages (one session at the limit 16,385, sixteen 262,160), pinned 73.29 GiB as at 131072."""
        for name, gib, blocks, lone in (('ctx24', 24, 19682, 17862), ('ctx25', 25, 20706, 17873)):
            goal = c.plan_servers(PINS, name)[0]
            expect = c.server_expectation(PINS, goal, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm', [])
            self.assertEqual((expect['max_model_len'], expect['kv_cache_bytes'], expect['max_tokens']),
                             (262144, gib * c.GIB, 4096))
            self.assertEqual(expect['layout']['pools'], [blocks, 100])
            self.assertEqual((expect['layout']['arena_pages'], expect['layout']['dram_arena_pages']), (71680, 71680))
            self.assertEqual(expect['layout']['pinned_host']['gib'], 73.29)
            fit = c.attention_fit(PINS, c.LAYOUT, 16, goal)
            self.assertEqual((fit['fa_blocks'], fit['usable_blocks'], fit['one_turn_blocks'], fit['fits']),
                             (blocks, blocks - 1, lone, True))
            memory = c.memory_fit(PINS, name)
            self.assertEqual((memory['pinned_host_gib'], memory['need_gib'], memory['headroom_gib'], memory['arena_gib'],
                              memory['fits']), (73.29, 105.29, 5.61, 70, True))
            fit = c.validate_plan(PINS, name)
            self.assertEqual((fit['time_seconds'], fit['worst_case_seconds'], fit['with_eval_wait_seconds']),
                             (8820, 8820, 7270))
            cost = c.plan_cost(PINS, name)
            self.assertEqual((cost['gpu_hours_limit'], cost['gpu_hours_expected']), (2.45, 1.861))
        # the stock control and every other plan keep 32 GiB / 131072
        expect = c.server_expectation(PINS, server('goalb4'), dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm', [])
        self.assertEqual((expect['max_model_len'], expect['kv_cache_bytes']), (131072, 32 * c.GIB))

    def test_ctx25_is_blocked_while_the_qwen_launcher_refuses_25(self):
        sizes = PINS['runtime_features'][c.QWEN_KV_KEY]  # serve.py KV_CACHE_GIB of the pin
        self.assertIn(25, PINS['runtime_features']['kv_cache_gib_choices'])  # (argparse lists Gemma's 25 / 28)
        self.assertIsNone(c.plan_blocked(PINS, 'ctx24'))
        self.assertTrue(all(c.plan_blocked(PINS, n) is None for n in PLAN_ORDER
                            if n != 'ctx25' and n not in GOALEDGE))  # (11-16 phase B: those wait for goal-rollback)
        before = copy.deepcopy(PINS)  # 53e3b7a / 4f5f4e7: the Qwen launcher's sizes 8 / 16 / 24 / 32: blocked
        before['runtime_features'][c.QWEN_KV_KEY] = [8, 16, 24, 32]
        self.assertIn('--kv-cache-gib must be one of 8, 16, 24, 32', c.plan_blocked(before, 'ctx25'))
        later = copy.deepcopy(PINS)  # a re-pin to a runtime whose KV_CACHE_GIB holds 25 (integrate-1026): unblocked
        later['runtime_features'][c.QWEN_KV_KEY] = [8, 16, 24, 25, 32]
        self.assertIsNone(c.plan_blocked(later, 'ctx25'))
        self.assertEqual(c.plan_blocked(PINS, 'ctx25') is None, 25 in sizes)  # this pin: exactly by its own sizes
        if sizes == [8, 16, 24, 32]:  # 53e3b7a / 4f5f4e7
            self.assertIn('--kv-cache-gib must be one of 8, 16, 24, 32', c.plan_blocked(PINS, 'ctx25'))
        c.validate_plan(before, 'ctx25')  # a blocked plan still validates (time, memory, the launcher's refusals)
        import inspect
        self.assertIn('plan_blocked(pins, options[\'plan\'])', inspect.getsource(c.setup))
        self.assertIn('plan_blocked(pins, options[\'plan\'])', inspect.getsource(c.qualify))
        # the reader: serve.py's module-level tuple, not the argparse choices
        self.assertEqual(c.serve_constant("KV_CACHE_GIB=(8,16,24,32)   # also written out\n", 'KV_CACHE_GIB'),
                         (8, 16, 24, 32))
        self.assertIsNone(c.serve_constant("X=1\n", 'KV_CACHE_GIB'))

    def test_override_refusals(self):
        def refused(change, pattern):
            pins = copy.deepcopy(PINS)
            change(pins['config']['plans']['ctx24'])
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, 'ctx24')
        refused(lambda p: p['server'].update(page_tokens=32), 'server may set')  # (akvs: max_running_sessions may;
        # 2026-11-16: max_tokens may, within the launcher's choices)
        refused(lambda p: p['server'].update(max_tokens=2048), 'max_tokens')
        refused(lambda p: p['server'].update(max_running_sessions=0), 'server may set')
        refused(lambda p: p['server'].update(max_running_sessions=33), 'above')  # (11-16: the sealed 32)
        refused(lambda p: p['server'].update(kv_cache_gib='24'), 'server may set')
        refused(lambda p: p.update(server={}), 'server may set')
        refused(lambda p: p['server'].update(max_model_len=200000), 'max-model-len choices')
        refused(lambda p: p['server'].update(kv_cache_gib=20), 'kv-cache-gib choices')
        refused(lambda p: p['servers'].append(dict(label='stock262', policy='stock', step_limit=120,
                                                   agent_cap_seconds=600, agent_min_seconds=300)), 'stock control')
        refused(lambda p: p['server'].update(dram_arena_gib=12), 'must hold one session')  # < 16,385 Pages
        refused(lambda p: p['server'].update(dram_arena_gib=71), 'spare')  # 4.61 GiB spare at --mem=113560M

    def test_every_other_plan_is_unchanged(self):
        """Every derived value of the twelve plans before ctx24 / ctx25 equals its digest recorded on the 4f5f4e7 build
        before they were added (fixtures/swe32-plans-4f5f4e7.json, plan_fingerprint.py)."""
        import plan_fingerprint
        recorded = json.loads((ROOT / 'fixtures' / 'swe32-plans-4f5f4e7.json').read_bytes())
        self.assertEqual(sorted(recorded['plans']), sorted(PLANS[:12]))
        for name, parts in recorded['plans'].items():
            self.assertEqual(plan_fingerprint.fingerprint(pins_51b4085_rows(), name), parts, name)
            self.assertEqual([k for k, v in plan_fingerprint.fingerprint(PINS, name).items() if v != parts[k]
                              and not k.startswith('server:')], [], name)

    def test_the_entry_script(self):
        """DryRun of ctx24 / ctx25 names their settings; a plan whose size the sealed Qwen launcher lacks (ctx25 on
        4f5f4e7) prints NOT_SUBMITTABLE and a real submission of it is refused before any connection; on a pin that
        serves 25 GiB (integrate-1026 cf2b28d) both are submittable."""
        entry = ROOT / 'run_one_connection.ps1'
        run = lambda *args: subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                            str(entry)] + list(args), capture_output=True, text=True, timeout=300)
        for name, gib in (('ctx24', 24), ('ctx25', 25)):
            phrase = c.cfg(PINS)['plans'][name]['authorization']
            value = run('-Plan', name, '-AfterJob', '1234567', '-Authorize', '-Authorization', phrase, '-DryRun')
            self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
            for item in ('servers=goal262:goal_query:32tasks:120steps:3900s:batch4,obs262:observation_query:32tasks:'
                         '120steps:2280s ', 'time=02:27:00 ', 'role=validation ', 'after_job=1234567 ',
                         'gpu_hours_limit=2.45 gpu_hours_expected=1.86 ', 'memory_need_gib=105.29 memory_spare_gib=5.61 ',
                         ' plan_server=own kv_cache_gib=%d max_model_len=262144 plan_dram_arena_gib=70 ' % gib):
                self.assertIn(item, value.stdout, name)
            # NOT_SUBMITTABLE exactly while the pin's Qwen launcher lacks the plan's size (ctx25 on 4f5f4e7; none on
            # integrate-1026 cf2b28d, which serves 25)
            self.assertEqual('NOT_SUBMITTABLE blocked_plan=' + name in value.stdout,
                             c.plan_blocked(PINS, name) is not None, value.stdout)
            self.assertEqual(c.plan_blocked(PINS, name) is None, gib in PINS['runtime_features'][c.QWEN_KV_KEY])
        if c.plan_blocked(PINS, 'ctx25'):  # (only while blocked: an unblocked plan would connect)
            phrase = c.cfg(PINS)['plans']['ctx25']['authorization']
            value = run('-Plan', 'ctx25', '-Authorize', '-Authorization', phrase)
            self.assertNotEqual(value.returncode, 0)
            self.assertIn('is blocked', value.stdout + value.stderr)
            self.assertNotIn('PLANNED_STAGE', value.stdout)


class VllmGoalOverlap(unittest.TestCase):
    """integrate-1030 (2026-10-30): plan vllm25 runs one server, goalvllm = ctx25's goal262 (goal_query, overlap 16, the
    0.6B rewriter, 25 GiB, 262,144, the 70-GiB arena, every switch) with the Goal rewriter on vLLM
    (--goal-rewriter-engine vllm, lm-format-enforcer, 3 GiB of rewriter KV: the plan's own setting) and --goal-overlap,
    and without --goal-rewriter-batch / -tokens; no earlier plan names either option, and its server checks
    (goal-rewriter-engine, goal-overlap, the goal_forward frame) refuse their records on every other server."""
    ADDED = ['--goal-rewriter-engine', 'vllm', '--goal-rewriter-guided-backend', 'lm-format-enforcer',
             '--goal-rewriter-vllm-kv-gib', '3', '--goal-rewriter-batch-invariant', '--goal-overlap']
    ENGINE = dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3, kv_cache_memory_bytes=3 * c.GIB,
                  batch_invariant=True, schema_unique_items=True, gpu_memory_utilization=0.05,
                  startup_check='goal rewriter vllm child: startup check passed: the guided reply in ')
    NEW = c.ENGINE_SERVER_KEYS + (c.ENGINE_BI_KEY, c.GOAL_OVERLAP_KEY)

    def test_goalvllm_is_ctx25s_goal_server_with_the_two_options(self):
        (vllm,) = c.plan_servers(PINS, 'vllm25')
        goal = c.plan_servers(PINS, 'ctx25')[0]
        rest = lambda s: {k: v for k, v in s.items() if k not in ('label', c.BATCH_KEY, c.BATCH_TOKENS_KEY) + self.NEW}
        self.assertEqual(rest(vllm), rest(goal))
        self.assertEqual([vllm[k] for k in ('label', c.BATCH_KEY, c.BATCH_TOKENS_KEY) + self.NEW],
                         ['goalvllm', None, None, 'vllm', 'lm-format-enforcer', 3, True, True])
        self.assertEqual(c.plan_settings(PINS, 'vllm25'), c.plan_settings(PINS, 'ctx25'))
        args = c.launch_arguments(PINS, goal, REWRITER)
        at = args.index('--goal-rewriter-batch')  # goal262's batch 4 (its cap 16384 is the sealed default: not passed)
        self.assertEqual(args[at:at + 2], ['--goal-rewriter-batch', '4'])
        self.assertEqual(c.launch_arguments(PINS, vllm, REWRITER), args[:at] + args[at + 2:] + self.ADDED)
        self.assertEqual(c.launch_arguments(PINS, vllm, REWRITER, rewriter=False),
                         c.launch_arguments(PINS, goal, REWRITER, rewriter=False))
        self.assertIn('--admission-overlap', args)
        for name in PLAN_ORDER:  # no earlier plan names either option (smoke, goalt, smoket: vllm25's)
            for s in c.plan_servers(PINS, name) if name not in ('vllm25', 'smoke', 'goalt', 'smoket', 'goalti',
                                                                'goal64', 'goalh2a', 'goalh2s', 'goal64a', 'goal64ak24', 'gdngoal', 'gdngoalon',
                                                                'keepedgegoal', 'keepedgegoalov', 'gdngoalp',
                                                                'gdngoalpi', 'sess24', 'sess24i', 'keepedgegoalp',
                                                                'keepedgegoalpi', 'sess24r', 'sess24ri') else ():
                # goal64 (11-12): goalti's server; goalh2a / goalh2s (11-16): goalt's on the second half; goal64a /
                # goal64ak24 / gdngoal / gdngoalon (11-16 phase B): goal64's
                self.assertFalse([k for k in self.NEW if k in s], name)
                if s['kind'] == 'runtime':
                    self.assertFalse(set(c.launch_arguments(PINS, s, REWRITER)) & set(self.ADDED[0::2] + ['vllm']), name)
                    self.assertIsNone(c.engine_settings(PINS, s))
                    self.assertFalse(c.goal_overlap_on(PINS, s))
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m/qwen3-0.6b', config_sha256='0' * 64)
        mine, theirs = (c.server_expectation(PINS, s, gate, 'm', ['t0']) for s in (vllm, goal))
        self.assertEqual(mine.pop(c.ENGINE_KEY), self.ENGINE)
        self.assertEqual(mine.pop(c.GOAL_OVERLAP_KEY), PINS['runtime_features'][c.GOAL_OVERLAP_KEY])
        self.assertEqual(mine.pop('admission'), dict(theirs.pop('admission'), goal_overlap=True))
        self.assertEqual([theirs.pop(k) for k in ('goal_rewriter_batch', 'goal_rewriter_batch_tokens', 'label')] +
                         [mine.pop(k) for k in ('goal_rewriter_batch', 'goal_rewriter_batch_tokens', 'label')],
                         [4, 16384, 'goal262', None, None, 'goalvllm'])
        self.assertEqual(mine, theirs)  # every other expectation: ctx25's goal server's
        self.assertEqual(c.server_admission(PINS, goal), c.admission_record(['admission_overlap=16']))

    def test_time_memory_cost_and_phrase(self):
        """The plan's own time model with the worst case kept: cap 3,900 s = ctx25's goal cap (job 3461703's goal262 pool
        ran 3,170 s with the HF rewriter), 840 s of job reserves + 900 s of server reserves + the cap + 300 s for the
        vLLM child's start before READY = 5,940 s (01:39:00); E = goal262's 3,723 s to its checks + 350 s = ~4,100 s."""
        fit = c.validate_plan(PINS, 'vllm25')
        self.assertEqual([fit[k] for k in ('time_seconds', 'worst_case_seconds', 'spare_seconds', 'with_eval_wait_seconds',
                                           'guaranteed_seconds', 'cap_after_wait_fits', 'fits')],
                         [5940, 5640, 300, 4570, 5820, True, True])
        cost = c.plan_cost(PINS, 'vllm25')
        self.assertEqual([cost[k] for k in ('expected_seconds', 'billing_factor', 'gpu_hours_limit', 'gpu_hours_expected',
                                            'limit_over_expected')], [4100, 1, 1.65, 1.139, 1.449])
        mine, theirs = c.memory_fit(PINS, 'vllm25'), c.memory_fit(PINS, 'ctx25')
        self.assertEqual(mine, theirs)  # 73.29 + 32 = 105.29 of 110.90 GiB, 5.61 spare
        self.assertEqual((mine['need_gib'], mine['headroom_gib'], mine['fits']), (105.29, 5.61, True))
        plan, (vllm,), goal = c.cfg(PINS)['plans']['vllm25'], c.plan_servers(PINS, 'vllm25'), server('ctx25')
        self.assertEqual((plan['partition'], plan['time'], plan['role'], plan['eval_wait_seconds'], plan.get('variant_of')),
                         ('ghx4', '01:39:00', 'validation', 600, None))
        self.assertEqual((vllm['agent_cap_seconds'], vllm['agent_min_seconds'], vllm['step_limit'], vllm['tasks']),
                         (goal['agent_cap_seconds'], goal['agent_min_seconds'], 120, 32))
        self.assertEqual(plan['authorization'], 'AUTHORIZE_SWE32_VLLM25_32TASKS_16SESSIONS_262K_KV25_VLLM_REWRITER_GOAL_'
                                                'OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_99MIN_NO_RETRY')
        self.assertEqual((PINS['slurm']['mem'], PINS['slurm']['mail_type'], PINS['slurm']['mail_user']),
                         ('113560M', 'BEGIN,END,FAIL', 'cao58@illinois.edu'))
        self.assertIsNone(c.plan_blocked(PINS, 'vllm25'))
        attention = c.attention_fit(PINS, c.LAYOUT, 16, vllm)
        self.assertEqual((attention['fa_blocks'], attention['one_turn_blocks'], attention['fits']), (20706, 17873, True))

    def test_refusals_where_serve_refuses(self):
        def refused(change, pattern, plan='vllm25'):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, plan)
        v = lambda pins: pins['config']['plans']['vllm25']['servers'][0]
        refused(lambda p: v(p).update(goal_rewriter_batch=4, goal_rewriter_batch_tokens=16384), 'excludes')
        refused(lambda p: p['config']['policies']['goal_query'].update(admission=[]), 'K >= 2')
        refused(lambda p: v(p).update(goal_rewriter_engine='hf'), 'all three named')
        refused(lambda p: v(p).pop(c.ENGINE_KV_KEY), 'all three named')
        refused(lambda p: v(p).pop(c.ENGINE_BACKEND_KEY), 'all three named')
        for bad in (1, 17, '3', 3.0, None):
            refused(lambda p, bad=bad: v(p).update(goal_rewriter_vllm_kv_gib=bad), 'kv-gib')
        refused(lambda p: v(p).update(goal_rewriter_guided_backend='guidance'), 'guided-backend')
        for bad in (1, 'true', None):
            refused(lambda p, bad=bad: v(p).update(goal_rewriter_batch_invariant=bad), 'must be true or absent')
        refused(lambda p: p['runtime_features'][c.ENGINE_KEY].update(batch_invariant_arg=None),
                'needs a sealed launcher with --goal-rewriter-batch-invariant')
        refused(lambda p: [v(p).pop(k) for k in c.ENGINE_SERVER_KEYS], 'applies to --goal-rewriter-engine vllm only')
        refused(lambda p: p['config']['plans']['stock']['servers'][0].update(goal_rewriter_batch_invariant=True),
                'stock server', 'stock')
        refused(lambda p: v(p).update(policy='observation_query'), 'goal_query with the Goal rewriter')
        for bad in (False, 1, 'true', None):
            refused(lambda p, bad=bad: v(p).update(goal_overlap=bad), 'goal_overlap must be true')
        refused(lambda p: [v(p).pop(k) for k in c.ENGINE_SERVER_KEYS + (c.ENGINE_BI_KEY,)] and v(p).update(policy='full'),
                '--goal-overlap needs goal_query')
        refused(lambda p: p['config']['plans']['full']['servers'][0].update(goal_overlap=True), 'goal_query', 'full')
        refused(lambda p: p['config']['plans']['obs']['servers'][0].update(
            goal_rewriter_engine='vllm', goal_rewriter_guided_backend='lm-format-enforcer', goal_rewriter_vllm_kv_gib=3),
            'goal_query with the Goal rewriter', 'obs')
        refused(lambda p: p['config']['plans']['stock']['servers'][0].update(goal_overlap=True), 'stock server', 'stock')
        refused(lambda p: p['runtime_features'].update(goal_rewriter_engine=None) or p['config']['plans'].pop('smoke'),
                'no --goal-rewriter-engine')
        refused(lambda p: p['runtime_features'].update(goal_overlap=None), 'no --goal-overlap')
        pins = copy.deepcopy(PINS)  # another KV size or backend in range is the plan's to choose
        v(pins).update(goal_rewriter_vllm_kv_gib=6, goal_rewriter_guided_backend='xgrammar')
        c.validate_plan(pins, 'vllm25')
        self.assertEqual(c.launch_arguments(pins, c.plan_servers(pins, 'vllm25')[0], REWRITER)[-8:],
                         ['--goal-rewriter-engine', 'vllm', '--goal-rewriter-guided-backend', 'xgrammar',
                          '--goal-rewriter-vllm-kv-gib', '6', '--goal-rewriter-batch-invariant', '--goal-overlap'])
        v(pins).pop(c.ENGINE_BI_KEY)  # absent: off, no switch
        self.assertNotIn(c.ENGINE_BI_SWITCH, c.launch_arguments(pins, c.plan_servers(pins, 'vllm25')[0], REWRITER))
        self.assertFalse(c.engine_settings(pins, c.plan_servers(pins, 'vllm25')[0])['batch_invariant'])
        v(pins).update({c.ENGINE_BI_KEY: True})
        self.assertEqual(c.engine_settings(pins, c.plan_servers(pins, 'vllm25')[0]),
                         dict(self.ENGINE, guided_backend='xgrammar', kv_gib=6, kv_cache_memory_bytes=6 * c.GIB,
                              schema_unique_items=False))
        # the build refuses a pin without the switches while vllm25 asks for them; without vllm25 it does not care
        for key, pattern in ((c.ENGINE_KEY, 'goal-rewriter-engine'), (c.GOAL_OVERLAP_KEY, 'goal-overlap')):
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.check_features(dict(PINS['runtime_features'], **{key: None}), PINS['config'])
        config = copy.deepcopy(PINS['config'])
        for name in ('vllm25', 'smoke', 'goalt', 'smoket', 'goalti', 'goal64', 'goalh2a', 'goalh2s', 'goal64a', 'goal64ak24', 'gdngoal', 'gdngoalon',
                     'keepedgegoal', 'keepedgegoalov', 'gdngoalp', 'gdngoalpi', 'sess24', 'sess24i',
                     'keepedgegoalp', 'keepedgegoalpi', 'sess24r', 'sess24ri'):  # (vllm25's)
            del config['plans'][name]
            config['plan_order'].remove(name)
        c.check_features(dict(PINS['runtime_features'], goal_rewriter_engine=None, goal_overlap=None), config)

    def test_the_sealed_features(self):
        features = PINS['runtime_features']
        self.assertEqual(features[c.ENGINE_KEY], dict(
            serve_arg='--goal-rewriter-engine', engine='vllm', config_key='goal_rewriter_engine',
            backend_arg='--goal-rewriter-guided-backend', backends=['lm-format-enforcer', 'xgrammar'],
            default_backend='lm-format-enforcer', without_unique_items=['xgrammar'], kv_arg='--goal-rewriter-vllm-kv-gib',
            kv_gib=[2, 16], default_kv_gib=3, gpu_memory_utilization=0.05, startup_key='startup_ms',
            batch_invariant_arg='--goal-rewriter-batch-invariant',
            startup_check='goal rewriter vllm child: startup check passed: the guided reply in '))
        self.assertEqual(features[c.GOAL_OVERLAP_KEY], dict(
            serve_arg='--goal-overlap', flag='goal_overlap', record_key='admission', frame='goal_forward',
            native=['goal_overlap_ms', 'goal_forward_ms']))
        self.assertEqual(features['admission_flags']['flags'], ['gdn_device_handoff', 'phase1_in_step'])
        with tempfile.TemporaryDirectory() as tmp:  # a runtime without the two modules: neither feature
            self.assertIsNone(c.goal_rewriter_engine_feature(Path(tmp), "'--goal-rewriter-engine'"))
            self.assertIsNone(c.goal_overlap_feature(Path(tmp), "'--goal-overlap'"))

    @staticmethod
    def engine_files(out, record, ready=None, attached=None, model='/m/qwen3-0.6b'):
        startup = dict(selection_contract=dict(goal_rewriter=dict(model=model)))
        if record is not None:
            startup[c.ENGINE_KEY] = record
        (out / 'startup.json').write_text(json.dumps(startup), encoding='utf-8')
        (out / 'ready.json').write_text(json.dumps({} if ready is None else {c.ENGINE_KEY: ready}), encoding='utf-8')
        (out / 'worker-attached.json').write_text(json.dumps({} if attached is None else {c.ENGINE_KEY: attached}),
                                                  encoding='utf-8')

    @staticmethod
    def record(engine, model='/m/qwen3-0.6b'):
        """goal_rewriter_vllm.record's shape (the keys the checks read)."""
        return dict(engine=engine['engine'], guided_backend=engine['guided_backend'],
                    schema_unique_items=engine['schema_unique_items'],
                    kv_cache_memory_bytes=engine['kv_cache_memory_bytes'],
                    gpu_memory_utilization=engine['gpu_memory_utilization'], batch_invariant=engine['batch_invariant'],
                    engine_args=dict(model=model, tokenizer=model, kv_cache_memory_bytes=engine['kv_cache_memory_bytes'],
                                     gpu_memory_utilization=engine['gpu_memory_utilization'],
                                     structured_outputs_config=dict(backend=engine['guided_backend']), max_num_seqs=16))

    PASSED = 'goal rewriter vllm child: startup check passed: the guided reply in 17 tokens, 412 ms\n'
    READY = 'goal rewriter vllm child: engine ready: structured outputs lm-format-enforcer, batch invariant True\n'

    def test_the_server_check_of_the_vllm_rewriter(self):
        expect = {c.ENGINE_KEY: self.ENGINE}
        good = self.record(self.ENGINE)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            log = out / 'server.log'
            log.write_text('vLLM lines\n' + self.PASSED + self.READY, encoding='utf-8')
            self.engine_files(out, good, dict(good, startup_ms=812.5), dict(startup_ms=812.5))
            value = swe_checks.check_goal_rewriter_engine(out, expect, log)
            self.assertEqual((value['goal_rewriter_engine'], value['startup_ms'], value['kv_cache_memory_bytes'],
                              value['batch_invariant'], value['startup_checks']),
                             (dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3), 812.5, 3 * c.GIB,
                              True, 1))
            # c07d42a: the child's startup-check line, once per start, no refusal (a restart checks again)
            log.write_text(self.PASSED + self.READY + self.PASSED + self.READY, encoding='utf-8')
            self.assertEqual(swe_checks.check_goal_rewriter_engine(out, expect, log)['startup_checks'], 2)
            for bad in ('', self.READY, self.PASSED, self.PASSED + self.READY + self.READY,
                        self.PASSED + self.READY + 'goal rewriter vllm child: refused: the guided decoding (x) failed\n'):
                log.write_text(bad, encoding='utf-8')
                with self.assertRaises(swe_checks.CheckFailed, msg=bad):
                    swe_checks.check_goal_rewriter_engine(out, expect, log)
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_goal_rewriter_engine(out, expect)  # no server log
            log.write_text(self.PASSED + self.READY, encoding='utf-8')
            self.assertEqual(c.engine_record_differences(good, self.ENGINE, '/m/qwen3-0.6b'), [])
            args = lambda **kw: dict(good, engine_args=dict(good['engine_args'], **kw))
            started, start = dict(good, startup_ms=1.0), dict(startup_ms=1.0)
            for record, ready, attached in (
                    (dict(good, kv_cache_memory_bytes=4 * c.GIB), started, start),
                    (dict(good, guided_backend='xgrammar'), started, start),
                    (dict(good, batch_invariant=not good['batch_invariant']), started, start),
                    (dict(good, extra=1), started, start),
                    (args(model='/other'), started, start), (args(tokenizer='/other'), started, start),
                    (args(kv_cache_memory_bytes=5), started, start),
                    (args(structured_outputs_config=dict(backend='xgrammar')), started, start),
                    (good, started, dict(startup_ms=2.0)),  # another start time
                    (good, dict(good), start),  # READY without the start
                    (good, started, None),  # the worker-attached record missing
                    (good, None, start),  # READY without the record
                    (good, dict(good, startup_ms=-1.0), dict(startup_ms=-1.0)),
                    (good, started, dict(startup_ms=1.0, x=1)),
                    (None, started, start)):  # startup.json without it
                self.engine_files(out, record, ready, attached)
                with self.assertRaises(swe_checks.CheckFailed, msg=(record, ready, attached)):
                    swe_checks.check_goal_rewriter_engine(out, expect, log)
            self.engine_files(out, good, started, start, model='/other')  # not the pinned rewriter's checkpoint
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_goal_rewriter_engine(out, expect, log)
            self.engine_files(out, None)  # a server without the engine: no key anywhere
            self.assertEqual(swe_checks.check_goal_rewriter_engine(out, {}), dict(goal_rewriter_engine=None))
            for files in ((good, None, None), (None, dict(startup_ms=1.0), None), (None, None, dict(startup_ms=1.0))):
                self.engine_files(out, *files)
                with self.assertRaises(swe_checks.CheckFailed, msg=files):
                    swe_checks.check_goal_rewriter_engine(out, {})

    def natives(self, out, turns):
        """Runtime rows and native files of one session: [(local_turn, goal_forward, frames, extra native fields)]."""
        import smoke_checks as sc
        rows = []
        for t, forward, ops, extra in turns:
            rid = 'r%d' % t
            frames = [dict(operation=op, goal_vector=[1.0] if op == 'goal_forward' else None, drained=True)
                      for op in ops]
            native = dict(local_turn=t, worker_frames=frames, goal_async_ms=0.0, goal_construct_ms=0.2)
            native.update(extra)
            sc.native_path(out, rid).write_text(json.dumps(native), encoding='utf-8')
            rows.append(dict(request_id=rid, session_index=0, turn=t,
                             stage_b=dict(phase1=None if t == 0 else dict(goal_forward=forward))))
        return rows

    def test_the_server_check_of_the_goal_overlap(self):
        import smoke_checks as sc
        head, tail = ['prepare', 'phase1', 'phase1_drain'], ['reserve', 'allocation_update', 'complete']
        frames = lambda *ops: [dict(operation=op) for op in ops]
        self.assertEqual(sc.frame_ops(frames(*head, 'goal_forward', *tail), 1, goal_overlap=True), 1)
        self.assertEqual(sc.frame_ops(frames(*head, *tail), 1, goal_overlap=True), 1)
        with self.assertRaises(sc.CheckFailed):  # a flag-off server never has the frame
            sc.frame_ops(frames(*head, 'goal_forward', *tail), 1)
        with self.assertRaises(sc.CheckFailed):  # a local turn 0 has no phase 1 and no Goal forward
            sc.frame_ops(frames('prepare', 'goal_forward', *tail), 0, goal_overlap=True)
        with self.assertRaises(sc.CheckFailed):  # only after phase 1's drain
            sc.frame_ops(frames('prepare', 'phase1', 'goal_forward', 'phase1_drain', *tail), 1, goal_overlap=True)
        on = {c.GOAL_OVERLAP_KEY: PINS['runtime_features'][c.GOAL_OVERLAP_KEY]}
        forward = (*head, 'goal_forward', *tail)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            good = [(0, False, ('prepare', *tail), dict(goal_overlap_ms=0.3)),
                    (1, True, forward, dict(goal_overlap_ms=0.4, goal_forward_ms=41.0)),
                    (2, False, (*head, *tail), dict(goal_overlap_ms=0.5))]
            value = swe_checks.check_goal_overlap(out, self.natives(out, good), on)
            self.assertEqual((value['turns'], value['goal_forwards'], value['wide']), (3, 1, []))
            for bad in ([good[0], (1, True, forward, dict(goal_overlap_ms=0.4))],  # no goal_forward_ms
                        [good[0], (1, True, (*head, *tail), dict(goal_overlap_ms=0.4, goal_forward_ms=4.0))],
                        [good[0], (1, False, forward, dict(goal_overlap_ms=0.4, goal_forward_ms=4.0))],
                        [good[0], (1, False, (*head, *tail), dict(goal_overlap_ms=0.4, goal_forward_ms=4.0))],
                        [(0, False, ('prepare', *tail), {})],  # no goal_overlap_ms
                        [(0, False, ('prepare', *tail), dict(goal_overlap_ms=-1.0))]):
                with self.assertRaises(swe_checks.CheckFailed, msg=bad):
                    swe_checks.check_goal_overlap(out, self.natives(out, bad), on)
            with self.assertRaises(swe_checks.Finding):  # a construction far outside start .. resume
                swe_checks.check_goal_overlap(out, self.natives(out, [
                    (0, False, ('prepare', *tail), dict(goal_overlap_ms=0.1, goal_construct_ms=900.0))]), on)
            # a server without the switch: neither field, no goal_forward frame
            plain = [(0, False, ('prepare', *tail), {}), (1, True, (*head, *tail), {})]
            self.assertEqual(swe_checks.check_goal_overlap(out, self.natives(out, plain), {})['turns'], 0)
            for bad in ([(0, False, ('prepare', *tail), dict(goal_overlap_ms=0.3))],
                        [(1, True, forward, {})], [(1, True, (*head, *tail), dict(goal_forward_ms=1.0))]):
                with self.assertRaises(swe_checks.CheckFailed, msg=bad):
                    swe_checks.check_goal_overlap(out, self.natives(out, bad), {})

    def test_the_entry_script(self):
        entry = ROOT / 'run_one_connection.ps1'
        phrase = c.cfg(PINS)['plans']['vllm25']['authorization']
        value = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(entry),
                                '-Plan', 'vllm25', '-AfterJob', '1234567', '-Authorize', '-Authorization', phrase,
                                '-DryRun'], capture_output=True, text=True, timeout=300)
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('servers=goalvllm:goal_query:32tasks:120steps:3900s:vllm3gib:overlap ', 'time=01:39:00 ',
                     'role=validation ', 'expected_seconds=4100 ', 'gpu_hours_limit=1.65 gpu_hours_expected=1.14 ',
                     'spare_seconds=300 ', 'memory_need_gib=105.29 memory_spare_gib=5.61 ', 'goal_rewriter_batch=off ',
                     ' plan_server=own kv_cache_gib=25 max_model_len=262144 plan_dram_arena_gib=70 ',
                     ' goal_rewriter_engine=goalvllm:vllm:lm-format-enforcer:3:batch-invariant goal_overlap=goalvllm',
                     'mem=113560M ', 'mail_type=BEGIN,END,FAIL '):
            self.assertIn(item, value.stdout)
        self.assertNotIn('NOT_SUBMITTABLE', value.stdout)
        other = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(entry),
                                '-Plan', 'ctx25', '-Authorize', '-Authorization',
                                c.cfg(PINS)['plans']['ctx25']['authorization'], '-DryRun'],
                               capture_output=True, text=True, timeout=300)
        self.assertEqual(other.returncode, 0, other.stdout + other.stderr)
        self.assertNotIn('goal_rewriter_engine=', other.stdout)
        self.assertNotIn('goal_overlap=', other.stdout)


class ChildEngineLog(unittest.TestCase):
    """vllm25 r2 (job 3500565) ended SWE32_FAIL on two false positives: the vLLM Goal rewriter child is a second vLLM
    engine (its own EngineCore, pid 33376) logging into the server log of the main engine (pid 15428), and
    context-limits read every engine's max_seq_len ([16384, 262144]) while capture-sizes (and the cuda-graphs finding
    CUDA_GRAPH_CONFIG_DIFFERS) counted the child's PIECEWISE bar (5/5) instead of the main engine's 7/7. The checks now
    read the main engine's lines only (smoke_checks.engine_logs) and record the child's own config as information.
    fixtures/server-log-child-engine-job3500565.log: that log's vLLM startup and shutdown lines (both engine config
    lines trimmed to the fields the checks read; the capture bars verbatim with their carriage returns; no request,
    admission or REMOTE_RA_LIVE line)."""
    RAW = (ROOT / 'fixtures' / 'server-log-child-engine-job3500565.log').read_bytes().decode('utf-8')
    MAIN, CHILD = '(EngineCore pid=15428) ', '(EngineCore pid=33376) '
    CHILD_MODEL = '/projects/bcjw/jgerard1/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/' \
                  'c1899de289a04d12100db370d81485cdf75e47ca'
    PIECEWISE, FULL = 'mixed prefill-decode, PIECEWISE', 'decode, FULL'

    def expect(self):
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model=self.CHILD_MODEL, config_sha256='0' * 64)
        return c.server_expectation(PINS, server('vllm25'), gate, 'm', ['t0'])

    def files(self, out, expect, engine=True):
        """startup / ready / worker-attached.json of a goalvllm server: the keys the three checks read."""
        import smoke_checks as sc
        L, T = expect['max_model_len'], expect['max_tokens']
        layout, context = expect['layout'], dict(max_model_len=L, max_tokens=T)
        page_bytes = layout['page_tokens'] * sc.FA_TOKEN_BYTES
        chunk = sc.ARENA_CHUNK_BYTES // page_bytes
        startup = dict(cuda_graphs='full', effective_server_config=dict(max_model_len=L), max_tokens=T,
                       dram_arena_pages=layout['dram_arena_pages'])
        if engine:
            startup[c.ENGINE_KEY] = dict(engine='vllm', engine_args=dict(
                model=self.CHILD_MODEL, tokenizer=self.CHILD_MODEL, max_model_len=16384,
                compilation_config=dict(cudagraph_capture_sizes=[1, 2, 4, 8, 16])))
        ready = dict(effective_server_config=dict(max_model_len=L, enforce_eager=False,
                                                  compilation_config=dict(sc.COMPILATION['full'])), context=context)
        sizes = sc.capture_sizes(expect['max_num_seqs'])
        receipt = dict(context=context, tensor_geometry=[dict(kind='attention', group=3)],
                       dram_arena=dict(max_pages=-(-layout['arena_pages'] // chunk) * chunk, page_bytes=page_bytes,
                                       chunk_slots=chunk),
                       cuda_graphs=dict(mode='full', compilation_mode='VLLM_COMPILE', cudagraph_mode='FULL_AND_PIECEWISE',
                                        capture_sizes=sizes, max_capture_size=sizes[-1], backend='inductor',
                                        custom_ops=['none'], splitting_ops=list(sc.REQUIRED_SPLITTING_OPS),
                                        seq_lens_bound_groups=[3], probe_forward='eager'))
        for name, value in (('startup', startup), ('ready', ready), ('worker-attached', dict(receipt=receipt))):
            (out / (name + '.json')).write_text(json.dumps(value), encoding='utf-8')

    @staticmethod
    def edit(text, prefix, old, new):
        """Replace `old` by `new` in the one line of `text` that starts with `prefix` and contains `old`."""
        lines = text.split('\n')
        at = [i for i, line in enumerate(lines) if line.startswith(prefix) and old in line]
        assert len(at) == 1, (prefix, old, at)
        lines[at[0]] = lines[at[0]].replace(old, new)
        return '\n'.join(lines)

    def run_checks(self, text, engine=True):
        """{check: ('PASS' | 'FINDING' | 'FAIL', detail or message)} of the three engine-side checks on a log."""
        import smoke_checks as sc
        expect = self.expect()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.files(out, expect, engine)
            log = out / 'server.log'
            log.write_bytes(text.encode('utf-8'))  # the bars' carriage returns kept: read as the job reads it
            got = {}
            for name, run in (('context-limits', lambda: sc.check_context_limits(out, expect, log, [])),
                              ('cuda-graphs', lambda: sc.check_cuda_graphs(out, expect, log)),
                              ('capture-sizes', lambda: sc.check_capture_sizes(out, expect, log))):
                try:
                    got[name] = ('PASS', run())
                except sc.Finding as exc:
                    got[name] = ('FINDING', exc.detail)
                except sc.CheckFailed as exc:
                    got[name] = ('FAIL', str(exc))
            return got

    def child_info(self, **changes):
        info = dict(pid=33376, model=self.CHILD_MODEL, max_seq_len=[16384], capture_sizes=[1, 2, 4, 8, 16],
                    capture_counts={self.PIECEWISE: [5, 5], self.FULL: [5, 5]})
        info.update(changes)
        return [info]

    def test_the_fixture_is_job_3500565s_structure(self):
        import smoke_checks as sc
        text = self.RAW.replace('\r\n', '\n').replace('\r', '\n')  # as read_text reads it: the bars' states split
        bars = [line for line in text.split('\n') if 'Capturing CUDA graphs' in line]
        self.assertEqual((len(bars), [b for b in bars if b.startswith('(')]), (18, []))  # no bar state has a pid
        # what the job's checks read from the whole log: both engines' values (the two false positives)
        self.assertEqual(sorted({int(v) for v in sc.MAX_SEQ_LEN.findall(text)}), [16384, 262144])
        self.assertEqual(sc.capture_counts(text), {self.PIECEWISE: (5, 5), self.FULL: (5, 5)})
        self.assertEqual(sc.engine_config(text)['cudagraph_capture_sizes'], [1, 2, 4, 8, 16, 24, 32])  # the first

    def test_main_engine_only(self):
        """A server without a vLLM child (the child's lines left out): one engine, the whole log, as before."""
        main = '\n'.join(line for line in self.RAW.split('\n') if line.startswith('(APIServer') or
                         line.startswith(self.MAIN))
        got = self.run_checks(main, engine=False)
        self.assertEqual({k: v[0] for k, v in got.items()}, dict.fromkeys(got, 'PASS'), got)
        self.assertEqual(got['context-limits'][1]['engine_max_seq_len'], 262144)
        self.assertEqual(got['capture-sizes'][1]['captures'], {self.PIECEWISE: [7, 7], self.FULL: [5, 5]})
        self.assertFalse([k for k, v in got.items() if 'rewriter_engines' in v[1]])
        self.assertEqual(self.run_checks(main), got)  # the record alone splits nothing
        # one engine config line: the whole log, as before, whatever else it holds (a stray pid's lines are read)
        stray = main + '\n(EngineCore pid=1) INFO 11-02 02:15:00 [registry.py:134] a line of another pid\n'
        self.assertEqual(self.run_checks(stray, engine=False), got)

    def test_main_engine_and_the_rewriter_child_pass(self):
        got = self.run_checks(self.RAW)
        self.assertEqual({k: v[0] for k, v in got.items()}, dict.fromkeys(got, 'PASS'), got)
        self.assertEqual(got['context-limits'][1]['engine_max_seq_len'], 262144)
        self.assertEqual(got['capture-sizes'][1]['captures'], {self.PIECEWISE: [7, 7], self.FULL: [5, 5]})
        self.assertEqual(got['cuda-graphs'][1]['capture_counts'], {self.PIECEWISE: [7, 7], self.FULL: [5, 5]})
        self.assertEqual(got['cuda-graphs'][1]['engine_config']['cudagraph_capture_sizes'], [1, 2, 4, 8, 16, 24, 32])
        for name, (_, detail) in got.items():
            self.assertEqual(detail['rewriter_engines'], self.child_info(), name)

    def test_the_childs_own_lines_are_information(self):
        text = self.edit(self.RAW, self.CHILD, 'max_seq_len=16384', 'max_seq_len=8192')
        text = self.edit(text, self.CHILD, "'cudagraph_capture_sizes': [1, 2, 4, 8, 16]",
                         "'cudagraph_capture_sizes': [1, 2, 4]")
        text = self.edit(text, self.CHILD + '\rCapturing CUDA graphs (mixed', '5/5', '3/5')
        got = self.run_checks(text)
        self.assertEqual({k: v[0] for k, v in got.items()}, dict.fromkeys(got, 'PASS'), got)
        for name, (_, detail) in got.items():
            self.assertEqual(detail['rewriter_engines'], self.child_info(
                max_seq_len=[8192], capture_sizes=[1, 2, 4], capture_counts={self.PIECEWISE: [3, 5], self.FULL: [5, 5]}),
                name)

    def test_the_main_engines_lines_stay_checked(self):
        status = lambda text: {k: v[0] for k, v in self.run_checks(text).items()}
        bar = self.edit(self.RAW, self.MAIN + '\rCapturing CUDA graphs (mixed', '| 7/7', '| 6/7')
        self.assertEqual(status(bar), {'context-limits': 'PASS', 'cuda-graphs': 'FAIL', 'capture-sizes': 'FAIL'})
        sizes = self.edit(self.RAW, self.MAIN, "'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32]",
                          "'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24]")
        self.assertEqual(status(sizes)['capture-sizes'], 'FAIL')
        seq = self.edit(self.RAW, self.MAIN, 'max_seq_len=262144', 'max_seq_len=131072')
        got = self.run_checks(seq)
        self.assertEqual(got['context-limits'][0], 'FAIL')
        self.assertIn('max_seq_len [131072], expected 262144', got['context-limits'][1])
        # a main engine whose capture differs from vLLM's defaults is still the finding
        full = self.edit(self.RAW, self.MAIN + '\rCapturing CUDA graphs (decode', '| 5/5', '| 4/4')
        got = self.run_checks(full)
        self.assertEqual((got['cuda-graphs'][0], got['capture-sizes'][0]), ('FINDING', 'FAIL'))
        self.assertEqual(got['cuda-graphs'][1]['differs']['capture_counts']['got'], {self.PIECEWISE: [7, 7],
                                                                                    self.FULL: [4, 4]})

    def test_engines_and_lines_that_cannot_be_attributed_fail(self):
        def failed(text, pattern, engine=True):
            got = self.run_checks(text, engine)
            self.assertEqual({k: v[0] for k, v in got.items()}, dict.fromkeys(got, 'FAIL'), got)
            for name, (_, message) in got.items():
                self.assertRegex(message, pattern, name)
        lines = self.RAW.split('\n')
        child = [line for line in lines if line.startswith(self.CHILD + 'INFO') and 'Initializing a V1' in line][0]
        third = child.replace('pid=33376', 'pid=40001').replace(self.CHILD_MODEL, '/projects/x/another-model')
        failed(self.RAW + third + '\n', 'vLLM engines that are not the Goal rewriter child')
        failed(self.RAW, 'vLLM engines that are not the Goal rewriter child', engine=False)  # no record: unknown
        failed(self.RAW + '(EngineCore pid=40002) INFO 11-02 02:20:00 [core.py:1] a fourth process\n',
               r'pid\(s\) \[40002\] log lines but no engine config line')
        failed(self.RAW + child + '\n', 'more than one engine config line')
        # an unprefixed capture line after both engines' startups: in no engine's startup
        failed(self.RAW + 'Capturing CUDA graphs (decode, FULL): 100%| 5/5\n', 'cannot be attributed')
        # the main engine's startup never ended (no "init engine ... took" line): the child's bar states lie in both
        unended = '\n'.join(line for line in lines if not (line.startswith(self.MAIN) and 'init engine' in line))
        failed(unended, 'cannot be attributed')
        # the child's startup never ended: its startup reaches the log's end, which no other startup overlaps
        unended = '\n'.join(line for line in lines if not (line.startswith(self.CHILD) and 'init engine' in line))
        self.assertEqual({k: v[0] for k, v in self.run_checks(unended).items()},
                         dict.fromkeys(('context-limits', 'cuda-graphs', 'capture-sizes'), 'PASS'))


class Stock25(unittest.TestCase):
    """2026-11-02: plan stock25 runs one server, stock262 = plan stock's server (config.stock: vLLM 0.24.0's defaults,
    async scheduling on, prefix caching, --gdn-prefill-backend triton, --language-model-only, the client-side generation
    contract) at the runtime servers' --max-model-len 262144 and 25 GiB of KV (config.plans.stock25.stock), with a
    preempted request the finding STOCK_PREEMPTED instead of a failure; every other plan, plan stock among them, runs
    config.stock unchanged."""
    OWN = dict(kv_cache_gib=25, max_model_len=262144, preemption='finding')

    def test_stock262_is_plan_stocks_server_at_the_runtime_context(self):
        (mine,) = c.plan_servers(PINS, 'stock25')
        self.assertEqual(mine.pop(c.STOCK_SETTINGS_KEY), self.OWN)
        self.assertEqual(mine.pop('label'), 'stock262')
        self.assertEqual(dict(mine, label='stock'), server('stock'))  # every other server field: plan stock's
        mine = server('stock25')
        self.assertEqual(c.stock_config(PINS, mine), dict(c.cfg(PINS)['stock'], **self.OWN))
        own = ('stock25', 'stock25x', 'stock64t', 'stock32t', 'stock32ti') + tuple(KEEP1116 + H2STOCK + PBSTOCK) \
            + ('stockkeep150k24',)  # (2026-11-17)
        for name in PLAN_ORDER:  # every other plan but stock25x (stock25's): config.stock itself
            self.assertEqual(name in own, 'stock' in c.cfg(PINS)['plans'][name], name)
            if name in own[2:]:  # (2026-11-16: the keep plans add the kept history; stockkeepsync32t async off)
                extra = dict(history='reasoning_kept', **(dict(async_scheduling=False) if name == 'stockkeepsync32t'
                                                          else {}),
                             **(dict(max_num_seqs=24) if name in ('stockkeep64k24', 'stockkeep150k24') else {})) \
                    if name in KEPT else {}
                self.assertEqual(c.cfg(PINS)['plans'][name]['stock'], dict(self.OWN, **extra), name)
            for s in c.plan_servers(PINS, name) if name not in own else ():
                self.assertNotIn(c.STOCK_SETTINGS_KEY, s, name)
                self.assertIs(c.stock_config(PINS, s), c.cfg(PINS)['stock'], name)
        argv, plain = c.stock_argv(PINS, 8123, mine), c.stock_argv(PINS, 8123)
        self.assertEqual(plain, c.stock_argv(PINS, 8123, server('stock')))
        self.assertEqual(len(argv), len(plain))
        self.assertEqual([(a, b) for a, b in zip(argv, plain) if a != b],
                         [('262144', '131072'), (str(25 * c.GIB), str(32 * c.GIB))])
        # the context, KV bytes, seqs, batched tokens and seed of ctx25's / vllm25's servers
        value = lambda flag: argv[argv.index(flag) + 1]
        goal = c.server_cfg(PINS, server('ctx25'))
        self.assertEqual(c.server_cfg(PINS, server('vllm25')), goal)
        self.assertEqual((value('--max-model-len'), value('--kv-cache-memory-bytes'), value('--max-num-seqs'),
                          value('--max-num-batched-tokens'), value('--seed')),
                         (str(goal['max_model_len']), str(goal['kv_cache_gib'] * c.GIB), str(goal['max_running_sessions']),
                          str(c.MAX_BATCHED), '42'))
        expected, before = c.stock_expected(PINS, 8123, mine), c.stock_expected(PINS, 8123)
        self.assertEqual({k: (expected[k], before[k]) for k in expected if expected[k] != before[k]},
                         dict(max_model_len=(262144, 131072), kv_cache_memory_bytes=(25 * c.GIB, 32 * c.GIB)))
        self.assertEqual(sorted(expected), sorted(before))
        for flag in ('--no-async-scheduling', '--enforce-eager', '--decode-fastpath', '--connector-pool-gib',
                     '--kv-transfer-config', '--block-size'):
            self.assertNotIn(flag, argv)
        self.assertEqual(argv.count('--language-model-only'), 1)
        # the agents get plan stock's model config and pool settings (only the label differs)
        self.assertEqual(c.model_override_text(PINS, mine, 23004), c.model_override_text(PINS, server('stock'), 23004))
        spec = c.pool_spec(PINS, mine, 23004, 1800000000)[1]
        self.assertEqual(dict(spec, label='stock'), c.pool_spec(PINS, server('stock'), 23004, 1800000000)[1])

    def test_time_memory_cost_and_phrase(self):
        """Plan stock's cap 1,800 s (1.25 x stock r2 job 3429193's 1,445-s wall) with every reserve and the full 600-s
        evaluation wait (the job runs behind vllm25 r2, whose evaluation may hold the lock): 540 + 600 + 1,800 + 300 =
        3,240 s (00:54:00, cap_after_wait_fits); E = ~2,360 s (3429193's ~1,780 s + 15 % of its wall + the lock wait)."""
        fit = c.validate_plan(PINS, 'stock25')
        self.assertEqual([fit[k] for k in ('time_seconds', 'worst_case_seconds', 'spare_seconds', 'with_eval_wait_seconds',
                                           'guaranteed_seconds', 'cap_after_wait_fits', 'finish_seconds', 'fits')],
                         [3240, 3060, 180, 2840, 3240, True, 180, True])
        stock_fit = c.plan_fit(PINS, 'stock')
        self.assertEqual([stock_fit[k] for k in ('worst_case_seconds', 'with_eval_wait_seconds', 'guaranteed_seconds')],
                         [fit['worst_case_seconds'], fit['with_eval_wait_seconds'], fit['guaranteed_seconds']])
        cost = c.plan_cost(PINS, 'stock25')
        self.assertEqual([cost[k] for k in ('expected_seconds', 'billing_factor', 'gpu_hours_limit', 'gpu_hours_expected',
                                            'limit_over_expected')], [2360, 1, 0.9, 0.656, 1.373])
        mine = c.memory_fit(PINS, 'stock25')
        self.assertEqual(mine, c.memory_fit(PINS, 'stock'))  # stock pins nothing: 32 GiB need, 78.90 spare
        self.assertEqual((mine['need_gib'], mine['headroom_gib'], mine['fits']), (32.0, 78.9, True))
        plan, (stock262,) = c.cfg(PINS)['plans']['stock25'], c.plan_servers(PINS, 'stock25')
        self.assertEqual((plan['partition'], plan['time'], plan['role'], plan['eval_wait_seconds'], plan.get('variant_of')),
                         ('ghx4', '00:54:00', 'validation', 600, None))
        self.assertEqual((stock262['agent_cap_seconds'], stock262['agent_min_seconds'], stock262['step_limit'],
                          stock262['tasks']), (1800, 1400, 120, 32))
        self.assertEqual(plan['authorization'], 'AUTHORIZE_SWE32_STOCK25_32TASKS_16SESSIONS_262K_KV25_STOCK_VLLM_BATCH_'
                                                'REUSED_V9_ENV_ONE_GH200_54MIN_NO_RETRY')
        self.assertEqual((PINS['slurm']['mem'], PINS['slurm']['mail_type'], PINS['slurm']['mail_user']),
                         ('113560M', 'BEGIN,END,FAIL', 'cao58@illinois.edu'))
        self.assertIsNone(c.plan_blocked(PINS, 'stock25'))

    def test_refusals(self):
        def refused(change, pattern, plan='stock25'):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, plan)
        own = lambda pins: pins['config']['plans']['stock25']['stock']
        for bad in (0, 33, '8', True):  # (11-16: max_num_seqs may, an int 1-32)
            refused(lambda p, bad=bad: own(p).update(max_num_seqs=bad), 'stock may set')
        for bad in ('25', True, 0, 25.0):
            refused(lambda p, bad=bad: own(p).update(kv_cache_gib=bad), 'stock may set')
        refused(lambda p: own(p).update(preemption='ignore'), 'stock may set')
        refused(lambda p: p['config']['plans']['stock25'].update(stock={}), 'stock may set')
        refused(lambda p: own(p).update(max_model_len=200000), 'not a context of the runtime servers')
        refused(lambda p: own(p).update(kv_cache_gib=15), 'below one request')
        refused(lambda p: p['config']['plans']['obs'].update(stock=dict(kv_cache_gib=25)), 'without a stock control',
                'obs')
        refused(lambda p: p['config']['plans']['stock25']['servers'][0].update(agent_cap_seconds=2000), 'does not fit')
        refused(lambda p: p['config']['plans']['stock25']['servers'][0].update(goal_overlap=True), 'stock server')
        refused(lambda p: p['config']['plans']['stock25'].update(expected_seconds=2000), 'tight limits')
        # an interactive variant must carry its batch plan's stock settings
        pins = copy.deepcopy(PINS)
        plans = pins['config']['plans']
        plans['stock25i'] = dict(copy.deepcopy(plans['stock25']), partition='ghx4-interactive', role='interactive',
                                 variant_of='stock25', authorization=plans['stock25']['authorization'].replace(
                                     'STOCK25_', 'STOCK25I_').replace('_BATCH_', '_INTERACTIVE_'))
        c.validate_plan(pins, 'stock25i')
        plans['stock25i']['stock'] = dict(plans['stock25i']['stock'], kv_cache_gib=24)
        with self.assertRaisesRegex(RuntimeError, 'variant_of'):
            c.validate_plan(pins, 'stock25i')
        # without `preemption`: config.stock's rule (a preempted request fails the server)
        pins = copy.deepcopy(PINS)
        own(pins).pop('preemption')
        c.validate_plan(pins, 'stock25')
        self.assertEqual(c.stock_config(pins, c.plan_servers(pins, 'stock25')[0]).get('preemption', 'fail'), 'fail')

    def test_the_stock_checks_of_the_settings_and_the_preemption(self):
        import stock
        import smoke_checks as sc
        from unittest import mock
        mine = server('stock25')
        config, settings = c.stock_config(PINS, mine), mine[c.STOCK_SETTINGS_KEY]
        base = ('INFO [vllm.py:1006] Asynchronous scheduling is enabled.\n' * 2 + 'enforce_eager=False\n'
                '(EngineCore pid=2) INFO [kv_cache_utils.py:2146] GPU KV cache size: 393,120 tokens\n')
        args = ("(APIServer pid=1) INFO 11-02 [api_utils.py:273] non-default args: {'host': '127.0.0.1', 'port': 5, "
                "'seed': 42, 'max_model_len': %d, 'served_model_name': ['selective-prefix'], "
                "'kv_cache_memory_bytes': %d, 'enable_prefix_caching': True}\n")
        concurrency = '(EngineCore pid=2) INFO [kv_cache_utils.py:2147] Maximum concurrency for %s tokens per request: 1.5x\n'
        good = base + args % (262144, 25 * c.GIB) + concurrency % '262,144'
        status = lambda report, name: {r['check']: r['status'] for r in report['checks']}.get(name)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'server.log'
            log.write_text(good)
            self.assertEqual(stock.planned_settings(good, config), dict(
                max_model_len=262144, kv_cache_memory_bytes=25 * c.GIB, kv_cache_size_tokens=393120, preemption='finding'))
            self.assertEqual(status(stock.check_server(tmp, log, config, {}, settings=settings), 'stock-settings'), 'PASS')
            self.assertIsNone(status(stock.check_server(tmp, log, c.cfg(PINS)['stock'], {}), 'stock-settings'))
            for bad in (base + args % (131072, 25 * c.GIB) + concurrency % '262,144',
                        base + args % (262144, 32 * c.GIB) + concurrency % '262,144',
                        base + args % (262144, 25 * c.GIB) + concurrency % '131,072',
                        base + args % (262144, 25 * c.GIB), base + concurrency % '262,144'):
                with self.assertRaises(sc.CheckFailed, msg=bad):
                    stock.planned_settings(bad, config)
                log.write_text(bad)
                self.assertEqual(status(stock.check_server(tmp, log, config, {}, settings=settings), 'stock-settings'),
                                 'FAIL')
            log.write_text(good)
            preempted = sc.CheckFailed("vLLM preempted: [('metrics/sample-3.prom', 2.0)]")
            with mock.patch.object(sc, 'check_no_preemption', side_effect=preempted):
                finding = stock.check_server(tmp, log, config, {}, settings=settings)
                strict = stock.check_server(tmp, log, c.cfg(PINS)['stock'], {})
            row = {r['check']: r for r in finding['checks']}['no-preemption']
            self.assertEqual((row['status'], row['finding']), ('FINDING', stock.PREEMPTED))
            self.assertNotIn('no-preemption', finding['failed'])
            self.assertEqual([f['finding'] for f in finding['findings']], [stock.PREEMPTED])
            self.assertEqual(status(strict, 'no-preemption'), 'FAIL')  # plan stock: a failure, as before
            self.assertIn('no-preemption', strict['failed'])
            unreadable = sc.CheckFailed('no /metrics scrape shows vllm:num_preemptions_total: preemption cannot be '
                                        'ruled out')
            with mock.patch.object(sc, 'check_no_preemption', side_effect=unreadable):  # still a failure
                self.assertEqual(status(stock.check_server(tmp, log, config, {}, settings=settings), 'no-preemption'),
                                 'FAIL')
            with mock.patch.object(sc, 'check_no_preemption', return_value=dict(scrapes=3, chain=1, samples=2)):
                both = [stock.check_server(tmp, log, cfg, {}, **kw) for cfg, kw in (
                    (config, dict(settings=settings)), (c.cfg(PINS)['stock'], {}))]
            self.assertEqual([status(b, 'no-preemption') for b in both], ['PASS', 'PASS'])

    def test_the_entry_script(self):
        entry = ROOT / 'run_one_connection.ps1'
        run = lambda *args: subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                            str(entry)] + list(args), capture_output=True, text=True, timeout=300)
        phrase = c.cfg(PINS)['plans']['stock25']['authorization']
        value = run('-Plan', 'stock25', '-RunLabel', 'r1', '-AfterJob', '3500565', '-Authorize', '-Authorization', phrase,
                    '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('servers=stock262:stock:32tasks:120steps:1800s ', 'partition=ghx4 ', 'time=00:54:00 ',
                     'role=validation ', 'expected_seconds=2360 ', 'gpu_hours_limit=0.90 gpu_hours_expected=0.66 ',
                     'cap_after_wait_fits=true ', 'spare_seconds=180 ', 'memory_need_gib=32.00 memory_spare_gib=78.90 ',
                     'job_name=swe32-stock25 ', 'after_job=3500565 ', 'mem=113560M ',
                     'mail_user=cao58@illinois.edu mail_type=BEGIN,END,FAIL ',
                     ' plan_stock=own stock_kv_cache_gib=25 stock_max_model_len=262144 stock_preemption=finding'):
            self.assertIn(item, value.stdout)
        self.assertNotIn('plan_server=own', value.stdout)
        self.assertNotIn('NOT_SUBMITTABLE', value.stdout)
        other = run('-Plan', 'stock', '-Authorize', '-Authorization', c.cfg(PINS)['plans']['stock']['authorization'],
                    '-DryRun')
        self.assertEqual(other.returncode, 0, other.stdout + other.stderr)
        self.assertNotIn('plan_stock=', other.stdout)


class Stock25x(unittest.TestCase):
    """2026-11-02 (evening, user): plan stock25x = plan stock25 with the agent cap 3,600 s. stock25 r2 (job 3506513,
    SWE32_PARTIAL_VERIFIED) hit its 1,800-s cap (pool wall 1,807 s) with 26 / 32 done and 6 truncated. The same server
    stock262 (stock25's own stock settings), tasks, steps, minimum 1,400 s, evaluation wait, memory and partition;
    --time by stock25's rule (the full cap after the full wait) 01:24:00; stock25 and every other plan unchanged."""

    def test_stock25x_is_stock25_with_the_longer_cap(self):
        mine, theirs = server('stock25x'), server('stock25')
        self.assertEqual(mine.pop('agent_cap_seconds'), 3600)
        self.assertEqual(theirs.pop('agent_cap_seconds'), 1800)
        self.assertEqual(mine, theirs)  # label stock262, policy, tasks, steps, minimum 1,400 s, stock settings
        mine, theirs = server('stock25x'), server('stock25')
        plans = c.cfg(PINS)['plans']
        self.assertEqual(sorted(k for k in plans['stock25'] if plans['stock25'][k] != plans['stock25x'][k]),
                         ['about', 'authorization', 'expected_seconds', 'servers', 'time'])
        self.assertEqual(c.stock_config(PINS, mine), c.stock_config(PINS, theirs))
        self.assertEqual(c.stock_argv(PINS, 8123, mine), c.stock_argv(PINS, 8123, theirs))
        self.assertEqual(c.stock_expected(PINS, 8123, mine), c.stock_expected(PINS, 8123, theirs))
        self.assertEqual(c.model_override_text(PINS, mine, 23004), c.model_override_text(PINS, theirs, 23004))
        self.assertEqual(c.pool_spec(PINS, mine, 23004, 1800000000), c.pool_spec(PINS, theirs, 23004, 1800000000))
        self.assertEqual(c.memory_fit(PINS, 'stock25x'), c.memory_fit(PINS, 'stock25'))
        order = PLAN_ORDER  # (then smoke, 2026-11-03, and the thinking-on plans, 2026-11-06)
        self.assertEqual(order[order.index('stock25'):order.index('stock25') + 3], ['stock25', 'stock25x', 'smoke'])

    def test_time_cost_and_phrase(self):
        """540 s of job reserves + the 600-s evaluation wait (covers the 420-s start and tunnel) + the 3,600-s cap + 300 s
        (stop 180, stock checks 120) = 5,040 s = 01:24:00 (cap_after_wait_fits; worst case without the wait 4,860 s,
        minimum after it 2,840 s); E ~3,400 s (3506513: 394 s outside its pool + ~3,000 s of pool), limit 1.48 x E."""
        fit = c.validate_plan(PINS, 'stock25x')
        self.assertEqual([fit[k] for k in ('time_seconds', 'worst_case_seconds', 'spare_seconds', 'with_eval_wait_seconds',
                                           'guaranteed_seconds', 'cap_after_wait_fits', 'finish_seconds', 'fits')],
                         [5040, 4860, 180, 2840, 5040, True, 180, True])
        cost = c.plan_cost(PINS, 'stock25x')
        self.assertEqual([cost[k] for k in ('expected_seconds', 'billing_factor', 'gpu_hours_limit', 'gpu_hours_expected',
                                            'limit_over_expected')], [3400, 1, 1.4, 0.944, 1.482])
        plan, (stock262,) = c.cfg(PINS)['plans']['stock25x'], c.plan_servers(PINS, 'stock25x')
        self.assertEqual((plan['partition'], plan['time'], plan['role'], plan['eval_wait_seconds'], plan.get('variant_of')),
                         ('ghx4', '01:24:00', 'validation', 600, None))
        self.assertEqual((stock262['label'], stock262['agent_cap_seconds'], stock262['agent_min_seconds'],
                          stock262['step_limit'], stock262['tasks']), ('stock262', 3600, 1400, 120, 32))
        self.assertEqual(plan['authorization'], 'AUTHORIZE_SWE32_STOCK25X_32TASKS_16SESSIONS_262K_KV25_STOCK_VLLM_CAP3600_'
                                                'BATCH_REUSED_V9_ENV_ONE_GH200_84MIN_NO_RETRY')
        self.assertEqual((PINS['slurm']['mem'], PINS['slurm']['mail_type'], PINS['slurm']['mail_user']),
                         ('113560M', 'BEGIN,END,FAIL', 'cao58@illinois.edu'))
        self.assertIsNone(c.plan_blocked(PINS, 'stock25x'))

    def test_refusals(self):
        def refused(change, pattern):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, 'stock25x')
        plan = lambda pins: pins['config']['plans']['stock25x']
        refused(lambda p: plan(p)['servers'][0].update(agent_cap_seconds=3800), 'does not fit')
        refused(lambda p: plan(p).update(expected_seconds=3300), 'tight limits')
        refused(lambda p: plan(p).update(authorization=plan(p)['authorization'].replace('_84MIN_', '_54MIN_')), 'phrase')
        refused(lambda p: plan(p).update(authorization=PINS['config']['plans']['stock25']['authorization']), 'phrase')
        refused(lambda p: plan(p)['stock'].update(kv_cache_gib=15), 'below one request')

    def test_the_entry_script(self):
        entry = ROOT / 'run_one_connection.ps1'
        run = lambda *args: subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                            str(entry)] + list(args), capture_output=True, text=True, timeout=300)
        phrase = c.cfg(PINS)['plans']['stock25x']['authorization']
        value = run('-Plan', 'stock25x', '-RunLabel', 'r1', '-Authorize', '-Authorization', phrase, '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('servers=stock262:stock:32tasks:120steps:3600s ', 'partition=ghx4 ', 'time=01:24:00 ',
                     'role=validation ', 'expected_seconds=3400 ', 'gpu_hours_limit=1.40 gpu_hours_expected=0.94 ',
                     'worst_case_seconds=4860 ', 'with_eval_wait_seconds=2840 ', 'cap_after_wait_fits=true ',
                     'spare_seconds=180 ', 'memory_need_gib=32.00 memory_spare_gib=78.90 ', 'job_name=swe32-stock25x ',
                     'run_label=r1 ', 'after_job= ', 'mem=113560M ', 'mail_user=cao58@illinois.edu mail_type=BEGIN,END,FAIL ',
                     ' plan_stock=own stock_kv_cache_gib=25 stock_max_model_len=262144 stock_preemption=finding'):
            self.assertIn(item, value.stdout)
        self.assertNotIn('NOT_SUBMITTABLE', value.stdout)
        wrong = run('-Plan', 'stock25x', '-RunLabel', 'r1', '-Authorize', '-Authorization',
                    c.cfg(PINS)['plans']['stock25']['authorization'], '-DryRun')  # stock25's phrase: refused
        self.assertNotEqual(wrong.returncode, 0)


class ReleaseSmoke(unittest.TestCase):
    """2026-11-03 (the release smoke matrix, J1): plan smoke, role release-smoke, on integrate-1103 ea18256. Three servers
    back to back: obs262s (observation_query), rfill262s (random_fill), goalvllms (vllm25's goalvllm: the vLLM rewriter,
    batch-invariant lm-format-enforcer, 3 GiB, --goal-overlap); ctx25's own settings; the first 32 instances at K = 16
    (the session pool's close and refill), the step limit 20; --async-scheduling on every runtime server (the plan's
    one switch, D2); its own limit rule (<= 2.25 x expected, D4); the async-scheduling and cell-coverage checks; the
    speed defaults of the release runtime never apply (every switch named)."""

    def test_the_plan(self):
        plan = c.cfg(PINS)['plans']['smoke']
        servers = c.plan_servers(PINS, 'smoke')
        self.assertEqual([(s['label'], s['policy'], s['tasks'], s['step_limit'], s['agent_cap_seconds'],
                           s['agent_min_seconds'], s['kind'], s[c.ASYNC_KEY], s[c.COVERAGE_KEY]) for s in servers],
                         [('obs262s', 'observation_query', 32, 20, 840, 450, 'runtime', True, True),
                          ('rfill262s', 'random_fill', 32, 20, 840, 450, 'runtime', True, True),
                          ('goalvllms', 'goal_query', 32, 20, 840, 450, 'runtime', True, True)])
        self.assertEqual((plan['role'], plan['partition'], plan['time'], plan['expected_seconds'],
                          plan['eval_wait_seconds'], plan['server']),
                         ('release-smoke', 'ghx4', '01:49:00', 3300, 900,
                          dict(kv_cache_gib=25, max_model_len=262144, dram_arena_gib=70)))
        self.assertEqual(plan['authorization'], 'AUTHORIZE_SWE32_SMOKE_32TASKS_16SESSIONS_3SERVERS_20STEPS_262K_KV25_'
                         'VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_109MIN_NO_RETRY')
        self.assertEqual(c.plan_settings(PINS, 'smoke'), c.plan_settings(PINS, 'ctx25'))
        # goalvllms is vllm25's goalvllm but the label, the step limit, the cap / minimum and the plan's async switch
        goal, vllm = dict(servers[2]), dict(c.plan_servers(PINS, 'vllm25')[0])
        for k in ('label', 'step_limit', 'agent_cap_seconds', 'agent_min_seconds', c.ASYNC_KEY, c.COVERAGE_KEY):
            goal.pop(k), vllm.pop(k, None)
        self.assertEqual(goal, vllm)
        # the policies' own flags (obs / rfill: the active budget only; goal: overlap 16 and the rewriter)
        self.assertEqual([(s['policy_args'], s['admission'], s['rewriter']) for s in servers],
                         [(['--active-token-budget', '8192'], [], False), (['--active-token-budget', '8192'], [], False),
                          (['--active-token-budget', '8192'], ['admission_overlap=16'], True)])

    def test_time_cost_and_the_role_rule(self):
        fit = c.validate_plan(PINS, 'smoke')
        self.assertEqual((fit['time_seconds'], fit['worst_case_seconds'], fit['guaranteed_seconds'],
                          fit['with_eval_wait_seconds'], fit['spare_seconds'], fit['cap_after_wait_fits']),
                         (6540, 6060, 6540, 5370, 480, True))
        # 840 s of job reserves + (the 900-s wait over the 420-s start + cap 840 + stop / checks 480) + 2 x (420 + 840
        # + 480) = 6,540 s = 01:49:00
        self.assertEqual(840 + (900 + 840 + 480) + 2 * (420 + 840 + 480), c.time_seconds('01:49:00'))
        cost = c.plan_cost(PINS, 'smoke')
        self.assertEqual((cost['gpu_hours_limit'], cost['gpu_hours_expected'], cost['limit_over_expected']),
                         (1.817, 0.917, 1.982))
        self.assertLess(cost['limit_over_expected'], 2.25)
        self.assertGreater(cost['limit_over_expected'], c.MAX_LIMIT_OVER_EXPECTED)  # only its role allows it
        self.assertEqual(c.max_limit_over_expected(c.cfg(PINS)['plans']['smoke']), 2.25)
        for name in PLAN_ORDER:  # every other role keeps 1.5 x (smoket: release-smoke too)
            if name not in ('smoke', 'smoket'):
                self.assertEqual(c.max_limit_over_expected(c.cfg(PINS)['plans'][name]), 1.5, name)
        memory = c.memory_fit(PINS, 'smoke')
        self.assertEqual((memory['need_gib'], memory['headroom_gib'], memory['fits']), (105.29, 5.61, True))
        self.assertLess(fit['time_seconds'], 3 * 3600)  # one job: below the ~3 h split rule

    def test_refusals(self):
        def refused(change, pattern, plan='smoke'):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, plan)
        p = lambda pins: pins['config']['plans']['smoke']
        refused(lambda q: p(q).update(expected_seconds=2900), 'tight limits')  # 6,540 > 2.25 x 2,900
        refused(lambda q: q['config']['plans']['ctx25'].update(expected_seconds=4000), 'tight limits', 'ctx25')
        refused(lambda q: q['config']['plans']['ctx25'].update(role='release-smoke', expected_seconds=3900),
                'tight limits', 'ctx25')  # 8,820 > 2.25 x 3,900
        for bad in ('true', 1, None):
            refused(lambda q, bad=bad: p(q).update(async_scheduling=bad), 'async_scheduling must be true or false')
        refused(lambda q: p(q)['servers'][0].update(async_scheduling=True), 'plan switch, not a server key')
        refused(lambda q: q['config']['plans']['stock'].update(async_scheduling=True), 'without a stock control',
                'stock')
        refused(lambda q: q['runtime_features'].update(async_scheduling=None), 'no --async-scheduling')
        refused(lambda q: q['config']['server'].update(decode_fastpath2=False, decode_fastpath3=False),
                'requires --decode-fastpath2')
        refused(lambda q: q['config']['policies']['observation_query'].update(admission=['phase1_in_step']),
                'refuses --gdn-device-handoff and --phase1-in-step')
        refused(lambda q: p(q).update(time='01:40:00', authorization=p(q)['authorization'].replace('109MIN', '100MIN')),
                'does not fit')  # below the worst case 6,060 s
        refused(lambda q: p(q).update(role='release'), 'role')
        off = copy.deepcopy(PINS)
        p(off)['async_scheduling'] = False  # the D2 switch off: valid, --no-async-scheduling
        c.validate_plan(off, 'smoke')
        self.assertTrue(all(c.ASYNC_SWITCH not in c.launch_arguments(off, s, REWRITER)
                            for s in c.plan_servers(off, 'smoke')))

    def test_launch_arguments_and_expectations(self):
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m/qwen3-0.6b', config_sha256='0' * 64)
        pairs = (('smoke', 0, 'ctx25', 1), ('smoke', 2, 'vllm25', 0))  # obs262s = ctx25's obs262, goalvllms = goalvllm
        for plan, i, other, j in pairs:
            mine, theirs = c.plan_servers(PINS, plan)[i], c.plan_servers(PINS, other)[j]
            args = c.launch_arguments(PINS, mine, REWRITER)
            self.assertEqual(args.count(c.ASYNC_SWITCH), 1)
            self.assertEqual([a for a in args if a != c.ASYNC_SWITCH], c.launch_arguments(PINS, theirs, REWRITER))
            self.assertLess(args.index(c.LMO_SWITCH), args.index(c.ASYNC_SWITCH))
            e, f = (c.server_expectation(PINS, s, gate, 'm', ['t0']) for s in (mine, theirs))
            self.assertEqual((e.pop(c.ASYNC_KEY), e.pop(c.COVERAGE_KEY)), (True, dict(switch_tokens=8192)))
            for k in ('label', 'step_limit'):
                e.pop(k), f.pop(k)
            self.assertEqual(e, f, (plan, i))
        rfill = c.plan_servers(PINS, 'smoke')[1]  # random_fill: the same flags as obs262 (its policy is passed apart)
        self.assertEqual([a for a in c.launch_arguments(PINS, rfill, REWRITER) if a != c.ASYNC_SWITCH],
                         c.launch_arguments(PINS, c.plan_servers(PINS, 'ctx25')[1], REWRITER))
        # no other plan's servers carry the async or coverage key (their fingerprints are unchanged; smoket: the
        # release smoke with thinking, ReleaseSmokeThinking)
        for name in PLAN_ORDER:
            if name not in ('smoke', 'smoket'):
                for s in c.plan_servers(PINS, name):  # (2026-11-16: the async pair names the switch, no coverage)
                    self.assertFalse(({c.COVERAGE_KEY} if name in ASYNC_NAMED else {c.ASYNC_KEY, c.COVERAGE_KEY})
                                     & set(s), name)
        # the speed defaults never apply: a switch turned off is passed as its off switch
        speed = PINS['runtime_features'][c.SPEED_KEY]
        self.assertEqual(speed['switches'], ['decode_fastpath', 'decode_fastpath2', 'decode_fastpath3',
                                             'deferred_completion_io', 'language_model_only'])
        off = copy.deepcopy(PINS)
        off['config']['server'].update(decode_fastpath2=False, decode_fastpath3=False, deferred_completion_io=False)
        args = c.launch_arguments(off, c.plan_servers(off, 'full')[0], REWRITER)
        self.assertEqual([a for a in args if a.startswith('--no-')],
                         ['--no-decode-fastpath2', '--no-decode-fastpath3', '--no-deferred-completion-io'])
        before = copy.deepcopy(off)
        before['runtime_features'][c.SPEED_KEY] = None  # a runtime before defaults-on: the switch is left out
        self.assertFalse([a for a in c.launch_arguments(before, c.plan_servers(before, 'full')[0], REWRITER)
                          if a.startswith('--no-')])

    def test_dry_run_gate_keeps_its_release_dir(self):
        """2026-11-13 (smoket r1 refused on the login node: '--release-dir False' for its second server): inside
        serve_dry_run_gate no statement rebinds `release`, the release directory every server's dry run is given."""
        import ast
        import inspect
        import textwrap
        tree = ast.parse(textwrap.dedent(inspect.getsource(c.serve_dry_run_gate)))
        bound = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Name) and n.id == 'release' and isinstance(n.ctx, ast.Store)]
        self.assertEqual(bound, [], 'serve_dry_run_gate rebinds release at lines %s' % bound)

    def test_dry_run_gate_rules(self):
        base = ['vllm', '--compilation-config', c.compilation_argument(c.COMPILATION['full'])]
        c.check_graph_command(base + ['--no-async-scheduling'], PINS)
        c.check_graph_command(base + ['--async-scheduling'], PINS, True)
        for command, asynchronous in ((base + ['--async-scheduling'], False), (base + ['--no-async-scheduling'], True),
                                      (base + ['--async-scheduling', '--no-async-scheduling'], True), (base, True)):
            with self.assertRaises(RuntimeError):
                c.check_graph_command(command, PINS, asynchronous)

    def test_async_check_reads_the_main_engine_only(self):
        import smoke_checks as sc
        import swe_checks
        log = (ROOT / 'fixtures' / 'server-log-child-engine-job3500565.log').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory(prefix='swe32-async-') as tmp:
            out = Path(tmp)
            child = '/projects/bcjw/jgerard1/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/' \
                    'c1899de289a04d12100db370d81485cdf75e47ca'
            record = dict(engine='vllm', engine_args=dict(model=child))

            def files(on, worker=None):
                effective = dict(max_num_seqs=16, **({c.ASYNC_KEY: True} if on else {}))
                startup = dict(effective_server_config=effective, goal_rewriter_engine=record,
                               **({c.ASYNC_KEY: True} if on else {}))
                (out / 'startup.json').write_text(json.dumps(startup))
                (out / 'ready.json').write_text(json.dumps(dict(effective_server_config=effective)))
                receipt = {c.ASYNC_KEY: True} if (on if worker is None else worker) else {}
                (out / 'worker-attached.json').write_text(json.dumps(dict(receipt=receipt)))

            def check(text, on, **kw):
                files(on, **kw)
                (out / 'server.log').write_text(text, encoding='utf-8')
                report = sc.Report()
                report.run('async-scheduling', lambda: swe_checks.check_async_scheduling(
                    out, {c.ASYNC_KEY: on}, out / 'server.log'))
                return report.checks[-1]

            # job 3500565 (async off): the main engine says disabled (APIServer, EngineCore 15428); the child's
            # unprefixed frontend line and its EngineCore 33376 say enabled: PASS, the child's as information
            seen = check(log, False)
            self.assertEqual(seen['status'], 'PASS', seen)
            self.assertEqual((seen['detail']['main_state'], seen['detail']['main_lines'], seen['detail']['rewriter']),
                             ('disabled', 1, [dict(pid=33376, states=['enabled'])]))
            # read whole, the log has both states: the false positive the main-engine reading removes
            self.assertEqual(sorted(set(sc.async_states(log))), ['disabled', 'enabled'])
            # async on: the main engine's lines must say enabled; the same log fails on its main line
            self.assertEqual(check(log, True)['status'], 'FAIL')
            main = '(EngineCore pid=15428) INFO 11-02 02:19:13 [vllm.py:1006] Asynchronous scheduling is '
            flipped = log.replace(main + 'disabled.', main + 'enabled.')
            self.assertNotEqual(flipped, log)
            self.assertEqual(check(flipped, True)['status'], 'PASS')
            self.assertEqual(check(flipped, False)['status'], 'FAIL')
            mixed = flipped + '(APIServer pid=13581) INFO 11-02 02:14:31 [vllm.py:1006] Asynchronous scheduling is ' \
                'disabled.\n'
            self.assertEqual(check(mixed, True)['status'], 'FAIL')  # the main engine's lines disagree
            self.assertEqual(check(log.replace(main + 'disabled.', ''), False)['status'], 'FAIL')  # no main line
            # the records: on everywhere, or off nowhere
            self.assertEqual(check(flipped, True, worker=False)['status'], 'FAIL')
            self.assertEqual(check(log, False, worker=True)['status'], 'FAIL')
            # a log with one engine is read whole (every earlier server)
            single = '(APIServer pid=1) INFO [vllm.py:1006] Asynchronous scheduling is enabled.\n' \
                     '(EngineCore pid=2) INFO [core.py:114] Initializing a V1 LLM engine (v0.24.0) with config: ' \
                     "model='/m'\n(EngineCore pid=2) INFO [vllm.py:1006] Asynchronous scheduling is enabled.\n"
            self.assertEqual(check(single, True)['status'], 'PASS')
            self.assertEqual(check(single.replace('enabled.\n(EngineCore', 'disabled.\n(EngineCore', 1), True)['status'],
                             'FAIL')

    def test_the_three_engine_checks_still_read_the_child_log_as_before(self):
        """engine_logs gives the child's unprefixed scheduling line to the child: the main text loses it, nothing else."""
        log = (ROOT / 'fixtures' / 'server-log-child-engine-job3500565.log').read_text(encoding='utf-8')
        child = '/projects/bcjw/jgerard1/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/' \
                'c1899de289a04d12100db370d81485cdf75e47ca'
        import smoke_checks as sc
        with tempfile.TemporaryDirectory(prefix='swe32-async-') as tmp:
            (Path(tmp) / 'startup.json').write_text(json.dumps(dict(goal_rewriter_engine=dict(
                engine='vllm', engine_args=dict(model=child)))))
            engines = sc.engine_logs(tmp, log)
        line = 'INFO 11-02 02:17:57 [vllm.py:1006] Asynchronous scheduling is enabled.'
        self.assertIn(line, log)
        self.assertNotIn(line, engines['main'])
        self.assertIn(line, engines['rewriter'][0]['text'])
        self.assertEqual(sc.async_states(engines['main']), ['disabled'])  # the main EngineCore's line only
        self.assertEqual(sorted({int(v) for v in sc.MAX_SEQ_LEN.findall(engines['main'])}), [262144])
        self.assertEqual(len(engines['main'].split('\n')), len(log.split('\n')) - len(
            engines['rewriter'][0]['text'].split('\n')))  # every other line stays where it was

    def test_coverage_line(self):
        import smoke_checks as sc
        import swe_checks
        expect = {c.COVERAGE_KEY: dict(switch_tokens=8192)}
        active = lambda n, s: dict(selection=dict(active=True, candidate_pages=n, selected=s), prompt_tokens=n * 16)
        quiet = dict(selection=dict(candidate_pages=100, selected=100, reason='below_switch_threshold'),
                     prompt_tokens=1600)

        def run(policy, turns, goals=()):
            report = sc.Report()
            report.run('cell-coverage', lambda: swe_checks.check_coverage(policy, turns, list(goals), expect))
            return report.checks[-1]
        seen = run('observation_query', [quiet, active(900, 512), active(1000, 512)])
        self.assertEqual((seen['status'], seen['detail']['coverage'], seen['detail']['dropped_page_turns'],
                          seen['detail']['past_switch'], seen['detail']['active_selection']), ('PASS', 'COVERED', 2, 2, 2))
        seen = run('random_fill', [quiet, quiet])
        self.assertEqual((seen['status'], seen['finding']), ('FINDING', 'NOT_EXERCISED'))
        seen = run('goal_query', [active(900, 512)], [dict(applied_decision='KEEP')])
        self.assertEqual((seen['status'], seen['finding'], seen['detail']['rewrite_turns']), ('FINDING', 'NOT_EXERCISED', 0))
        seen = run('goal_query', [active(900, 512)], [dict(applied_decision='REWRITE'), None])
        self.assertEqual((seen['status'], seen['detail']['coverage'], seen['detail']['rewrite_turns']), ('PASS', 'COVERED', 1))
        self.assertEqual(run('full', [quiet])['detail']['coverage'], 'NOT_APPLICABLE')
        self.assertEqual(run('observation_query', [active(512, 512)])['finding'], 'NOT_EXERCISED')  # nothing dropped

    def test_entry_script_dry_run(self):
        entry = ROOT / 'run_one_connection.ps1'
        run = lambda *args: subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                            str(entry)] + list(args), capture_output=True, text=True, timeout=300)
        phrase = c.cfg(PINS)['plans']['smoke']['authorization']
        value = run('-Plan', 'smoke', '-RunLabel', 'r1', '-AfterJob', '3507294', '-Authorize', '-Authorization', phrase,
                    '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('DRY_RUN connections=0', 'plan=smoke role=release-smoke variant_of=- ',
                     'servers=obs262s:observation_query:32tasks:20steps:840s,rfill262s:random_fill:32tasks:20steps:840s,'
                     'goalvllms:goal_query:32tasks:20steps:840s:vllm3gib:overlap ', 'partition=ghx4 time=01:49:00 ',
                     'expected_seconds=3300 billing_factor=1 gpu_hours_limit=1.82 gpu_hours_expected=0.92 ',
                     'release_check=PINNED receipt_result_sha256=' + RECEIPT_PIN['result_line_sha256'] + ' ',
                     'worst_case_seconds=6060 with_eval_wait_seconds=5370 cap_after_wait_fits=true eval_wait_seconds=900 ',
                     'mem=113560M ', 'memory_need_gib=105.29 memory_spare_gib=5.61 ', 'job_name=swe32-smoke ',
                     'after_job=3507294 ', 'mail_user=cao58@illinois.edu mail_type=BEGIN,END,FAIL ',
                     'plan_server=own kv_cache_gib=25 max_model_len=262144 plan_dram_arena_gib=70 ',
                     'goal_rewriter_engine=goalvllms:vllm:lm-format-enforcer:3:batch-invariant goal_overlap=goalvllms ',
                     ' async_scheduling=obs262s:on,rfill262s:on,goalvllms:on'):
            self.assertIn(item, value.stdout)
        self.assertNotIn('NOT_SUBMITTABLE', value.stdout)
        other = run('-Plan', 'vllm25', '-Authorize', '-Authorization', c.cfg(PINS)['plans']['vllm25']['authorization'],
                    '-DryRun')
        self.assertEqual(other.returncode, 0, other.stdout + other.stderr)
        self.assertNotIn('async_scheduling=', other.stdout)
        wrong = run('-Plan', 'smoke', '-Authorize', '-Authorization', phrase.replace('109MIN', '108MIN'), '-DryRun')
        self.assertNotEqual(wrong.returncode, 0)


class ReleaseSmokeThinking(unittest.TestCase):
    """2026-11-09 (the re-pin to main ba4d1ba, PLAN-REPIN 4.2; decisions D-B, D-D, D-E): plan smoket, role release-smoke.
    smoke's three servers (labels + t: obs262st, rfill262st, goalvllmst = vllm25's goalvllm) with the plan switches
    thinking (serve.py --enable-thinking, keep on the XML protocol) and async_scheduling both true, caps 1,440 / minimum
    800 s from job 3539242's 960-s step-20 wall, the evaluation wait 1,800 s (last in the tbench chain): 02:34:00,
    2.57 / 1.26 GPU-h. smoke stays the sealed thinking-off fallback; both need the async fixes (AsyncFix)."""

    def test_the_plan(self):
        plan = c.cfg(PINS)['plans']['smoket']
        servers = c.plan_servers(PINS, 'smoket')
        self.assertEqual([(s['label'], s['policy'], s['tasks'], s['step_limit'], s['agent_cap_seconds'],
                           s['agent_min_seconds'], s['kind'], s[c.ASYNC_KEY], s[c.THINKING_KEY], s[c.COVERAGE_KEY])
                          for s in servers],
                         [('obs262st', 'observation_query', 32, 20, 1440, 800, 'runtime', True, True, True),
                          ('rfill262st', 'random_fill', 32, 20, 1440, 800, 'runtime', True, True, True),
                          ('goalvllmst', 'goal_query', 32, 20, 1440, 800, 'runtime', True, True, True)])
        self.assertEqual((plan['role'], plan['partition'], plan['time'], plan['expected_seconds'],
                          plan['eval_wait_seconds'], plan['server'], plan[c.ASYNC_KEY], plan[c.THINKING_KEY]),
                         ('release-smoke', 'ghx4', '02:34:00', 4530, 1800,
                          dict(kv_cache_gib=25, max_model_len=262144, dram_arena_gib=70), True, True))
        self.assertEqual(plan['authorization'], 'AUTHORIZE_SWE32_SMOKET_32TASKS_16SESSIONS_3SERVERS_20STEPS_262K_KV25_'
                         'THINKING_ON_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_154MIN_NO_RETRY')
        self.assertEqual(c.plan_settings(PINS, 'smoket'), c.plan_settings(PINS, 'smoke'))
        # smoke's servers but the label (+ t), the cap / minimum and the thinking switch
        for mine, theirs in zip(servers, c.plan_servers(PINS, 'smoke')):
            self.assertEqual(mine['label'], theirs['label'] + 't')
            a, b = dict(mine), dict(theirs)
            for k in ('label', 'agent_cap_seconds', 'agent_min_seconds', c.THINKING_KEY):
                a.pop(k), b.pop(k, None)
            self.assertEqual(a, b, mine['label'])
        self.assertIn('(thinking.py)', plan['about'])
        self.assertIn('960.2 s (recentt 3539242', plan['about'])
        self.assertEqual(ORDER1112[-1 - len(THINKI + FULLT + NEW64 + NEW1112)], 'smoket')  # (then 11-11, 11-12)

    def test_time_cost_and_the_role_rule(self):
        fit = c.validate_plan(PINS, 'smoket')
        self.assertEqual((fit['time_seconds'], fit['worst_case_seconds'], fit['guaranteed_seconds'],
                          fit['with_eval_wait_seconds'], fit['spare_seconds'], fit['cap_after_wait_fits']),
                         (9240, 7860, 9240, 7320, 1380, True))
        # 840 s of job reserves + (the 1,800-s wait over the 420-s start + cap 1,440 + stop / checks 480) + 2 x (420 +
        # 1,440 + 480) = 9,240 s = 02:34:00 (PLAN-REPIN 4.2)
        self.assertEqual(840 + (1800 + 1440 + 480) + 2 * (420 + 1440 + 480), c.time_seconds('02:34:00'))
        # E: job checks 40 + READY 250 / 90 / 180 + pools 960 / 960 / 1,000 + 3 x 200 stop / checks + 450 finish
        self.assertEqual(40 + (250 + 90 + 180) + (960 + 960 + 1000) + 3 * 200 + 450, 4530)
        # the caps: 1.5 x the measured thinking-on step-20 pool wall (recentt 3539242, 960.2 s), minimum 800 s (D-D)
        self.assertEqual(round(1.5 * 960.2 / 10) * 10, 1440)
        cost = c.plan_cost(PINS, 'smoket')
        self.assertEqual((cost['gpu_hours_limit'], cost['gpu_hours_expected'], cost['limit_over_expected']),
                         (2.567, 1.258, 2.04))
        self.assertLess(cost['limit_over_expected'], 2.25)
        self.assertEqual(c.max_limit_over_expected(c.cfg(PINS)['plans']['smoket']), 2.25)
        memory = c.memory_fit(PINS, 'smoket')
        self.assertEqual((memory['need_gib'], memory['headroom_gib'], memory['fits']), (105.29, 5.61, True))
        self.assertLess(fit['time_seconds'], 3 * 3600)  # one job: below the ~3 h split rule
        # with the 900-s wait it would be 02:19:00 (PLAN-REPIN 4.2): the wait is the only difference to the fit
        self.assertEqual(840 + (900 + 1440 + 480) + 2 * (420 + 1440 + 480), c.time_seconds('02:19:00'))

    def test_refusals(self):
        def refused(change, pattern, plan='smoket'):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, plan)
        p = lambda pins: pins['config']['plans']['smoket']
        refused(lambda q: p(q).update(expected_seconds=4000), 'tight limits')  # 9,240 > 2.25 x 4,000
        refused(lambda q: [s.update(agent_cap_seconds=1950) for s in p(q)['servers']], 'does not fit')  # 9,390 s
        refused(lambda q: p(q).update(eval_wait_seconds=3800), 'does not fit')  # the minimums after the wait: 9,320 s
        refused(lambda q: p(q).update(time='02:10:00', authorization=p(q)['authorization'].replace('154MIN', '130MIN')),
                'does not fit')  # below the worst case 7,860 s
        refused(lambda q: p(q).update(authorization=p(q)['authorization'].replace('154MIN', '153MIN')), 'phrase')
        for bad in ('true', 1, None):
            refused(lambda q, bad=bad: p(q).update(thinking=bad), 'thinking must be true or false')
            refused(lambda q, bad=bad: p(q).update(async_scheduling=bad), 'async_scheduling must be true or false')
        refused(lambda q: p(q)['servers'][1].update(thinking=True), 'thinking is a plan switch')
        refused(lambda q: p(q)['servers'][0].update(async_scheduling=True), 'plan switch, not a server key')
        refused(lambda q: q['runtime_features'].update(thinking=None), 'no --enable-thinking')
        refused(lambda q: q['runtime_features'].update(async_scheduling=None), 'no --async-scheduling')
        refused(lambda q: q['config']['server'].update(decode_fastpath2=False, decode_fastpath3=False),
                'requires --decode-fastpath2')
        with self.assertRaisesRegex(RuntimeError, 'ask for --enable-thinking'):
            c.check_features(dict(PINS['runtime_features'], thinking=None), PINS['config'])
        off = copy.deepcopy(PINS)
        p(off)['thinking'] = False  # thinking off: smoke's arguments under smoket's labels
        c.validate_plan(off, 'smoket')
        self.assertEqual([c.launch_arguments(off, s, REWRITER) for s in c.plan_servers(off, 'smoket')],
                         [c.launch_arguments(PINS, s, REWRITER) for s in c.plan_servers(PINS, 'smoke')])

    def test_launch_arguments_and_expectations(self):
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m/qwen3-0.6b', config_sha256='0' * 64)
        record = PINS['runtime_features'][c.THINKING_KEY]['record']
        for mine, theirs in zip(c.plan_servers(PINS, 'smoket'), c.plan_servers(PINS, 'smoke')):
            args = c.launch_arguments(PINS, mine, REWRITER)
            # smoke's arguments plus --enable-thinking right after --async-scheduling (both before the policy flags)
            self.assertEqual((args.count(c.ASYNC_SWITCH), args.count(c.THINKING_SWITCH)), (1, 1), mine['label'])
            self.assertEqual(args.index(c.THINKING_SWITCH), args.index(c.ASYNC_SWITCH) + 1, mine['label'])
            self.assertEqual([a for a in args if a != c.THINKING_SWITCH], c.launch_arguments(PINS, theirs, REWRITER))
            e, f = (c.server_expectation(PINS, s, gate, 'm', ['t0']) for s in (mine, theirs))
            self.assertEqual((e[c.ASYNC_KEY], e[c.THINKING_KEY], e[c.COVERAGE_KEY]),
                             (True, record, dict(switch_tokens=8192)), mine['label'])
            e.pop(c.THINKING_KEY)
            self.assertEqual(e.pop(c.KEEP_REBUILD_KEY), PINS['runtime_features'][c.KEEP_REBUILD_KEY])  # (a2f9e36)
            self.assertNotIn(c.KEEP_REBUILD_KEY, f)
            for k in ('label',):
                e.pop(k), f.pop(k)
            self.assertEqual(e, f, mine['label'])
            text = c.model_override_text(PINS, mine, 23004)
            self.assertIn('    extra_body: {"chat_template_kwargs": {"enable_thinking": true}}', text.splitlines())
            self.assertNotIn('    extra_body: {"chat_template_kwargs": {"enable_thinking": true}}',
                             c.model_override_text(PINS, theirs, 23004).splitlines())
        # no plan before smoket has thinking and async together
        both = [n for n in PLAN_ORDER if c.cfg(PINS)['plans'][n].get(c.ASYNC_KEY) is True
                and c.cfg(PINS)['plans'][n].get(c.THINKING_KEY) is True]
        self.assertEqual(both, ['smoket'] + ASYNC_ON)  # 2026-11-16: and the async pair / second-half async plans

    def test_entry_script_dry_run(self):
        entry = ROOT / 'run_one_connection.ps1'
        run = lambda *args: subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                            str(entry)] + list(args), capture_output=True, text=True, timeout=300)
        phrase = c.cfg(PINS)['plans']['smoket']['authorization']
        value = run('-Plan', 'smoket', '-RunLabel', 'r1', '-AfterJob', '3560498', '-Authorize', '-Authorization', phrase,
                    '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('DRY_RUN connections=0', 'plan=smoket role=release-smoke variant_of=- ',
                     'servers=obs262st:observation_query:32tasks:20steps:1440s:thinking,rfill262st:random_fill:32tasks:'
                     '20steps:1440s:thinking,goalvllmst:goal_query:32tasks:20steps:1440s:vllm3gib:overlap:thinking ',
                     'partition=ghx4 time=02:34:00 ',
                     'expected_seconds=4530 billing_factor=1 gpu_hours_limit=2.57 gpu_hours_expected=1.26 ',
                     'release_check=PINNED receipt_result_sha256=' + RECEIPT_PIN['result_line_sha256'] + ' ',
                     'worst_case_seconds=7860 with_eval_wait_seconds=7320 cap_after_wait_fits=true eval_wait_seconds=1800 ',
                     'mem=113560M ', 'memory_need_gib=105.29 memory_spare_gib=5.61 ', 'job_name=swe32-smoket ',
                     'after_job=3560498 ', 'mail_user=cao58@illinois.edu mail_type=BEGIN,END,FAIL ',
                     'plan_server=own kv_cache_gib=25 max_model_len=262144 plan_dram_arena_gib=70 ',
                     'goal_rewriter_engine=goalvllmst:vllm:lm-format-enforcer:3:batch-invariant goal_overlap=goalvllmst ',
                     ' async_scheduling=obs262st:on,rfill262st:on,goalvllmst:on',
                     ' thinking=obs262st:on,rfill262st:on,goalvllmst:on'):
            self.assertIn(item, value.stdout)
        self.assertNotIn('NOT_SUBMITTABLE', value.stdout)
        wrong = run('-Plan', 'smoket', '-Authorize', '-Authorization', c.cfg(PINS)['plans']['smoke']['authorization'],
                    '-DryRun')
        self.assertNotEqual(wrong.returncode, 0)  # smoke's phrase never authorizes smoket


class AsyncFix(unittest.TestCase):
    """2026-11-09 (decision D-G, PLAN-REPIN 4.1.8): the feature async_fix (async-fix a0798a8: the decoupled layout's
    scheduler keeps deferred block frees by pool instead of refusing them; async-fix2 4ed1013: native_probe's
    _runner_pass) read from the sealed source; every plan with async_scheduling true (smoke, smoket) is blocked
    (NOT_SUBMITTABLE, setup / qualify refuse it) on a pin without it (ea18256 / f4ccd4b: async G1 r1 3499729 / r2
    3519531 failed there); no other plan is."""

    RESOLVED = '                        if i.request in scheduled and i.request in qwen.requests\n'

    def tree(self, base, scheduler, probe, resolve=RESOLVED):
        for rel, text in ((c.ASYNC_FIX_SCHEDULER, scheduler), (c.ASYNC_FIX_PROBE, probe), (c.ASYNC_FIX_RESOLVE, resolve)):
            if text is not None:
                path = Path(base) / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding='utf-8')
        return base

    def test_the_reader(self):
        fixed = ('        self._remote_ra_deferred_by_pool = kv_cache_config.group_pool is not None\n')
        refusal = ('                raise ValueError(\n                    "decoupled KV layout does not support deferred '
                   'block frees "\n')
        runner = '    def _runner_pass(self,call,output):\n        return call(output)\n'
        cases = ((fixed, runner, True), (refusal, '', False), (fixed, '', False), (refusal + fixed, runner, False),
                 ('', runner, False), (fixed, None, False), (None, runner, False))
        for k, (scheduler, probe, ok) in enumerate(cases):
            with tempfile.TemporaryDirectory(prefix='swe32-asyncfix-') as tmp:
                value = c.async_fix_feature(self.tree(tmp, scheduler, probe))
            self.assertEqual(value is not None, ok, k)
        # 2026-11-16: the third fix (async-maxtok-1112 13ee618): without its resolve line, or without the file, None
        for k, resolve in enumerate(('    if i.request in scheduled:\n', None, '')):
            with tempfile.TemporaryDirectory(prefix='swe32-asyncfix-') as tmp:
                self.assertIsNone(c.async_fix_feature(self.tree(tmp, cases[0][0], cases[0][1], resolve)), k)
        self.assertEqual(PINS['runtime_features'][c.ASYNC_FIX_KEY],
                         dict(scheduler=c.ASYNC_FIX_SCHEDULER, probe=c.ASYNC_FIX_PROBE, by_pool=c.ASYNC_FIX_BY_POOL,
                              runner_pass='_runner_pass',
                              resolve='%s with %r (async-maxtok-1112 13ee618)' % (c.ASYNC_FIX_RESOLVE,
                                                                                  c.ASYNC_FIX_RESOLVE_MARK)))

    def test_async_plans_are_blocked_without_the_fixes(self):
        asynchronous = [n for n in PLAN_ORDER if c.cfg(PINS)['plans'][n].get(c.ASYNC_KEY) is True]
        self.assertEqual(asynchronous, ['smoke', 'smoket'] + ASYNC_ON)  # 2026-11-16: the async pair, the second half
        # a pin with only the two earlier fixes (the seal before 2026-11-16) blocks every async plan
        two = copy.deepcopy(PINS)
        two['runtime_features'][c.ASYNC_FIX_KEY].pop('resolve')
        self.assertEqual([n for n in asynchronous if c.async_fix_block(two, n)], asynchronous)
        self.assertIsNone(c.async_fix_block(two, 'rfillsync32t'))
        self.assertTrue(all(c.plan_blocked(PINS, n) is None for n in PLAN_ORDER
                            if n not in GOALEDGE))  # (11-16 phase B: keepedgegoal* wait for goal-rollback-1116)
        before = copy.deepcopy(PINS)  # ea18256 / f4ccd4b: no async fixes
        before['runtime_features'][c.ASYNC_FIX_KEY] = None
        for name in PLAN_ORDER:
            if name in EDGE1116:  # (blocked by keep_rebuild_block on every pin without keep-rebuild-1116)
                continue
            reason = c.plan_blocked(before, name)
            self.assertEqual(reason is not None, name in asynchronous, name)
            if reason:
                self.assertIn('lacks the async fixes', reason)
                self.assertIn('3499729', reason)
                c.validate_plan(before, name)  # a blocked plan still validates (time, memory, the launcher's refusals)
        off = copy.deepcopy(before)
        off['config']['plans']['smoke'][c.ASYNC_KEY] = False  # the async switch off: no block (thinking-off, sync)
        self.assertIsNone(c.plan_blocked(off, 'smoke'))
        kv = copy.deepcopy(before)  # a KV block comes first (ctx25's reason), the async block after it
        kv['runtime_features'][c.QWEN_KV_KEY] = [8, 16, 24, 32]
        self.assertIn('--kv-cache-gib must be one of', c.plan_blocked(kv, 'smoke'))
        import inspect
        self.assertIn('async_fix_block(pins, name)', inspect.getsource(c.plan_blocked))

    def test_the_plan_table_and_the_entry_template(self):
        import build
        before = copy.deepcopy(PINS)
        before['runtime_features'][c.ASYNC_FIX_KEY] = None
        rows = {line.split('=', 1)[0].strip().strip("'"): line for line in build.plan_table(before).splitlines()[1:-1]}
        self.assertIn("blocked='the sealed runtime lacks the async fixes", rows['smoke'])
        self.assertIn("blocked='the sealed runtime lacks the async fixes", rows['smoket'])
        self.assertIn("blocked=''", rows['ctx24'])  # (a plan with its own settings, not async: unchanged)
        self.assertNotIn('blocked=', rows['full'])
        now = {line.split('=', 1)[0].strip().strip("'"): line for line in build.plan_table(PINS).splitlines()[1:-1]}
        self.assertIn("blocked=''", now['smoke'])
        self.assertIn("blocked=''", now['smoket'])
        template = (ROOT / 'run_one_connection.ps1.in').read_text(encoding='utf-8')
        for item in ("$AsyncBlocked = ($AsyncValue -eq $true -and -not $Features.async_fix.resolve)",
                     "if ($AsyncBlocked -and -not $Blocked) { throw",
                     "if ($Blocked -and -not $KvBlocked -and -not $AsyncBlocked -and -not $KeepBlocked -and -not $GoalRollbackBlocked) { throw",
                     "if ($Blocked) { throw ('plan '+$Plan+' is blocked: '"):
            self.assertIn(item, template)
        self.assertLess(template.index('$AsyncBlocked ='), template.index('if ($SelfTest) {'))


class ThinkingRecentFill(unittest.TestCase):
    """2026-11-06: the recent-window policies of runtime 60c5d21 (recent_fill, recent_tools; required) and the
    thinking-on comparisons (observation_query, vllm25's goal_query), one server per ghx4 batch job, all with the plan
    switch thinking (serve.py --enable-thinking, the agents' chat_template_kwargs, the thinking-on header and record in
    the checks), async scheduling off, everything else ctx25's / vllm25's servers; time from an estimate (above 2 h)."""
    SERVERS = dict(recentt=('recent262t', 'recent_fill'), rtoolst=('rtools262t', 'recent_tools'),
                   obst=('obs262t', 'observation_query'), goalt=('goalvllmt', 'goal_query'),
                   akvt=('akv262t', 'akv_rule'))

    def test_the_plans(self):
        for name, (label, policy) in self.SERVERS.items():
            plan, (server,) = c.cfg(PINS)['plans'][name], c.plan_servers(PINS, name)
            self.assertEqual((server['label'], server['policy'], server['tasks'], server['step_limit'],
                              server['agent_cap_seconds'], server['agent_min_seconds'], server['kind'],
                              server[c.THINKING_KEY]), (label, policy, 32, 120, 7200, 4500, 'runtime', True), name)
            self.assertEqual((plan['role'], plan['partition'], plan['eval_wait_seconds'], plan['thinking'],
                              c.ASYNC_KEY in plan, plan['server']),
                             ('validation', 'ghx4', 600, True, False, dict(kv_cache_gib=25, max_model_len=262144,
                                                                           dram_arena_gib=70)), name)
            self.assertEqual(c.plan_settings(PINS, name), c.plan_settings(PINS, 'ctx25'))
            self.assertTrue(plan['authorization'].startswith('AUTHORIZE_SWE32_%s_32TASKS_16SESSIONS_262K_KV25_THINKING_ON_'
                                                             % name.upper()) and 'BATCH_REUSED_V9_ENV_ONE_GH200' in
                            plan['authorization'], name)
        # the same servers as ctx25's obs262 / vllm25's goalvllm but the label, the cap / minimum and thinking
        for name, reference in (('obst', ('ctx25', 1)), ('goalt', ('vllm25', 0))):
            mine, theirs = dict(c.plan_servers(PINS, name)[0]), dict(c.plan_servers(PINS, reference[0])[reference[1]])
            for k in ('label', 'agent_cap_seconds', 'agent_min_seconds', c.THINKING_KEY):
                mine.pop(k), theirs.pop(k, None)
            self.assertEqual(mine, theirs, name)
            args = c.launch_arguments(PINS, c.plan_servers(PINS, name)[0], REWRITER)
            # serve.py's arguments are the reference server's plus --enable-thinking (after the speed switches)
            self.assertEqual(args.count('--enable-thinking'), 1, name)
            self.assertEqual([a for a in args if a != '--enable-thinking'],
                             c.launch_arguments(PINS, c.plan_servers(PINS, reference[0])[reference[1]], REWRITER), name)
        # the window policies take random_fill's flags: the active budget, no admission flag, no rewriter
        rfill = c.plan_servers(PINS, 'rfill')[0]
        for name in ('recentt', 'rtoolst'):
            (s,) = c.plan_servers(PINS, name)
            self.assertEqual((s['policy_args'], s['admission'], s['rewriter']), (rfill['policy_args'], [], False))
            args = c.launch_arguments(PINS, s, REWRITER)
            self.assertEqual(args[-3:], ['--enable-thinking', '--active-token-budget', '8192'])
            self.assertNotIn('--async-scheduling', args)
            self.assertNotIn('--goal-rewriter-model', args)
        self.assertEqual(ORDER1112[-12 - len(NEW64 + NEW1112):-len(NEW64 + NEW1112)],
                         THINK + AKVS + SMOKET + THINKI + FULLT)  # 11-09, 11-11 (then the 11-12 plans)

    def test_time_cost_and_the_estimate(self):
        for name in THINK:
            fit, cost = c.validate_plan(PINS, name), c.plan_cost(PINS, name)
            extra = 300 if name == 'goalt' else 0  # the vLLM child's start, as vllm25
            # 840 s of job reserves + (420 s start + the 7,200-s cap + 480 s stop / checks) (+ 300 s)
            self.assertEqual((fit['worst_case_seconds'], fit['with_eval_wait_seconds'], fit['time_seconds']),
                             (840 + 420 + 7200 + 480, 840 + 600 + 4500 + 480, 8940 + extra), name)
            self.assertLessEqual(cost['limit_over_expected'], c.MAX_LIMIT_OVER_EXPECTED)
            self.assertGreater(fit['time_seconds'], 2 * 3600)  # above the 2-h target: stated in the plans
            memory = c.memory_fit(PINS, name)
            self.assertEqual((memory['need_gib'], memory['headroom_gib'], memory['fits']), (105.29, 5.61, True))
            self.assertIn('ESTIMATE', c.cfg(PINS)['plans'][name]['about'])

    def test_model_override_and_expectation(self):
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m/qwen3-0.6b', config_sha256='0' * 64)
        record = PINS['runtime_features'][c.THINKING_KEY]['record']
        self.assertEqual(record['header_text'], '<|im_start|>assistant\n<think>\n')
        for name in PLAN_ORDER:
            for s in c.plan_servers(PINS, name):
                text = c.model_override_text(PINS, s, 23004)
                on = name in THINK + AKVS + SMOKET + THINKI + FULLT + NEW64 + NEW1112 + NEW1116 + LEDGERT + PHASEB \
                    + P150  # (11-12, 11-16: all on; 11-14: ledger*t; 11-17: the 150-task plans)
                self.assertEqual('    extra_body: {"chat_template_kwargs": {"enable_thinking": true}}' in text.splitlines(), on, name)
                self.assertEqual('--enable-thinking' in (c.launch_arguments(PINS, s, REWRITER) if s['kind'] == 'runtime'
                                                         else []), on and s['kind'] == 'runtime', name)
                if s['kind'] != 'runtime':
                    continue
                expect = c.server_expectation(PINS, s, gate, 'm', ['t0'])
                self.assertEqual(expect.get(c.THINKING_KEY, 'absent'), record if on else 'absent', name)
                self.assertEqual(expect.get('window_pin'), dict(order=c.WINDOW_ORDERS[s['policy']])
                                 if s['policy'] in c.WINDOW else None, name)

    def test_refusals(self):
        def refused(change, pattern, plan='recentt'):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, plan)
        p = lambda pins: pins['config']['plans']['recentt']
        refused(lambda pins: p(pins).update(thinking='yes'), 'thinking must be true or false')
        refused(lambda pins: p(pins)['servers'][0].update(thinking=True), 'thinking is a plan switch')
        refused(lambda pins: pins['runtime_features'].update(thinking=None), 'no --enable-thinking')
        refused(lambda pins: pins['runtime_features'].update(window=None), 'does not serve recent_fill')
        refused(lambda pins: pins['config']['policies']['recent_fill'].update(rewriter=True),
                'only goal_query starts the rewriter')
        refused(lambda pins: pins['config']['policies']['recent_fill'].update(policy_args=[]),
                'takes --active-token-budget')
        with self.assertRaisesRegex(RuntimeError, 'ask for --enable-thinking'):
            c.check_features(dict(PINS['runtime_features'], thinking=None), PINS['config'])
        with self.assertRaisesRegex(RuntimeError, 'run recent_fill / recent_tools'):
            c.check_features(dict(PINS['runtime_features'], window=None), PINS['config'])
        config = copy.deepcopy(PINS['config'])
        for name in THINK + AKVS + SMOKET + THINKI + FULLT + NEW64 + NEW1112 + NEW1116 + LEDGERT + PHASEB + P150 \
                + GATE2:  # (smoket 11-09, 11-11 / 11-12 / 11-16 / 11-17: thinking on; 11-17: gate 2)
            del config['plans'][name]
            config['plan_order'].remove(name)
        c.check_features(dict(PINS['runtime_features'], thinking=None, window=None, akv=None), config)

    def test_the_akv_plan(self):
        # akv-rule (2026-11-07): akv262t = rtoolst's server with akv_rule: no flag at all (no budget), thinking on
        (akv,), (rtools,) = c.plan_servers(PINS, 'akvt'), c.plan_servers(PINS, 'rtoolst')
        self.assertEqual((akv['policy_args'], akv['admission'], akv['rewriter']), ([], [], False))
        self.assertEqual({k: v for k, v in akv.items() if k not in ('label', 'policy', 'policy_args')},
                         {k: v for k, v in rtools.items() if k not in ('label', 'policy', 'policy_args')})
        args = c.launch_arguments(PINS, akv, REWRITER)
        self.assertEqual(args, [a for a in c.launch_arguments(PINS, rtools, REWRITER)
                                if a not in ('--active-token-budget', '8192')])
        self.assertEqual(args[-1], '--enable-thinking')
        self.assertEqual(PINS['runtime_features']['akv'], dict(policy='akv_rule', pin=c.AKV_PIN, evidence_key='window'))
        self.assertEqual(c.AKV_PIN, dict(rule='replies_and_latest_tool_responses_v1', latest_tool_responses=12))
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m/qwen3-0.6b', config_sha256='0' * 64)
        expect = c.server_expectation(PINS, akv, gate, 'm', ['t0'])
        import smoke_checks as sc
        contract = sc.wanted_contract('akv_rule', expect)
        self.assertEqual((expect['akv_pin'], contract['akv_rule'], contract['active_ratio'],
                          contract['active_token_budget'], contract['switch_threshold_tokens'], contract['goal_rewriter']),
                         (c.AKV_PIN, c.AKV_PIN, None, None, 0, None))
        self.assertEqual(sc.wanted_rewriter_record('akv_rule', expect),
                         dict(used=False, reason=c.GOAL_REWRITER_UNUSED_AKV, requested_model=None))
        self.assertEqual(c.GOAL_REWRITER_UNUSED_AKV, sc.GOAL_REWRITER_UNUSED_AKV)
        self.assertIn('stall', c.cfg(PINS)['plans']['akvt']['about'])

        def refused(change, pattern):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, 'akvt')
        refused(lambda pins: pins['runtime_features'].update(akv=None), 'does not serve akv_rule')
        refused(lambda pins: pins['config']['policies']['akv_rule'].update(policy_args=['--active-token-budget', '8192']),
                'akv_rule(: | takes no )--active-token-budget')     # (the budget check refuses it first)
        refused(lambda pins: pins['config']['policies']['akv_rule'].update(rewriter=True),
                'only goal_query starts the rewriter')
        with self.assertRaisesRegex(RuntimeError, 'run akv_rule'):
            c.check_features(dict(PINS['runtime_features'], akv=None), PINS['config'])

    def test_the_akvs_plan(self):
        # akvs (2026-11-07): akvt's server on the first 8 instances with the plan's own K = 4: --max-running-sessions 4,
        # the layout / pool / capture sizes at 4, the agents' cap 4, the arena 60 GiB (70 would hold every session)
        (s,), (akvt,) = c.plan_servers(PINS, 'akvs'), c.plan_servers(PINS, 'akvt')
        plan = c.cfg(PINS)['plans']['akvs']
        self.assertEqual((s['label'], s['policy'], s['tasks'], s['step_limit'], s['agent_cap_seconds'],
                          s['agent_min_seconds'], s[c.THINKING_KEY], s['policy_args'], s['admission'], s['rewriter']),
                         ('akvs262t', 'akv_rule', 8, 120, 7200, 4500, True, [], [], False))
        self.assertEqual(plan['server'], dict(kv_cache_gib=25, max_model_len=262144, dram_arena_gib=60,
                                              max_running_sessions=4))
        self.assertEqual((c.plan_running(PINS, 'akvs'), c.plan_running(PINS, 'akvt'), c.plan_running(PINS, 'full')),
                         (4, 16, 16))
        args = c.launch_arguments(PINS, s, REWRITER)
        theirs = c.launch_arguments(PINS, akvt, REWRITER)
        self.assertEqual(args[args.index('--max-running-sessions') + 1], '4')
        self.assertEqual(args[args.index('--dram-arena-gib') + 1], '60')
        swap = {'--max-running-sessions': '4', '--dram-arena-gib': '60'}
        self.assertEqual(args, [swap.get(theirs[k - 1], a) if k else a for k, a in enumerate(theirs)])
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m/qwen3-0.6b', config_sha256='0' * 64)
        expect = c.server_expectation(PINS, s, gate, 'm', ['t%d' % k for k in range(8)])
        self.assertEqual((expect['running'], expect['max_num_seqs'], expect['akv_pin'], expect['layout']['pools'],
                          expect['capture_sizes']), (4, 4, c.AKV_PIN, [24229, 28], [1, 2, 4, 8]))
        self.assertEqual(c.pool_spec(PINS, s, 23004, 1800000000)[1]['max_running'], 4)
        self.assertEqual(c.pool_spec(PINS, akvt, 23004, 1800000000)[1]['max_running'], 16)
        fit, cost, memory = c.validate_plan(PINS, 'akvs'), c.plan_cost(PINS, 'akvs'), c.memory_fit(PINS, 'akvs')
        self.assertEqual((fit['time_seconds'], plan['time'], plan['expected_seconds']), (8940, '02:29:00', 6600))
        self.assertLessEqual(cost['gpu_hours_limit'], 2.5)
        self.assertEqual((memory['arena_gib'], memory['need_gib'], memory['fits']), (60, 93.57, True))
        self.assertEqual(c.attention_fit(PINS, c.LAYOUT, 4, s)['fa_blocks'], 24229)
        self.assertTrue(plan['authorization'].startswith('AUTHORIZE_SWE32_AKVS_8TASKS_4SESSIONS_'))
        self.assertIn('ESTIMATE', plan['about'])
        # the arena of 70 GiB holds every one of 4 sessions at the context limit: serve.py's refusal, here first
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['akvs']['server']['dram_arena_gib'] = 70
        with self.assertRaisesRegex(RuntimeError, 'less than every session'):
            c.validate_plan(pins, 'akvs')

    def test_the_akv_check(self):
        import smoke_checks as sc
        # P = 16; mandatory Pages 0-5 (prefix 96); 14 turns: replies of 20 tokens, observations of 64 (Page-aligned
        # starts after a 20-token reply: [116, 180), [200, 264), ...); local turn 14: 13 sealed spans, 11 kept.
        ends, spans, x = [96], [], 116
        for k in range(1, 15):
            ends.append(x)
            spans.append([x, x + 64])
            x += 84
        P, M, stable = 16, 6, ends[14]
        ends, spans, n = ends[:14], spans[:13], stable // 16
        kept, keep = sc.akv_kept(ends, spans, stable, P, M, n)
        dropped = set(range(M, n)) - kept
        self.assertEqual((keep, dropped), (11, {8, 9, 10, 13, 14, 15}))
        self.assertEqual(sc.akv_kept(ends, spans, stable, P, M, n, 14)[0], set(range(M, n)))     # 13 sealed kept
        recent = sorted(p for p in kept if p >= n - 10)
        order = sc.window_order('akv_rule', sorted(set(range(M, n)) - set(recent)), kept)
        filled = order[:len(kept) - len(recent)]
        selected = sorted(set(range(M)) | kept)
        window = sc.window_record('akv_rule', n, M, selected, order, len(filled), ends, spans, stable, P, kept)
        self.assertEqual({k: window[k] for k in ('kept_tool_responses', 'dropped_tool_responses', 'dropped_pages',
                                                  'selected_tokens', 'first_tool_position', 'latest_tool_responses')},
                         dict(kept_tool_responses=11, dropped_tool_responses=2, dropped_pages=6,
                              selected_tokens=(n - 6) * 16, first_tool_position=spans[2][0], latest_tool_responses=12))
        selection = dict(recent_pages=recent, retrieved_pages=filled, projection_sha256=None,
                         selector_config_hash='s' * 64, capacity_pages=M + len(kept), selected_pages=selected,
                         window=window)
        native = dict(stable_end=stable, query=dict(body=dict(selection=dict(selection), goal=dict(forward=False),
                                                              ranking=list(range(M)) + recent + order)))
        row = dict(query=dict(query_token_sha256=None))
        value = sc.check_akv(row, native, selection, n, M, (ends, spans), P)
        self.assertEqual((value['akv_kept_tool_responses'], value['akv_dropped_pages']), (11, 6))
        # 11 instead of 12 (one more span dropped), reply Pages dropped, a budget capacity, the reversed fill and a
        # forged record are refused
        eleven = sc.akv_kept(ends, spans, stable, P, M, n, 11)[0]
        for bad in (dict(selection, retrieved_pages=[p for p in filled if p in eleven]),
                    dict(selection, retrieved_pages=[p for p in filled if p not in {6, 7}]),
                    dict(selection, capacity_pages=512), dict(selection, retrieved_pages=list(reversed(filled))),
                    dict(selection, window=dict(window, kept_tool_responses=12))):
            with self.assertRaises(sc.CheckFailed):
                sc.check_akv(row, native, bad, n, M, (ends, spans), P)

    def test_the_window_check(self):
        import smoke_checks as sc
        # P = 16; mandatory Pages 0-6 (prefix 100), turns observing [130, 200) and [260, 300), stable end 330 (n = 20);
        # the replies [100, 130), [200, 260), [300, 330) start with a 3-token header; thinking on, </think> (id 5) at
        # 119 and 244 (calls from 120 and 245), none in the last reply (unclosed: cut while thinking, no call: its
        # start is its end 330 since ledger-fixes-1116 / a2f9e36; before, from 303 after the header).
        ends, spans, stable, P, n, M = [100, 130, 260], [[130, 200], [260, 300]], 330, 16, 20, 7
        prompt = [1] * 340
        prompt[119] = prompt[244] = 5
        on, off = (3, (5,)), (3, None)
        self.assertEqual([sc.call_start(prompt, a, b, *on) for a, b in sc.reply_spans(ends, spans, stable)],
                         [120, 245, 330])
        calls, reasoning = sc.window_classes(ends, spans, stable, P, M, n, prompt, *on)
        self.assertEqual((calls, reasoning), ({7, 8, 15, 16}, {13, 14, 19}))  # (Page 18: the observation's, TOOL)
        # Thinking off: the call is the whole reply after the header (a </think> in it does not count): no REASONING.
        self.assertEqual(sc.window_classes(ends, spans, stable, P, M, n, prompt, *off),
                         ({7, 8, 12, 13, 14, 15, 16, 18, 19}, set()))
        self.assertEqual(sc.window_order('recent_tools', range(7, 18), calls, reasoning),
                         [16, 15, 8, 7, 17, 12, 11, 10, 9, 14, 13])
        self.assertEqual(sc.window_order('recent_fill', range(7, 18)), list(range(17, 6, -1)))
        for policy, filled in [(x, f) for x in c.WINDOW for f in (3, 6, 10)]:
            split = on if policy == 'recent_tools' else None
            classes = (calls, reasoning) if policy == 'recent_tools' else (frozenset(), frozenset())
            order = sc.window_order(policy, range(7, 18), *classes)
            selected = sorted(set(range(M)) | {18, 19} | set(order[:filled]))
            window = sc.window_record(policy, n, M, selected, order, filled, ends, spans, stable, P, *classes)
            selection = dict(recent_pages=[18, 19], retrieved_pages=order[:filled], projection_sha256=None,
                             selector_config_hash='s' * 64, window=window)
            native = dict(stable_end=stable, prompt_token_ids=prompt,
                          query=dict(body=dict(selection=dict(selection), goal=dict(forward=False),
                                               ranking=list(range(M)) + [18, 19] + order)))
            row = dict(query=dict(query_token_sha256=None))
            capacity = M + 2 + filled
            value = sc.check_window(policy, row, native, selection, n, M, capacity, (ends, spans), P, split)
            self.assertEqual(value['window_filled'], filled)
            if policy == 'recent_tools':
                self.assertEqual((window['call_pages'], window['tool_pages'], window['reasoning_pages']), (4, 5, 2))
                self.assertEqual((value['window_calls_truncated'], value['window_calls_kept'], value['window_tools_kept'],
                                  value['window_reasoning_kept']),
                                 {3: (True, 3, 0, 0), 6: (False, 4, 2, 0), 10: (False, 4, 5, 1)}[filled])
                # TOOL before CALL (precedence) or the classes in another order are refused
                tools_first = sc.window_order('recent_tools', range(7, 18), calls - {8, 16, 18}, reasoning)
                thought_first = ([p for p in order if p in calls] + [p for p in order if p in reasoning]
                                 + [p for p in order if p not in calls and p not in reasoning])
                for wrong in (tools_first, thought_first):
                    if wrong[:filled] != order[:filled]:
                        with self.assertRaises(sc.CheckFailed):
                            sc.check_window(policy, row, native, dict(selection, retrieved_pages=wrong[:filled]), n, M,
                                            capacity, (ends, spans), P, split)
                with self.assertRaises(sc.CheckFailed):  # no split: the classes cannot be checked
                    sc.check_window(policy, row, native, selection, n, M, capacity, (ends, spans), P)
            # a reversed fill order (oldest first) and a forged window record are refused
            for bad in (dict(selection, retrieved_pages=list(reversed(order))[:filled]),
                        dict(selection, window=dict(window, covered_turns=window['covered_turns'] + 1))):
                with self.assertRaises(sc.CheckFailed):
                    sc.check_window(policy, row, native, bad, n, M, capacity, (ends, spans), P, split)

    def test_the_thinking_check(self):
        import swe_checks
        record = PINS['runtime_features'][c.THINKING_KEY]['record']
        row = lambda local, history, split=(5, 1, 3): dict(request_id='r', stage_b=dict(
            local_turn=local, rebuild=False, prompt_end=100, computed_end=100 + sum(split) - 1),
            thinking=dict(reasoning_tokens=split[0], think_end_tokens=split[1], content_tokens=split[2],
                          closed=split[1] == 1, reasoning_chars=10, content_chars=4, history=history))
        value = swe_checks.check_thinking([row(0, 'fresh'), row(1, 'append')], dict(thinking=record))
        self.assertEqual(value['histories'], dict(fresh=1, append=1))
        for bad in ([row(1, 'fresh')], [row(0, 'append')], [dict(row(1, 'append'), thinking=None)]):
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_thinking(bad, dict(thinking=record))
        with self.assertRaises(swe_checks.CheckFailed):  # a thinking-off server's row with a record
            swe_checks.check_thinking([row(0, 'fresh')], dict(thinking=None))
        self.assertEqual(swe_checks.check_thinking([dict(request_id='r')], dict(thinking=None))['thinking'], False)

    def test_entry_script_dry_run(self):
        entry = ROOT / 'run_one_connection.ps1'
        run = lambda *args: subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                            str(entry)] + list(args), capture_output=True, text=True, timeout=300)
        for name, (label, policy) in self.SERVERS.items():
            phrase = c.cfg(PINS)['plans'][name]['authorization']
            value = run('-Plan', name, '-RunLabel', 'r1', '-Authorize', '-Authorization', phrase, '-DryRun')
            self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
            minutes = 154 if name == 'goalt' else 149
            for item in ('DRY_RUN connections=0', 'plan=%s role=validation variant_of=- ' % name,
                         'servers=%s:%s:32tasks:120steps:7200s' % (label, policy),
                         ':thinking ', 'partition=ghx4 time=%02d:%02d:00 ' % divmod(minutes, 60),
                         'release_check=PINNED receipt_result_sha256=' + RECEIPT_PIN['result_line_sha256'] + ' ',
                         'mem=113560M ', 'job_name=swe32-%s ' % name, 'mail_user=cao58@illinois.edu mail_type=BEGIN,END,FAIL ',
                         'plan_server=own kv_cache_gib=25 max_model_len=262144 plan_dram_arena_gib=70 ',
                         ' thinking=%s:on' % label):
                self.assertIn(item, value.stdout, name)
            self.assertNotIn('async_scheduling=', value.stdout)
            self.assertNotIn('NOT_SUBMITTABLE', value.stdout)
        other = run('-Plan', 'vllm25', '-Authorize', '-Authorization', c.cfg(PINS)['plans']['vllm25']['authorization'],
                    '-DryRun')
        self.assertEqual(other.returncode, 0, other.stdout + other.stderr)
        self.assertNotIn('thinking=', other.stdout)


class EntryAndDeploy(unittest.TestCase):
    def test_entry_script_self_test_and_refusals(self):
        entry = ROOT / 'run_one_connection.ps1'
        run = lambda *args: subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                            str(entry)] + list(args), capture_output=True, text=True, timeout=300)
        value = run('-Plan', 'full', '-SelfTest')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        self.assertIn('LOCAL_SELFTEST_PASS connections=0', value.stdout)
        phrase = c.cfg(PINS)['plans']['goalb4']['authorization']
        value = run('-Plan', 'goalb4', '-Authorize', '-Authorization', phrase, '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('DRY_RUN connections=0', 'role=table1 ', 'partition=ghx4 ', 'time=01:28:00 ', 'expected_seconds=4057 ',
                     'billing_factor=1 gpu_hours_limit=1.47 gpu_hours_expected=1.13 ',
                     'release_check=PINNED receipt_result_sha256=' + RECEIPT_PIN['result_line_sha256'] + ' ',
                     'servers=goalb4:goal_query:32tasks:120steps:3540s:batch4 ',
                     'worst_case_seconds=5280 with_eval_wait_seconds=4570 cap_after_wait_fits=false ',
                     'mem=113560M dram_arena_gib=70 memory_need_gib=105.29 memory_spare_gib=5.61 '
                     'startup_spike=ELIMINATED ', 'job_name=swe32-goalb4 ',
                     'decode_fastpath=on decode_fastpath2=on decode_fastpath3=on deferred_completion_io=on '
                     'language_model_only=on connector_pool_gib=1 release_idle_pages=off '
                     'goal_rewriter_batch=4 goal_rewriter_batch_tokens=16384 ',
                     'tasks=CONFIRMED(32 ', 'goal_text=goals.jsonl', 'mail_type=BEGIN,END,FAIL'):
            self.assertIn(item, value.stdout)
        value = run('-Plan', 'goalb4', '-AfterJob', '777', '-Authorize', '-Authorization', phrase, '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)  # a batch job may queue after another
        self.assertIn('after_job=777', value.stdout)
        phrase_i = c.cfg(PINS)['plans']['goalb4i']['authorization']
        value = run('-Plan', 'goalb4i', '-Authorize', '-Authorization', phrase_i, '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('role=interactive variant_of=goalb4 ', 'partition=ghx4-interactive ', 'time=01:28:00 ',
                     'billing_factor=2 gpu_hours_limit=2.93 gpu_hours_expected=2.25 ', 'job_name=swe32-goalb4i '):
            self.assertIn(item, value.stdout)
        for args in (['-Plan', 'full', '-Authorize', '-Authorization', phrase, '-DryRun'],
                     ['-Plan', 'goalb4', '-Authorize', '-Authorization', phrase_i, '-DryRun'],
                     ['-Plan', 'goalb4i', '-Authorize', '-Authorization', phrase, '-DryRun'],
                     ['-Plan', 'goalb4i', '-AfterJob', '777', '-Authorize', '-Authorization', phrase_i, '-DryRun'],
                     ['-Plan', 'goalb4', '-Authorize', '-Authorization', phrase.replace('_BATCH_', '_INTERACTIVE_'), '-DryRun'],
                     ['-Plan', 'goalb4', '-Authorize', '-Authorization', phrase.replace('_88MIN_', '_108MIN_'), '-DryRun'],
                     ['-Plan', 'goalb4', '-AfterJob', 'x7', '-Authorize', '-Authorization', phrase, '-DryRun'],
                     ['-Plan', 'goalb4', '-Authorization', phrase],
                     ['-Plan', 'all', '-SelfTest'], ['-Plan', 'nope', '-SelfTest'],
                     ['-Plan', 'full', '-RunLabel', 'x1', '-SelfTest']):
            value = run(*args)
            self.assertNotEqual(value.returncode, 0, args)
    def test_deploy_manifest_equals_the_pin(self):
        value = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                str(ROOT / 'tbench' / 'deploy_tbench.ps1'), '-SelfTest'], capture_output=True, text=True,
                               timeout=300)
        self.assertEqual(value.returncode, 0, value.stderr)
        self.assertIn('MANIFEST_SHA256=' + PINS['tbench_half']['manifest_sha256'], value.stdout)
        self.assertIn('tasks/swe32-tasks.json', value.stdout)


class HarnessFix1110(unittest.TestCase):
    """2026-11-10 (perf review FINDINGS 1.1, 1.2, 2.8, section 3; HARNESS-FIX-BRIEF): the clean cells next to the
    steady ones, the decode time lost by the ITL histogram, the join's partial-run tokens, the tables' titles, the
    scenarios' own server ports and e2e_swe32.py --jobs (order and verdicts as with one process)."""
    LABELS = ('0.01', '0.025', '0.05', '0.075', '0.1', '0.15', '0.2', '0.3', '0.4', '0.5', '0.75', '1.0', '2.5')

    def timeline(self, intervals):
        """A synthetic batch timeline: per interval (running at its start, running at its end, [step seconds]); every
        step adds one ITL observation per running request (the start count)."""
        import measure
        points, t, s, n, buckets, generated = [], 0.0, 0.0, 0.0, dict.fromkeys(self.LABELS, 0.0), 0.0

        def point(running):
            return dict(index=len(points), t_s=round(t, 3), ok=True, running=float(running), waiting=0.0, kv_usage=0.1,
                        generated=generated, preemptions=0.0, finished=0.0, itl_sum=s, itl_count=n,
                        itl_slow=n - buckets['0.5'], itl_buckets=dict(buckets))
        points.append(point(intervals[0][0]))
        for start, end, steps in intervals:
            for step in steps:
                s += step * start
                n += start
                generated += start
                for le in self.LABELS:
                    if step <= float(le):
                        buckets[le] += start
            t += 0.25
            points.append(point(end))
        self.assertEqual(set(measure.SAMPLE_KEYS) - set(points[0]), set())
        return points

    def test_clean_bound_follows_the_batch_step_and_is_capped(self):
        import measure
        got = {ref: measure.clean_bound(self.LABELS, ref) for ref in (20.0, 25.0, 26.0, 42.0, 80.0)}
        self.assertEqual(got, {20.0: '0.05', 25.0: '0.05', 26.0: '0.075', 42.0: '0.1', 80.0: '0.15'})
        self.assertEqual((measure.CLEAN_FACTOR, measure.CLEAN_CAP_SECONDS, measure.STALL_SECONDS), (2.0, 0.15, 0.5))

    def test_sample_values_keep_the_itl_buckets_summed_over_series(self):
        import measure
        lines = ['# TYPE vllm:inter_token_latency_seconds histogram']
        for engine, counts in (('0', (1, 3, 4)), ('1', (2, 2, 5))):
            for le, count in zip(('0.05', '0.5', '+Inf'), counts):
                lines.append('vllm:inter_token_latency_seconds_bucket{engine="%s",le="%s"} %d' % (engine, le, count))
            lines.append('vllm:inter_token_latency_seconds_sum{engine="%s"} 1.5' % engine)
            lines.append('vllm:inter_token_latency_seconds_count{engine="%s"} %d' % (engine, counts[-1]))
        values = measure.sample_values('\n'.join(lines) + '\n')
        self.assertEqual(values['itl_buckets'], {'0.05': 3.0, '0.5': 5.0})
        self.assertEqual((values['itl_count'], values['itl_slow']), (9.0, 4.0))

    def test_the_clean_cells_leave_an_admission_stall_out_and_the_steady_cells_stay_as_they_were(self):
        import measure
        plain = [0.025] * 10
        stalled = [0.025] * 9 + [0.3]  # a 0.3-s admission: below STALL_SECONDS, so `steady` keeps it
        intervals = [(4, 4, plain)] * 20 + [(4, 4, stalled)] * 5 + [(4, 8, plain)] + [(8, 8, [0.03] * 10)] * 10
        profile = measure.batch_profile(self.timeline(intervals))
        cells = {c['batch']: c for c in profile['batches']}
        four, eight = cells[4], cells[8]
        S4, N4 = 4 * (20 * 10 * 0.025 + 5 * (9 * 0.025 + 0.3)), 4 * 250
        self.assertEqual((four['intervals'], four['steady_intervals'], four['steady_itl_ms']),
                         (25, 25, round(S4 / N4 * 1000, 3)))  # the existing column, as before
        self.assertEqual((four['median_interval_itl_ms'], four['clean_bound_s'], four['clean_intervals'],
                          four['clean_itl_ms'], four['clean_s']), (25.0, 0.05, 20, 25.0, 5.0))
        self.assertEqual((eight['clean_bound_s'], eight['clean_intervals'], eight['clean_itl_ms']), (0.075, 10, 30.0))
        self.assertEqual(profile['stall_intervals'], 0)
        lost = profile['decode_lost']
        # only the stalls are lost; the 4 -> 8 interval at the mean count 6 (27.5 ms) vs its own 25 ms steps: -2.5 ms x 40
        self.assertAlmostEqual(lost['lost_s'], 5 * 4 * (0.3 - 0.025) - 40 * 0.0025, places=6)
        self.assertAlmostEqual(lost['lost_high_s'], 5 * 4 * (0.3 - 0.025), places=6)  # that interval at 25 ms
        S = S4 + 40 * 0.025 + 10 * 8 * 10 * 0.03
        self.assertAlmostEqual(lost['lost_low_s'], S - (N4 + 40 + 800) * 0.03, places=6)
        self.assertTrue(lost['lost_share_low'] <= lost['lost_share'] <= lost['lost_share_high'])
        self.assertEqual(lost['clean_sizes'], [4, 8])
        # phase1_summary's plain step is the clean cell now
        phase1 = measure.phase1_summary([dict(turn=1, phase1_tokens=5, phase1_mode='mixed', phase1_step_rows=5,
                                              phase1_step_ms=40.0)], profile)
        self.assertEqual(phase1['plain_itl_ms_at_rows']['mean'], 25.0)

    def test_a_clean_run_loses_nothing(self):
        import measure
        profile = measure.batch_profile(self.timeline([(2, 2, [0.02, 0.021, 0.019])] * 8))
        self.assertAlmostEqual(profile['decode_lost']['lost_s'], 0.0, places=9)
        self.assertEqual(measure.batch_profile([])['decode_lost'], None)

    def test_the_reverify_path_reproduces_the_existing_columns(self):
        import measure
        import itl_reverify
        import join_swe32
        timeline = self.timeline([(4, 4, [0.025] * 10)] * 6 + [(4, 4, [0.025] * 9 + [0.3])])
        profile = measure.batch_profile(timeline)
        old = {k: v for k, v in profile.items() if k not in ('clean_rule', 'decode_lost')}
        old['batches'] = [{k: v for k, v in c.items() if k in itl_reverify.OLD_CELL} for c in profile['batches']]
        self.assertEqual(itl_reverify.existing_differ(old, profile), [])
        self.assertEqual(join_swe32.decode_profile(old, timeline), (profile, 'recomputed from the samples (measure.py '
                                                                            'of this harness)'))
        self.assertEqual(join_swe32.decode_profile(profile, None), (profile, 'job'))
        changed = dict(old, stall_intervals=3)
        self.assertEqual(join_swe32.decode_profile(changed, timeline)[1],
                         'recorded (the recomputation differs in the existing columns)')

    def test_join_adds_the_server_tokens_of_an_instance_without_a_trajectory(self):
        import join_swe32
        with tempfile.TemporaryDirectory(prefix='swe32-fake-') as tmp:
            out = Path(tmp)
            (out / 'measurements.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in (
                dict(session=0, gen_tokens=100), dict(session=0, gen_tokens=50), dict(session=2, gen_tokens=70))))
            records = [dict(session=0, join='NO_TRAJECTORY', start=1.0, end=9.0),
                       dict(session=1, join='PASS', start=1.0, end=5.0, completion_tokens=30),
                       dict(session=2, join='NO_TRAJECTORY', start=None)]  # never started: no time, no tokens
            got = join_swe32.all_tokens(dict(kind='runtime'), records, out, 30)
            self.assertEqual(got[0], 180)
            self.assertIn('1 instance(s) without a trajectory (150 tokens)', got[1])
            self.assertEqual(join_swe32.all_tokens(dict(kind='stock'), records, out, 30)[0], None)
            self.assertEqual(join_swe32.all_tokens(dict(kind='runtime'), records[1:], out, 30), (30, 'client'))

    def test_the_tables_keep_the_titles_the_eval_pipeline_reads(self):
        import swe32_tables
        titles = [t for _, t in swe32_tables.TABLE]
        for title in ('Planned', 'Resolved', 'Gen tok (client)', 'Span s', 'tok/s (runner span)', 'Selective turns'):
            self.assertIn(title, titles)
        for old in ('Stall share', 'REWRITE share', 'Queue wait p95 s'):
            self.assertNotIn(old, titles)
        keys = dict(swe32_tables.TABLE)
        self.assertEqual(keys['admission_stall_fraction'], 'Admission-pass share (engine time / span)')
        self.assertIn('decode_lost_pct', keys)
        self.assertIn('throughput_all', keys)
        # the tables of a minimal join (tables.md, table1.csv with the new columns, batch-itl.csv)
        import contextlib
        import io
        cell = dict(batch=4, intervals=25, steady_intervals=25, steady_itl_ms=30.5, steady_tok_s=150.0,
                    median_interval_itl_ms=25.0, clean_bound_s=0.05, clean_intervals=20, clean_itl_ms=25.0,
                    clean_tok_s=160.0)
        server = dict(label='obs', policy='observation_query', instances=2, done=1, resolved=1, accuracy=50.0,
                      completion_tokens=30, tokens_all=180, throughput_all=18.0, throughput_runner=3.0,
                      span_runner_s=10.0, no_trajectory=1, decode_lost_pct=43.1, decode_lost_pct_low=40.8,
                      decode_lost_pct_high=43.4, admission_stall_fraction=0.1952, evaluation='present',
                      kernel_alignment='PASS', kernels='k', batch=dict(batches=[cell]), batch_source='job')
        with tempfile.TemporaryDirectory(prefix='swe32-fake-') as tmp:
            join = Path(tmp) / 'join'
            join.mkdir()
            (join / 'join.json').write_text(json.dumps(dict(run_id='swe32-1-r1', servers=[server], instances=[],
                                                            verdict='JOIN_PASS', deltaai_verdict='SWE32_PASS_VERIFIED',
                                                            provisional=[], tasks_provisional=False)))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(swe32_tables.main(['--out', str(Path(tmp) / 'tables'), str(join)]), 0)
            text = (Path(tmp) / 'tables' / 'tables.md').read_text(encoding='utf-8')
            self.assertIn('"Decode time lost % of ITL sum"', text)
            self.assertIn('| 43.1 | 40.8 | 43.4 |', text)
            rows = (Path(tmp) / 'tables' / 'batch-itl.csv').read_text(encoding='utf-8').splitlines()
            self.assertEqual(rows[1], 'swe32-1-r1,obs,observation_query,job,4,25,25,30.5,150.0,25.0,0.05,20,25.0,160.0')

    def test_each_scenario_takes_its_server_ports_from_its_own_block(self):
        import socket
        import fake_stage
        first = fake_stage.PORT_BLOCK_BASE + 99 * fake_stage.PORT_BLOCK_SIZE
        block = range(first, first + fake_stage.PORT_BLOCK_SIZE)
        factory = fake_stage.block_socket_factory('4100099')
        held = factory(socket.AF_INET, socket.SOCK_STREAM)
        held.bind(('127.0.0.1', 0))
        with factory(socket.AF_INET, socket.SOCK_STREAM) as second:
            second.bind(('127.0.0.1', 0))
            ports = [held.getsockname()[1], second.getsockname()[1]]
        held.close()
        with factory(socket.AF_INET, socket.SOCK_STREAM) as third:
            third.bind(('127.0.0.1', 0))
            ports.append(third.getsockname()[1])
        self.assertTrue(all(p in block for p in ports), ports)
        self.assertEqual(len(set(ports)), 3)  # a released port is not offered again at once
        other = fake_stage.block_socket_factory('4100098')(socket.AF_INET, socket.SOCK_STREAM)
        with other:
            other.bind(('127.0.0.1', 0))
            self.assertNotIn(other.getsockname()[1], block)

    def test_e2e_jobs_print_every_case_in_order_with_its_verdict(self):
        import contextlib
        import io
        import e2e_swe32
        with tempfile.TemporaryDirectory(prefix='swe32-fake-') as tmp:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                failed = e2e_swe32.run_parallel(['nope2', 'nope1', 'nope3'], Path(tmp), 2, Path(tmp) / 'logs')
            lines = [line for line in stream.getvalue().splitlines() if line.startswith(('E2E ', '  PROBLEM'))]
            self.assertEqual(failed, ['nope2', 'nope1', 'nope3'])
            self.assertEqual([line.split()[1] for line in lines if line.startswith('E2E ')], ['nope2', 'nope1', 'nope3'])
            self.assertTrue(all(line.split()[2] == 'FAIL' for line in lines if line.startswith('E2E ')))
            self.assertEqual(sorted(p.name for p in (Path(tmp) / 'logs').iterdir()),
                             ['nope1.log', 'nope2.log', 'nope3.log'])
        self.assertEqual(e2e_swe32.AFTER, {'fetch': ('pass',), 'pack': ('pass', 'goalb4', 'obs', 'rfill', 'stock')})
        self.assertEqual(sorted(e2e_swe32.EXPECTED_S), sorted(list(e2e_swe32.JOB) + ['fetch', 'pack']))


class InteractiveThinking1111(unittest.TestCase):
    """2026-11-11 (the user: the thinking-on runs urgent for the group meeting, on ghx4-interactive one at a time):
    goalti / akvsi / rtoolsti = goalt / akvt / rtoolst's servers, settings and thinking with the agent cap and minimum
    refitted to 02:00:00 from the measured thinking-on jobs; the queue gate lets an interactive plan start beside held
    tbench jobs and those pending only on them (gctx25 3560498 held, smokeqt 3568310 waiting on it)."""
    VARIANTS = dict(goalti=('goalt', '02:00:00', 5480, 5460, 3750),
                    akvsi=('akvt', '01:52:00', 4490, 4980, 3150),
                    rtoolsti=('rtoolst', '01:22:00', 3300, 3180, 2100))

    def test_the_variants(self):
        plans = c.cfg(PINS)['plans']
        for name, (base, time, expected, cap, minimum) in self.VARIANTS.items():
            plan, batch = plans[name], plans[base]
            self.assertEqual((plan['partition'], plan['role'], plan['variant_of'], plan['time'], plan['expected_seconds'],
                              plan['eval_wait_seconds']), ('ghx4-interactive', 'interactive', base, time, expected, 600))
            self.assertEqual((plan['thinking'], plan['server']), (True, batch['server']))
            self.assertNotIn(c.ASYNC_KEY, plan)
            (mine,), (theirs,) = plan['servers'], batch['servers']
            self.assertEqual((mine['agent_cap_seconds'], mine['agent_min_seconds']), (cap, minimum))
            self.assertEqual({k: v for k, v in mine.items() if k not in c.VARIANT_REFIT_KEYS},
                             {k: v for k, v in theirs.items() if k not in c.VARIANT_REFIT_KEYS})
            self.assertLess(cap, theirs['agent_cap_seconds'])
            self.assertGreater(c.time_seconds(batch['time']), 2 * 3600)  # why the refit is allowed
            (s,), (b,) = c.plan_servers(PINS, name), c.plan_servers(PINS, base)
            self.assertEqual(c.launch_arguments(PINS, s, REWRITER), c.launch_arguments(PINS, b, REWRITER), name)
            self.assertEqual(c.model_override_text(PINS, s, 23004), c.model_override_text(PINS, b, 23004))
            self.assertEqual((c.plan_running(PINS, name), c.plan_tasks(PINS, name)), (16, 32))
            fit, cost, memory = c.validate_plan(PINS, name), c.plan_cost(PINS, name), c.memory_fit(PINS, name)
            # 1,740 s of reserves (60 + 480 + 300 + 420 + 180 + 300) + the cap = the limit; the minimum after the wait
            self.assertEqual((fit['worst_case_seconds'], fit['with_eval_wait_seconds'], fit['time_seconds']),
                             (1740 + cap, 1920 + minimum, c.time_seconds(time)), name)
            self.assertTrue(fit['fits'] and fit['time_seconds'] <= 7200 and minimum <= cap)
            self.assertLessEqual(cost['limit_over_expected'], 1.5)
            self.assertEqual((cost['billing_factor'], cost['gpu_hours_limit'], cost['gpu_hours_expected']),
                             (2, round(2 * c.time_seconds(time) / 3600, 3), round(2 * expected / 3600, 3)))
            self.assertEqual((memory['need_gib'], memory['headroom_gib'], memory['fits']), (105.29, 5.61, True))
            phrase = plan['authorization']
            self.assertTrue(phrase.startswith('AUTHORIZE_SWE32_%s_32TASKS_16SESSIONS_' % name.upper()), phrase)
            self.assertTrue(phrase.endswith('_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_%dMIN_NO_RETRY'
                                            % (c.time_seconds(time) // 60)), phrase)
            self.assertNotIn('REWRITER', phrase)  # (no argv of these runs names it)
            self.assertNotIn('_BATCH_', phrase)
            self.assertIsNone(c.plan_blocked(PINS, name))
        # akvsi is the full-scale AKV plan (32 tasks, 16 sessions: the user's decision), not akvs (8 tasks, 4 sessions)
        self.assertEqual(c.plan_servers(PINS, 'akvsi')[0]['label'], 'akv262t')
        self.assertEqual((c.plan_running(PINS, 'akvs'), c.plan_tasks(PINS, 'akvs')), (4, 8))
        self.assertIn('CAPACITY_WAIT', plans['akvsi']['about'])
        self.assertEqual(c.plan_servers(PINS, 'goalti')[0]['goal_rewriter_engine'], 'vllm')

    def test_the_variant_rule_refuses(self):
        def refused(change, name, pattern='variant_of'):
            pins = copy.deepcopy(PINS)
            change(pins['config']['plans'])
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, name)
        refused(lambda p: p['goalti']['servers'][0].update(agent_cap_seconds=7300), 'goalti')  # above the batch cap
        refused(lambda p: p['goalti']['servers'][0].update(agent_min_seconds=4600), 'goalti')  # above the batch min
        refused(lambda p: p['goalti']['servers'][0].update(step_limit=100), 'goalti')
        refused(lambda p: p['goalti']['servers'][0].update(goal_overlap=None), 'goalti')
        refused(lambda p: p['goalti']['servers'][0].update(label='goalvllmti'), 'goalti')
        refused(lambda p: p['rtoolsti']['servers'][0].update(tasks=16), 'rtoolsti')
        refused(lambda p: p['akvsi'].update(thinking=False), 'akvsi')
        refused(lambda p: p['akvsi'].update(variant_of='akvs'), 'akvsi')  # akvs: 8 tasks, 4 sessions, arena 60
        refused(lambda p: p['akvsi']['server'].update(dram_arena_gib=60), 'akvsi')
        refused(lambda p: p['rtoolsti'].update(async_scheduling=True), 'rtoolsti')
        refused(lambda p: p['rtoolsti'].pop('server'), 'rtoolsti')
        # a batch plan that fits the partition as it is: its variant stays identical (obs 01:07:00, goal 01:42:00)
        refused(lambda p: p['obsi']['servers'][0].update(agent_cap_seconds=2000), 'obsi')
        refused(lambda p: p['goali']['servers'][0].update(agent_min_seconds=3000), 'goali')
        # the refit still answers to the time model and the 1.5 x rule
        refused(lambda p: p['rtoolsti'].update(expected_seconds=3000), 'rtoolsti', 'within 1.50 x')
        refused(lambda p: p['rtoolsti']['servers'][0].update(agent_cap_seconds=3300), 'rtoolsti', 'does not fit')
        refused(lambda p: p['goalti'].update(time='02:01:00', authorization=p['goalti']['authorization'].replace(
            '_120MIN_', '_121MIN_')), 'goalti', 'maximum')
        refused(lambda p: p['goalti'].update(authorization=p['goalti']['authorization'].replace(
            '_INTERACTIVE_', '_BATCH_')), 'goalti', 'phrase')

    def test_queue_parked(self):
        held = '3560498|PENDING|ghx4|gswe32-gctx25|(null)|JobHeldUser'
        waits = '3568310|PENDING|ghx4|tb16-smokeqt|afterany:3560498(unfulfilled)|Dependency'
        self.assertEqual(c.queue_parked([held, waits]), ['3560498', '3568310'])
        self.assertEqual(c.queue_parked([held.replace('JobHeldUser', 'JobHeldAdmin'), waits]), ['3560498', '3568310'])
        self.assertEqual(c.queue_parked([held, waits, '3568311|PENDING|ghx4|tb16-x|afterok:3568310(unfulfilled)|'
                                         'Dependency']), ['3560498', '3568310', '3568311'])  # recursively
        self.assertEqual(c.queue_parked([held.replace('JobHeldUser', 'Priority'), waits]), [])
        self.assertEqual(c.queue_parked([waits]), [])  # its dependency is not in the listing
        self.assertEqual(c.queue_parked(['3560498|PENDING|ghx4|gswe32-gctx25|(null)']), [])  # no reason field
        self.assertEqual(c.queue_parked(['777|RUNNING|ghx4|swe32-x|(null)|None',
                                         '666|PENDING|ghx4|tb16-x|afterany:777(unfulfilled)|Dependency']), [])
        both = '666|PENDING|ghx4|tb16-x|afterany:3560498(unfulfilled)%safterany:777(unfulfilled)|Dependency'
        for sep in (',', '?'):  # every entry must be on a parked job
            self.assertEqual(c.queue_parked([held, '777|RUNNING|ghx4|swe32-x|(null)|None', both % sep]), ['3560498'])
        self.assertEqual(c.queue_parked([held, '666|PENDING|ghx4|tb16-x|singleton|Dependency']), ['3560498'])
        self.assertEqual(c.queue_parked([held, '666|PENDING|ghx4|tb16-x|afterany:3560498(failed)|'
                                         'DependencyNeverSatisfied']), ['3560498'])
        self.assertEqual(c.queue_parked(['1|PENDING|ghx4|tb16-a|afterany:2(unfulfilled)|Dependency',
                                         '2|PENDING|ghx4|tb16-b|afterany:1(unfulfilled)|Dependency']), [])  # a cycle

    def test_queue_gate_interactive_beside_held_tbench_jobs(self):
        held = '3560498|PENDING|ghx4|gswe32-gctx25|(null)|JobHeldUser\n'
        waits = '3568310|PENDING|ghx4|tb16-smokeqt|afterany:3560498(unfulfilled)|Dependency\n'
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            def gate(out, plan='goalti', after=None):
                calls.append(plan)  # (queue_gate writes its record once per name)
                return c.queue_gate(Path(tmp), PINS, c.cfg(PINS)['plans'][plan], dict(plan=plan, after_job=after),
                                    'q%d.json' % len(calls), runner=lambda *a, **k: FakeRun(out), user='ycao11')
            for plan in self.VARIANTS:  # the user's queue of 11-11: gctx25 held, smokeqt waiting on it
                value = gate(held + waits + '888|PENDING|ghx4|other-job|(null)|Priority\n', plan)
                self.assertEqual((value['status'], value['blocking']), ('QUEUE_GATE_PASS', []), plan)
                self.assertEqual(value['parked'], [held.strip(), waits.strip()])
                self.assertEqual(value['partition'], 'ghx4-interactive')
            for out in (held.replace('JobHeldUser', 'Priority') + waits,  # released: it may start
                        held + waits.replace('3560498', '777') + '777|RUNNING|ghx4|swe32-x|(null)|None\n',
                        waits,  # waiting on a job not in the listing
                        held + waits + '555|PENDING|ghx4|swe32-goalt|(null)|Priority\n',  # another one queued
                        held + waits + '556|RUNNING|ghx4|swe32-obst|(null)|None\n',
                        held + waits + '557|PENDING|ghx4|tb16-x|afterany:556(unfulfilled)|Dependency\n'
                        '556|RUNNING|ghx4|other|(null)|None\n',
                        held.replace('|JobHeldUser', '') + waits):  # (no reason field: as before 11-11)
                with self.assertRaisesRegex(RuntimeError, 'queue holds', msg=out):
                    gate(out)
            with self.assertRaisesRegex(RuntimeError, 'at most 1'):  # a held job on ghx4-interactive still counts
                gate('3560498|PENDING|ghx4-interactive|gswe32-gctx25|(null)|JobHeldUser\n')
            for plan in ('goal', 'goalt'):  # a batch plan keeps the rule: name the chain with -AfterJob
                with self.assertRaisesRegex(RuntimeError, 'queue holds'):
                    gate(held + waits, plan)
                value = gate(held + waits, plan, '3568310')
                self.assertEqual((value['status'], value['chain'], value['parked']),
                                 ('QUEUE_GATE_PASS', ['3560498', '3568310'], []))



class FullThinkingAndPush1111(unittest.TestCase):
    """2026-11-11: fullt / fullti (policy full with thinking ON at obst's settings: the control that separates thinking
    from Page selection) and the evidence push to tbench at the job's end (config.tbench_verify)."""

    def test_the_full_thinking_plans(self):
        full, obst = c.plan_servers(PINS, 'full')[0], c.plan_servers(PINS, 'obst')[0]
        for name, partition in (('fullt', 'ghx4'), ('fullti', 'ghx4-interactive')):
            plan, (server,) = c.cfg(PINS)['plans'][name], c.plan_servers(PINS, name)
            self.assertEqual((server['label'], server['policy'], server['tasks'], server['step_limit'],
                              server['agent_cap_seconds'], server['agent_min_seconds'], server[c.THINKING_KEY]),
                             ('full262t', 'full', 32, 120, 5460, 3700, True), name)
            self.assertEqual((plan['partition'], plan['time'], plan['expected_seconds'], plan['eval_wait_seconds'],
                              plan['server'], c.ASYNC_KEY in plan), (partition, '02:00:00', 5140, 600,
                                                                     c.cfg(PINS)['plans']['obst']['server'], False))
            self.assertEqual(c.plan_settings(PINS, name), c.plan_settings(PINS, 'obst'))
            # plan full's server (no Goal machinery, no active budget) with obst's settings and thinking
            mine, theirs = dict(server), dict(full)
            for k in ('label', 'agent_cap_seconds', 'agent_min_seconds', c.THINKING_KEY, 'step_limit', 'settings'):
                mine.pop(k), theirs.pop(k, None)
            self.assertEqual(mine, theirs, name)
            self.assertEqual(server['settings'], obst['settings'])
            self.assertEqual((server['admission'], server['rewriter']), ([], False))
            args = c.launch_arguments(PINS, server, REWRITER)
            self.assertEqual(args.count('--enable-thinking'), 1)
            self.assertNotIn('--active-token-budget', args)
            self.assertNotIn('--goal-rewriter-model', args)
            self.assertNotIn('--async-scheduling', args)
            for flag in ('--kv-cache-gib', '--max-model-len', '--dram-arena-gib'):  # obst's values
                self.assertEqual(args[args.index(flag) + 1], c.launch_arguments(PINS, obst, REWRITER)[
                    c.launch_arguments(PINS, obst, REWRITER).index(flag) + 1], flag)
            fit = c.plan_fit(PINS, name)
            self.assertEqual(fit['time_seconds'], 7200)
            self.assertLessEqual(c.plan_cost(PINS, name)['limit_over_expected'], c.MAX_LIMIT_OVER_EXPECTED)
            self.assertEqual(c.memory_fit(PINS, name)['need_gib'], 105.29)
            self.assertIn('ESTIMATE', plan['about'])
        self.assertEqual(c.cfg(PINS)['plans']['fullti']['variant_of'], 'fullt')
        self.assertEqual(ORDER1112[-2 - len(NEW64 + NEW1112):-len(NEW64 + NEW1112)], FULLT)

    def test_push_scripts(self):
        settings = c.cfg(PINS)['tbench_verify']
        self.assertEqual((settings['push'], settings['root'], settings['total_seconds']),
                         (True, '/home/ubuntu/swe32-verify', 120))
        script = tbench_link.push_script(settings['root'], 'swe32-1-r1', 'swe32-core.tar.xz', 'a' * 64)
        self.assertTrue(script.startswith(': swe32_evidence_push;'))
        self.assertIn('D=/home/ubuntu/swe32-verify/evidence/swe32-1-r1;', script)
        self.assertIn('mv -n "$P" "$D/$F"', script)
        for bad in (('/home/ubuntu/x;rm', 'swe32-1-r1', 'swe32-core.tar.xz', 'a' * 64),
                    ('/home/ubuntu/../x', 'swe32-1-r1', 'swe32-core.tar.xz', 'a' * 64),
                    (settings['root'], 'swe32-1-r1', 'other.tar.xz', 'a' * 64),
                    (settings['root'], 'swe32-1-r1', 'swe32-core.tar.xz', 'A' * 64)):
            with self.assertRaises(RuntimeError):
                tbench_link.push_script(*bad)
        archives = dict(core=dict(sha256='b' * 64, bytes=10), bulk=dict(sha256='c' * 64, bytes=20))
        end = dict(job_status='SWE32_PASS', slurm_state='COMPLETED', slurm_exit_code='0:0')
        text = tbench_link.trigger_script(settings, PINS['tbench'], 'swe32-1-r1', '1', '/u/ycao11/swe32-' + '0' * 32,
                                          archives, end, 'd' * 64, 'core=PUSHED,bulk=PUSHED')
        self.assertIn('exec /usr/bin/python3 -I -B "$R/code/$T/tbench_verify/remote_verify.py" trigger --run-id '
                      'swe32-1-r1 --job-id 1 ', text)
        self.assertIn('--tbench-bin /home/ubuntu/swe32/bin --tbench-runs /home/ubuntu/swe32/runs', text)
        with self.assertRaises(RuntimeError):
            tbench_link.trigger_script(settings, PINS['tbench'], 'swe32-1-r1', '1', '/u/ycao11/x; rm', archives, end,
                                       'd' * 64, 'core=PUSHED')
        # the members id is the same on every side
        sys.path.insert(0, str(ROOT / 'postrun' / 'tbench_verify'))
        import remote_verify
        import verify_on_tbench
        members = verify_on_tbench.payload_members(ROOT)
        self.assertEqual(tbench_link.members_id(members), verify_on_tbench.members_id(members))
        self.assertEqual(tbench_link.members_id(members), remote_verify.members_id(members))

    def test_push_is_bounded_and_never_raises(self):
        settings = dict(c.cfg(PINS)['tbench_verify'])
        budget = job.Budget(0.0, PINS, 'fullt', clock=lambda: 0.0)
        archives = dict(core=dict(sha256='b' * 64, bytes=10), bulk=dict(sha256='c' * 64, bytes=20))
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            c.save(stage / 'delivery.json', dict(files={'job.py': dict(sha256='e' * 64)}, options={}))
            c.save(stage / 'submission.json', dict(stage='/u/ycao11/swe32-' + '0' * 32))
            pins = copy.deepcopy(PINS)
            pins['config']['tbench_verify']['push'] = False
            self.assertEqual(job.push_evidence(stage, pins, archives, 'SWE32_PASS', 'swe32-1-r1', object(), '1', budget),
                             dict(status='PUSH_DISABLED'))
            # less than the push's minimum before the Slurm limit: skipped, not tried
            late = job.Budget(0.0, PINS, 'fullt', clock=lambda: 7200.0 - 150)
            value = job.push_evidence(stage, PINS, archives, 'SWE32_PASS', 'swe32-1-r1', object(), '1', late)
            self.assertEqual(value['status'], 'PUSH_FAILED')
            self.assertIn('SKIPPED_TIME', value['reason'])

            class Link:  # the link raises (or the job's SIGTERM arrives): recorded, never raised
                clock = staticmethod(lambda: 0.0)

                def push_evidence(self, *args):
                    raise job.JobTerminated('SIGTERM')
            value = job.push_evidence(stage, PINS, archives, 'SWE32_PARTIAL', 'swe32-1-r1', Link(), '1', budget)
            self.assertEqual(value['status'], 'PUSH_FAILED')
            self.assertIn('JobTerminated', value['reason'])
            seen = {}

            class Recording(Link):
                def push_evidence(self, settings_, run, job_id, stage_dir, stage_, archives_, end, members, deadline):
                    seen.update(end=end, members=members, deadline=deadline, stage=stage_)
                    return dict(status='PUSH_OK')
            value = job.push_evidence(stage, PINS, archives, 'SWE32_PARTIAL', 'swe32-1-r1', Recording(), '1', budget)
            self.assertEqual((value['status'], value['allowed_s']), ('PUSH_OK', 120))
            self.assertEqual((seen['end']['slurm_state'], seen['end']['slurm_exit_code'], seen['deadline']),
                             ('FAILED', '3:0', 120))
            self.assertEqual(seen['members'], tbench_link.members_id({'job.py': 'e' * 64}))
            self.assertEqual(seen['stage'], '/u/ycao11/swe32-' + '0' * 32)



class Proxy64IdleRelease1112(unittest.TestCase):
    """2026-11-12: the 64-instance proxy (the task count as a plan key, the sealed 64-task list), the idle-release plan
    key (--release-idle-pages) with its server checks, CAPACITY_STALL, the client-timeout wording and the Goal-wait
    metric of the join."""

    def test_the_sealed_64_task_list(self):
        sets = PINS[c.TASK_SETS_KEY]
        self.assertEqual(sorted(sets), ['150', '64', 'h2'])  # (2026-11-16: the second half; 11-17: the 150 list)
        value = sets['64']
        self.assertEqual(len(value['instances']), 64)
        self.assertEqual(value['instances'][:32], PINS['tasks']['instances'])
        self.assertEqual(len(set(value['instances'])), 64)
        self.assertEqual((value['instances'][32], value['instances'][-1]), ('django__django-11119', 'django__django-11848'))
        self.assertEqual(value['pool']['file'], 'tasks-pool-64.json')
        raw = (ROOT / 'tasks' / 'tasks-pool-64.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), value['pool']['sha256'])
        tasks = json.loads(raw)
        self.assertEqual(c.tasks_bytes(tasks[:32]), (ROOT / 'tasks' / 'tasks-pool.json').read_bytes())
        self.assertEqual(value['provisional'], False)
        self.assertIn('CONFIRMED', value['confirmed'])
        # the tbench half's manifest still names the 32 list only (no redeploy)
        self.assertEqual(json.loads((ROOT / 'tasks' / 'swe32-tasks.json').read_bytes())['instances'],
                         PINS['tasks']['instances'])
        payload = json.loads((ROOT / 'dist' / 'payload.json').read_bytes())
        self.assertEqual(payload['files']['tasks-pool-64.json']['sha256'], value['pool']['sha256'])

    def test_the_new_plans(self):
        plans = c.cfg(PINS)['plans']
        self.assertEqual(ORDER1112[-len(NEW64 + NEW1112):-len(NEW1112)], NEW64)
        goalti, obst, fullt = (c.plan_servers(PINS, n)[0] for n in ('goalti', 'obst', 'fullt'))
        wanted = dict(obs64=('obs64t', obst, 64, 'ghx4', '02:13:00', 5343, 5640, 3450, 1200, False),
                      obs64i=('obs64t', obst, 64, 'ghx4-interactive', '02:00:00', 5343, 4860, 3450, 1200, False),
                      goal64=('goalvllm64t', goalti, 64, 'ghx4', '02:34:00', 7458, 6900, 4900, 1200, False),
                      # (re-fitted 2026-11-12 from the measured fulltri 3583081: span 1,694 s, elapsed 2,421 s)
                      full64=('full64tr', fullt, 64, 'ghx4', '01:53:00', 4555, 4440, 2890, 1200, True),
                      fulltr=('full262tr', fullt, 32, 'ghx4', '01:14:00', 2978, 2700, 1440, 600, True),
                      fulltri=('full262tr', fullt, 32, 'ghx4-interactive', '01:14:00', 2978, 2700, 1440, 600, True))
        for name, (label, base, tasks, partition, time, expected, cap, minimum, wait, release) in wanted.items():
            plan, (server,) = plans[name], c.plan_servers(PINS, name)
            self.assertEqual((server['label'], server['tasks'], plan['partition'], plan['time'], plan['expected_seconds'],
                              server['agent_cap_seconds'], server['agent_min_seconds'], plan['eval_wait_seconds'],
                              c.release_idle_on(PINS, server), server[c.THINKING_KEY]),
                             (label, tasks, partition, time, expected, cap, minimum, wait, release, True), name)
            self.assertEqual(c.plan_task_count(PINS, name), 64 if tasks == 64 else None, name)
            self.assertEqual(len(c.instance_ids(PINS, name)), tasks, name)
            # the base server of the 32-task thinking-on runs, but for the label, the cap / minimum, the task count and
            # (full) the idle release in the plan's settings
            mine, theirs = dict(server), dict(base)
            for k in ('label', 'agent_cap_seconds', 'agent_min_seconds', 'tasks', 'settings'):
                mine.pop(k), theirs.pop(k)
            self.assertEqual(mine, theirs, name)
            self.assertEqual(server['settings'], dict(base['settings'], **({c.RELEASE_KEY: True} if release else {})))
            fit = c.validate_plan(PINS, name)
            spare = 600 if tasks == 64 else 0  # 64 sessions' stop, checks and archives: an extra 600 s above the reserves
            self.assertEqual(fit['time_seconds'] - fit['worst_case_seconds'], spare, name)
            self.assertLessEqual(c.plan_cost(PINS, name)['limit_over_expected'], c.MAX_LIMIT_OVER_EXPECTED)
            self.assertEqual(c.memory_fit(PINS, name)['need_gib'], 105.29)
            self.assertIn('ESTIMATE', plan['about'], name)
            args = c.launch_arguments(PINS, server, REWRITER)
            self.assertEqual(args.count('--release-idle-pages'), 1 if release else 0, name)
            self.assertEqual(args.count('--enable-thinking'), 1, name)
            base_args = c.launch_arguments(PINS, base, REWRITER)
            self.assertEqual([a for a in args if a != '--release-idle-pages'], base_args, name)
            expect = c.server_expectation(PINS, server, None, 'm', ['t'])
            self.assertEqual(expect.get(c.RELEASE_KEY), True if release else None, name)
        self.assertEqual((plans['obs64i']['variant_of'], plans['fulltri']['variant_of']), ('obs64', 'fulltr'))
        self.assertNotIn('goal64i', plans)  # 02:00:00 leaves a cap of 4,860 s, below goal64's W
        self.assertNotIn('full64i', plans)
        costs = {n: (c.plan_cost(PINS, n)['gpu_hours_limit'], c.plan_cost(PINS, n)['gpu_hours_expected']) for n in NEW64}
        self.assertEqual(costs, dict(obs64=(2.217, 1.484), obs64i=(4.0, 2.968), goal64=(2.567, 2.072),
                                     full64=(1.883, 1.265), fulltr=(1.233, 0.827), fulltri=(2.467, 1.654)))
        for name in ('fulltr', 'fulltri', 'full64'):  # the measurement they were re-fitted from
            self.assertIn('fulltri 3583081', plans[name]['about'], name)

    def test_every_earlier_plan_is_unchanged(self):
        for name in PLANS[:PLANS.index('fullti') + 1]:
            self.assertIsNone(c.plan_task_count(PINS, name), name)
            self.assertEqual(c.instance_ids(PINS, name), PINS['tasks']['instances'], name)
            self.assertEqual(c.pool_file(PINS, name), c.TASK_POOL, name)
            for server in c.plan_servers(PINS, name):
                self.assertFalse(c.release_idle_on(PINS, server), name)
                self.assertNotIn(c.RELEASE_KEY, c.server_expectation(PINS, server, None, 'm', ['t']), name)
                if server['kind'] == 'runtime':
                    self.assertNotIn('--release-idle-pages', c.launch_arguments(PINS, server, REWRITER), name)
                if name != 'akvs':
                    self.assertEqual(server['tasks'], 32, name)
        self.assertNotIn(c.RELEASE_KEY, c.cfg(PINS)['server'])  # idle release stays off in config.server

    def test_the_refusals(self):
        def refused(change, plan, text):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, text):
                c.validate_plan(pins, plan)
        refused(lambda p: p['config']['plans']['obs64'].update(tasks=48), 'obs64', 'names 48 tasks')
        refused(lambda p: p.pop(c.TASK_SETS_KEY), 'obs64', 'no sealed list of that size')
        refused(lambda p: p['runtime_features']['session_pool'].update(max_tasks=32), 'obs64',
                'more than the sealed session pool')
        refused(lambda p: p['config']['plans']['obs64i'].pop('tasks'), 'obs64i', 'variant_of')
        refused(lambda p: p['config']['plans']['fulltr']['server'].update(release_idle_pages=False), 'fulltr',
                'release_idle_pages true')
        refused(lambda p: p['runtime_features'].update({c.RELEASE_KEY: None}), 'fulltr', '--release-idle-pages')
        refused(lambda p: p['config']['plans']['fulltri']['server'].pop('release_idle_pages'), 'fulltri', 'variant_of')

    def test_the_load_pool_of_a_64_task_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            for name in ('tasks-pool.json', 'tasks-pool-64.json'):
                (stage / name).write_bytes((ROOT / 'tasks' / name).read_bytes())
            self.assertEqual(len(c.load_pool(stage, PINS, 'obs64')), 64)
            self.assertEqual(len(c.load_pool(stage, PINS, 'obst')), 32)
            self.assertEqual(len(c.load_pool(stage, PINS)), 32)
            (stage / 'tasks-pool-64.json').write_bytes((ROOT / 'tasks' / 'tasks-pool.json').read_bytes())
            with self.assertRaisesRegex(RuntimeError, 'tasks-pool-64.json differs from its pin'):
                c.load_pool(stage, PINS, 'obs64')

    def idle_evidence(self, out, release=True, events=None, totals=None, records=None):
        """A minimal server output: startup / READY / worker records, two rows (s1's readmission after s0 evicted it)
        and their natives, capacity.jsonl and engine-cleanup.json."""
        import smoke_checks as sc
        ids = ['sid-0', 'sid-1']
        flag = {c.RELEASE_KEY: True} if release else {}
        c.save(out / 'startup.json', dict(flag))
        c.save(out / 'ready.json', dict(flag, sessions=[dict(session_id=i) for i in ids]))
        c.save(out / 'worker-attached.json', dict(receipt=dict(flag)))
        records = records if records is not None else [
            dict(evicted_sessions=['sid-1'], evicted_pages=3, readmitted_pages=[]),
            dict(evicted_sessions=[], evicted_pages=0, readmitted_pages=[0, 1, 2])]
        rows = []
        for k, record in enumerate(records):
            row = dict(request_id='r%d' % k, session_index=k, session_id=ids[k], turn=1)
            if release:
                row['idle_release'] = record
            rows.append(row)
        (out / 'runtime.jsonl').write_bytes(b''.join(sc.canonical(r) + b'\n' for r in rows))
        events = events if events is not None else [
            dict(event='evict', session_id='sid-1', session_index=1, pages=3, blocks=3, request_id='r0',
                 free_blocks_by_pool_before=[100, 9], free_blocks_by_pool=[103, 9], monotonic_ns=1),
            dict(event='readmit', session_id='sid-1', session_index=1, pages=3, blocks=3, request_id='r1',
                 free_blocks_by_pool_before=[103, 9], free_blocks_by_pool=[100, 9], monotonic_ns=2)]
        if release:
            (out / 'capacity.jsonl').write_bytes(b''.join(
                json.dumps(dict(e, schema='remote-ra-capacity-event-v1')).encode() + b'\n' for e in events))
        totals = totals if totals is not None else dict(evictions=1, evicted_pages=3, readmissions=1, readmitted_pages=3)
        c.save(out / 'engine-cleanup.json', dict(idle_release=totals) if release else {})

    def test_the_release_idle_pages_check(self):
        expect = {c.RELEASE_KEY: True}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.idle_evidence(out)
            got = swe_checks.check_release_idle_pages(out, expect)
            self.assertEqual((got['evictions'], got['readmissions'], got['readmitted_pages'],
                              got['evicting_requests'], got['readmitting_requests']), (1, 1, 3, 1, 1))
            with self.assertRaisesRegex(Exception, 'records it'):  # the evidence of idle release on a plan without it
                swe_checks.check_release_idle_pages(out, {})
        cases = dict(
            totals=dict(totals=dict(evictions=1, evicted_pages=3, readmissions=0, readmitted_pages=0)),
            readmit_other_pages=dict(events=[
                dict(event='evict', session_id='sid-1', session_index=1, pages=3, blocks=3, request_id='r0',
                     free_blocks_by_pool_before=[100, 9], free_blocks_by_pool=[103, 9], monotonic_ns=1),
                dict(event='readmit', session_id='sid-1', session_index=1, pages=2, blocks=2, request_id='r1',
                     free_blocks_by_pool_before=[103, 9], free_blocks_by_pool=[101, 9], monotonic_ns=2)],
                totals=dict(evictions=1, evicted_pages=3, readmissions=1, readmitted_pages=2)),
            mamba_moved=dict(events=[
                dict(event='evict', session_id='sid-1', session_index=1, pages=3, blocks=3, request_id='r0',
                     free_blocks_by_pool_before=[100, 9], free_blocks_by_pool=[103, 8], monotonic_ns=1),
                dict(event='readmit', session_id='sid-1', session_index=1, pages=3, blocks=3, request_id='r1',
                     free_blocks_by_pool_before=[103, 8], free_blocks_by_pool=[100, 8], monotonic_ns=2)]),
            record_differs=dict(records=[dict(evicted_sessions=[], evicted_pages=0, readmitted_pages=[]),
                                         dict(evicted_sessions=[], evicted_pages=0, readmitted_pages=[0, 1, 2])]))
        for name, change in cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                self.idle_evidence(out, **change)
                with self.assertRaises(swe_checks.CheckFailed, msg=name):
                    swe_checks.check_release_idle_pages(out, expect)
        with tempfile.TemporaryDirectory() as tmp:  # off: no records, no file, no totals
            out = Path(tmp)
            self.idle_evidence(out, release=False)
            self.assertEqual(swe_checks.check_release_idle_pages(out, {}), dict(release_idle_pages=None))
            with self.assertRaisesRegex(swe_checks.CheckFailed, 'expected true everywhere'):
                swe_checks.check_release_idle_pages(out, expect)

    def rebuild_evidence(self, out, admitted=('rA', 'rR', 'rB', 'rC'), indexes=None, rebuild=True, second=4,
                         sealed_pages=4):
        """sess24ri r2 (job 3616345, s13) in small: s1's request rA evicts s0 (3 Pages); s0's rebuild rR (a cold
        re-prefill of `sealed_pages` Pages) is admitted; s2's request rB, which arrived before rR (request_index 1 <
        2) but was admitted after it (capacity waits), evicts s0 again (`second` Pages); s0's next turn rC readmits
        them. `admitted`: engine-cleanup admitted_request_ids (the admission order); None leaves the key out."""
        import smoke_checks as sc
        ids, P = ['sid-0', 'sid-1', 'sid-2'], sc.LEGACY_LAYOUT['page_tokens']
        flag = {c.RELEASE_KEY: True}
        c.save(out / 'startup.json', dict(flag))
        c.save(out / 'ready.json', dict(flag, sessions=[dict(session_id=i) for i in ids]))
        c.save(out / 'worker-attached.json', dict(receipt=dict(flag)))
        indexes = indexes or dict(rA=0, rB=1, rR=2, rC=3)
        none = dict(evicted_sessions=[], evicted_pages=0, readmitted_pages=[])
        rows = [dict(request_id='rA', session_index=1, turn=3, idle_release=dict(none, evicted_sessions=['sid-0'],
                                                                                    evicted_pages=3)),
                dict(request_id='rR', session_index=0, turn=4, idle_release=dict(none),
                     stage_b=dict(rebuild=rebuild, sealed_start=0, sealed_end=sealed_pages * P + 5)),
                dict(request_id='rB', session_index=2, turn=3, idle_release=dict(none, evicted_sessions=['sid-0'],
                                                                                    evicted_pages=second)),
                dict(request_id='rC', session_index=0, turn=5, idle_release=dict(none, readmitted_pages=list(
                    range(second))))]
        for row in rows:
            row.update(session_id=ids[row['session_index']], request_index=indexes[row['request_id']])
        (out / 'runtime.jsonl').write_bytes(b''.join(sc.canonical(r) + b'\n' for r in rows))
        events = [dict(event='evict', session_id='sid-0', session_index=0, pages=3, blocks=3, request_id='rA',
                       free_blocks_by_pool_before=[100, 9], free_blocks_by_pool=[103, 9], monotonic_ns=1),
                  dict(event='evict', session_id='sid-0', session_index=0, pages=second, blocks=second,
                       request_id='rB', free_blocks_by_pool_before=[90, 9], free_blocks_by_pool=[90 + second, 9],
                       monotonic_ns=2),
                  dict(event='readmit', session_id='sid-0', session_index=0, pages=second, blocks=second,
                       request_id='rC', free_blocks_by_pool_before=[95, 9], free_blocks_by_pool=[95 - second, 9],
                       monotonic_ns=3)]
        (out / 'capacity.jsonl').write_bytes(b''.join(
            json.dumps(dict(e, schema='remote-ra-capacity-event-v1')).encode() + b'\n' for e in events))
        cleanup = dict(idle_release=dict(evictions=2, evicted_pages=3 + second, readmissions=1,
                                         readmitted_pages=second))
        if admitted is not None:
            cleanup['admitted_request_ids'] = list(admitted)
        c.save(out / 'engine-cleanup.json', cleanup)

    def test_a_second_eviction_after_a_rebuild_in_admission_order(self):
        """2026-11-17 (sess24ri r2 = job 3616345 failed only this): a second eviction without a readmission is legal
        exactly when a rebuild of the session was ADMITTED between the two evicting admissions (engine-cleanup
        admitted_request_ids), not when its request_index (the arrival order) lies between them; the second eviction
        releases the rebuild's sealed Pages."""
        expect = {c.RELEASE_KEY: True}
        with tempfile.TemporaryDirectory() as tmp:  # the reordered pattern of job 3616345: PASS
            out = Path(tmp)
            self.rebuild_evidence(out)
            got = swe_checks.check_release_idle_pages(out, expect)
            self.assertEqual((got['evictions'], got['readmissions'], got['dropped_by_rebuild'], got['pending_at_end']),
                             (2, 1, 1, 0))
        cases = dict(
            # request_index puts the rebuild between the two evictions (the old rule passed it), but it was admitted
            # after the second one: no rebuild admitted in between
            rebuild_admitted_after=(dict(admitted=('rA', 'rB', 'rR', 'rC'), indexes=dict(rA=0, rR=1, rB=2, rC=3)),
                                    'no rebuild of it admitted in between'),
            rebuild_admitted_before=(dict(admitted=('rR', 'rA', 'rB', 'rC')), 'no rebuild of it admitted in between'),
            not_a_rebuild=(dict(rebuild=False), 'no rebuild of it admitted in between'),
            no_admission_order=(dict(admitted=None), 'no rebuild of it admitted in between'),
            other_pages=(dict(second=5), 'evicts 5 Pages after its rebuild sealed'))
        for name, (change, message) in cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                self.rebuild_evidence(out, **change)
                with self.assertRaisesRegex(swe_checks.CheckFailed, message, msg=name):
                    swe_checks.check_release_idle_pages(out, expect)

    def test_the_per_request_record_and_the_readmit_frame(self):
        import smoke_checks as sc
        record = dict(evicted_sessions=[], evicted_pages=0, readmitted_pages=[4, 5])
        native, row = dict(idle_release=record), dict(idle_release=dict(record))
        self.assertEqual(swe_checks.idle_record(row, native, True), record)
        with self.assertRaisesRegex(sc.CheckFailed, 'without --release-idle-pages'):
            swe_checks.idle_record(row, native, False)
        self.assertIsNone(swe_checks.idle_record({}, {}, False))
        for bad in (dict(record, readmitted_pages=[5, 4]), dict(record, evicted_sessions=['a'], evicted_pages=0),
                    dict(record, extra=1)):
            with self.assertRaises(sc.CheckFailed):
                swe_checks.idle_record(dict(idle_release=bad), dict(idle_release=bad), True)
        frames = [dict(operation=op) for op in ('prepare', 'readmit', 'phase1', 'phase1_drain', 'reserve',
                                                'allocation_update', 'complete')]
        self.assertEqual(sc.frame_ops(frames, 3, readmit=True), 1)
        with self.assertRaises(sc.CheckFailed):
            sc.frame_ops(frames, 3)  # a readmit frame without the record's readmitted Pages
        with self.assertRaises(sc.CheckFailed):
            sc.frame_ops([f for f in frames if f['operation'] != 'readmit'], 3, readmit=True)

    def test_capacity_stall(self):
        def stats(stamp, running):
            return ('(APIServer pid=1) INFO 11-11 %s [loggers.py:273] Engine 000: Avg prompt throughput: 0.0 tokens/s, '
                    'Avg generation throughput: 0.0 tokens/s, Running: %d reqs, Waiting: 0 reqs, GPU KV cache usage: '
                    '99.0%%\n' % (stamp, running))
        refusal = ('(EngineCore pid=2) REMOTE_RA_REGISTERED_ADD {"error": "CapacityWait: CAPACITY_WAIT: FA pool: turn needs '
                   '3 (phase 1) / 259 (phase 2) blocks, running requests %d more, watermark 0; 214 free", "kind": '
                   '"RegisteredAdd", "phase": "engine_rejected", "pid": 2, "request_sha256": "%s"}\n')
        stall = stats('11:17:05', 0) + refusal % (0, 'a' * 64) * 400 + stats('12:12:41', 2)
        self.assertEqual(swe_checks.capacity_stalls(stall), [dict(start='11-11 11:17:05', end='11-11 12:12:41',
                                                                  seconds=3336, refusals=400, accepted=0)])
        self.assertEqual(swe_checks.capacity_stalls(stats('11:17:05', 0) + stats('12:12:41', 2)), [])  # idle, no wait
        self.assertEqual(swe_checks.capacity_stalls(stats('11:17:05', 0) + refusal % (0, 'a' * 64) * 400
                                                    + stats('11:19:05', 2)), [])  # 120 s: not a stall
        self.assertEqual(swe_checks.capacity_stalls(stats('11:17:05', 3) + refusal % (3, 'a' * 64) * 400
                                                    + stats('12:12:41', 2)), [])  # requests ran
        self.assertEqual(len(swe_checks.capacity_stalls(stats('11:17:05', 0) + refusal % (0, 'a' * 64) * 400)), 1)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'server.log'
            log.write_text(stall, encoding='utf-8')
            with self.assertRaisesRegex(swe_checks.CheckFailed, 'CAPACITY_STALL'):
                swe_checks.check_capacity_stall(log)
            log.write_text(stats('11:17:05', 3) + stats('11:17:15', 0), encoding='utf-8')
            self.assertEqual(swe_checks.check_capacity_stall(log)['stalls'], 0)
        real = Path(r'D:\Remote-RA-temp\scratch\fullti-3579273\evidence\core\server-full262t.log')
        if real.is_file():  # job 3579273's own log: the 55.6-min stall
            found = swe_checks.capacity_stalls(real.read_text(encoding='utf-8', errors='replace'))
            self.assertEqual([(s['start'], s['end'], s['seconds']) for s in found],
                             [('11-11 11:17:05', '11-11 12:12:41', 3336)])

    def test_the_client_timeout_wording(self):
        failure = dict(session_index=7, turn=27, request_id=None, capacity_waits=2826, queue_wait_s=3403.2,
                       error='CancelledError', detail='')
        outcome = dict(status='done', exit_status='Timeout', api_calls=28)
        rows = [dict(turn=t) for t in range(27)]
        with self.assertRaisesRegex(swe_checks.CheckFailed, '^client timeout during CAPACITY_WAIT: session 7'):
            swe_checks.check_session(7, 'astropy__astropy-14096', outcome, rows, [failure])
        with self.assertRaisesRegex(swe_checks.CheckFailed, '^session 7'):  # without the waits: the old message
            swe_checks.check_session(7, 'astropy__astropy-14096', outcome, rows, [dict(failure, capacity_waits=0)])
        tasks = [dict(instance_id='i%d' % k) for k in range(8)]
        allowed, disallowed = swe_checks.classify_failures([failure], {7: rows}, {'i7': outcome}, tasks)
        self.assertEqual((allowed, [d['reason'] for d in disallowed]), ([], ['client timeout during CAPACITY_WAIT of i7']))

    def test_the_goal_wait_metric(self):
        import join_swe32
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = [dict(goal_async_ms=v) for v in (0.0, 1500.0, 0.0, 2500.0, 500.0)]
            (out / 'measurements.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
            records = [dict(wall_s=40.0), dict(wall_s=60.0), dict(wall_s=None)]
            got = join_swe32.goal_wait(dict(kind='runtime', policy='goal_query'), records, out)
            self.assertEqual(got, dict(goal_wait_s=4.5, goal_wait_share=0.045, goal_wait_turns=3, goal_wait_ms_p50=1500.0,
                                       goal_wait_ms_p95=2500.0, goal_wait_unknown=0, goal_wait_wall_s=100.0))
            self.assertEqual(join_swe32.goal_wait(dict(kind='runtime', policy='observation_query'), records, out), {})
            self.assertEqual(join_swe32.goal_wait(dict(kind='stock', policy='stock'), records, out), {})



class PolicyRegistry1112(unittest.TestCase):
    """2026-11-12 (policy-registry-1112, integrate-1112): every page-selection policy comes from the sealed runtime's
    policy_registry.describe() (runtime_features.policy_registry), not from serve.py's / config.py's POLICIES literals
    or the selection contract's markers; the features read from the rows equal what the text readers read at main
    ba4d1ba; the plan checks derive their policy sets from the rows and run only policies this harness's evidence
    checks verify (CHECKED); a runtime without the registry is refused."""

    def test_the_pinned_rows_and_the_features_read_from_them(self):
        rows = PINS['runtime_features'][c.REGISTRY_KEY]
        self.assertEqual(rows, REGISTRY_ROWS)
        self.assertEqual([r['name'] for r in rows], REGISTERED)  # 51b4085: + ledger_query; a2f9e36: + the markers
        named = {r['name']: r for r in rows}
        features = PINS['runtime_features']
        # the values of the text readers at main ba4d1ba (the frozen payload 67c55f26), now read from the rows
        self.assertEqual(features['policies'], REGISTERED)
        self.assertEqual(features['random_fill'], dict(seed=42, rng='splitmix64_fisher_yates_v1'))
        self.assertEqual(features['window'], dict(policies=['recent_fill', 'recent_tools'], orders=c.WINDOW_ORDERS,
                                                  evidence_key='window'))
        self.assertEqual(features['akv'], dict(policy='akv_rule', pin=c.AKV_PIN, evidence_key='window'))
        self.assertIs(features['observation_query_rewriter'], False)
        self.assertEqual((c.window_feature(rows), c.akv_feature(named), c.random_fill_pin(named)),
                         (features['window'], features['akv'], features['random_fill']))
        self.assertFalse(hasattr(c, 'policy_list') or hasattr(c, 'RUNTIME_POLICIES') or hasattr(c, 'WINDOW_MODULE'))

    def test_the_policy_sets_and_the_checks_reference(self):
        self.assertEqual(c.runtime_policies(PINS), ('full', 'goal_query', 'observation_query', 'random_fill',
                                                    'recent_fill', 'recent_tools', 'akv_rule', 'ledger_query'))
        self.assertEqual(set(c.runtime_policies(PINS)), set(c.CHECKED))
        self.assertEqual((c.window_policies(PINS), c.rule_policies(PINS), c.rewriter_policies(PINS)),
                         (('recent_fill', 'recent_tools'), ('akv_rule',), ('goal_query', 'goal_query_markers')))
        # (a2f9e36: the marker plug-ins are registered but no swe32 runtime policy: no evidence check here)
        for name in ('goal_query_markers', 'observation_query_markers'):
            self.assertIn('no evidence check', c.policy_problem(PINS, name))
        self.assertEqual(c.budget_policies(PINS), ('goal_query', 'observation_query', 'random_fill', 'recent_fill',
                                                   'recent_tools', 'ledger_query', 'goal_query_markers',
                                                   'observation_query_markers'))  # (a2f9e36: + the markers)
        self.assertIn('not a swe32 runtime policy', c.policy_problem(PINS, 'random50'))
        self.assertIn('does not register', c.policy_problem(PINS, 'oldest_first'))
        self.assertIsNone(c.policy_problem(PINS, 'akv_rule'))
        named = {r['name']: r for r in REGISTRY_ROWS}
        # a plug-in (a window policy of its own order): registered and a window policy, but no evidence check here
        plug = dict(named['recent_fill'], name='oldest_first', builtin=False, contract_key='oldest_first',
                    order_id='oldest_first_v1', pin=dict(order='oldest_first_v1'),
                    rewriter_unused='oldest_first fills the remaining Pages in its own order and ranks nothing')
        pins = copy.deepcopy(PINS)
        pins['runtime_features'][c.REGISTRY_KEY].append(plug)
        self.assertIn('oldest_first', c.window_policies(pins))
        self.assertNotIn('oldest_first', c.runtime_policies(pins))
        self.assertIn('no evidence check', c.policy_problem(pins, 'oldest_first'))
        # a row that differs from what the checks verify: refused by name of the property
        for policy, change, word in (('recent_tools', dict(order_id='x_v1', pin=dict(order='x_v1')), 'order_id, pin'),
                                     ('observation_query', dict(goal_engine=True), 'goal_engine'),
                                     ('akv_rule', dict(pin=dict(rule='replies_and_latest_tool_responses_v1',
                                                                latest_tool_responses=8)), 'pin'),
                                     ('random_fill', dict(pin=dict(seed=42, rng='other')), 'pin')):
            pins = copy.deepcopy(PINS)
            row = next(r for r in pins['runtime_features'][c.REGISTRY_KEY] if r['name'] == policy)
            row.update(change)
            self.assertNotIn(policy, c.runtime_policies(pins), policy)
            self.assertIn(word, c.policy_problem(pins, policy), policy)
        # CHECKED is smoke_checks' own reference (the evidence checks re-implement each fill and rule)
        import smoke_checks as sc
        self.assertEqual({n: v['order_id'] for n, v in c.CHECKED.items() if v['kind'] == 'window'}, sc.WINDOW_ORDERS)
        self.assertEqual((c.CHECKED['akv_rule']['pin'], c.CHECKED['akv_rule']['rewriter_unused']),
                         (sc.AKV_PIN, sc.GOAL_REWRITER_UNUSED_AKV))
        self.assertEqual(c.CHECKED['random_fill']['order_id'], sc.RANDOM_FILL_RNG)
        self.assertEqual({n: v['rewriter_unused'] for n, v in c.CHECKED.items() if v['kind'] == 'window'},
                         sc.GOAL_REWRITER_UNUSED_WINDOW)
        self.assertEqual((c.CHECKED['observation_query']['rewriter_unused'], c.CHECKED['random_fill']['rewriter_unused']),
                         (sc.GOAL_REWRITER_UNUSED, sc.GOAL_REWRITER_UNUSED_RANDOM_FILL))

    def test_registry_rows_runs_describe_and_refuses_without_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'source-overlay').mkdir()
            with self.assertRaisesRegex(RuntimeError, 'has no policy registry'):
                c.registry_rows(root)
            with self.assertRaisesRegex(RuntimeError, 'has no policy registry'):
                c.runtime_features(root)  # (before any other file is read)
            module = root / 'source-overlay' / 'policy_registry.py'
            for source, pattern in (('raise SystemExit(3)\n', 'failed'),
                                    ('import sys\nsys.stdout.write("x")\ndef describe():\n    return []\n', 'no JSON'),
                                    ('def describe():\n    return [dict(name="a")]\n', 'rows lack'),
                                    ('def describe():\n    return %r\n' % ([REGISTRY_ROWS[0], REGISTRY_ROWS[0]],),
                                     'not distinct')):
                module.write_text(source, encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError, pattern):
                    c.registry_rows(root)
            module.write_text('def describe():\n    return %r\n' % (REGISTRY_ROWS,), encoding='utf-8')
            self.assertEqual(c.registry_rows(root), REGISTRY_ROWS)
            self.assertEqual(list(root.rglob('__pycache__')), [])  # -B: nothing written into the source
        tree = ROOT.parents[2] / 'worktrees' / 'integrate-1116'  # the pinned commit's worktree, when at the pin
        if (tree / c.REGISTRY_MODULE).is_file() and subprocess.run(
                ['git', '-C', str(tree), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() == \
                PINS['commit']:
            before = sorted(str(p) for p in (tree / 'source-overlay').rglob('__pycache__'))
            self.assertEqual(c.registry_rows(tree), PINS['runtime_features'][c.REGISTRY_KEY])
            self.assertEqual(sorted(str(p) for p in (tree / 'source-overlay').rglob('__pycache__')), before)

    def test_the_checks_and_plans_read_the_registry(self):
        c.check_features(PINS['runtime_features'], PINS['config'])
        for name in PLANS:
            c.validate_plan(PINS, name)
        pins = copy.deepcopy(PINS)
        pins['runtime_features'].pop(c.REGISTRY_KEY)
        with self.assertRaisesRegex(RuntimeError, 'policy registry'):
            c.check_features(pins['runtime_features'], pins['config'])
        with self.assertRaisesRegex(RuntimeError, 'policy registry'):
            c.validate_plan(pins, 'obs')
        pins = copy.deepcopy(PINS)  # a registry without recent_fill: the plans that run it are refused
        pins['runtime_features'][c.REGISTRY_KEY] = [r for r in REGISTRY_ROWS if r['name'] != 'recent_fill']
        with self.assertRaisesRegex(RuntimeError, r"plans \['recent64', 'recent64a', 'recent64ak24', 'recenth2a', "
                                                  r"'recenth2s', 'recentt', "
                                                  r"'rfillasync32t', 'rfillasync32ti', 'rfillsync32t', 'rfillsync32ti'\] "
                                                  r"run recent_fill: the sealed"):
            c.check_features(pins['runtime_features'], pins['config'])
        with self.assertRaisesRegex(RuntimeError, 'does not register'):
            c.validate_plan(pins, 'recentt')
        pins = copy.deepcopy(PINS)  # random50 (the native half draw) is no swe32 runtime policy
        pins['config']['plans']['obs']['servers'][0]['policy'] = 'random50'
        pins['config']['policies']['random50'] = dict(policy_args=[], admission=[], rewriter=False)
        with self.assertRaisesRegex(RuntimeError, 'not a swe32 runtime policy'):
            c.validate_plan(pins, 'obs')
        named = {r['name']: r for r in REGISTRY_ROWS}
        for plan, key, policy in (('recentt', 'window_pin', 'recent_fill'), ('rtoolst', 'window_pin', 'recent_tools'),
                                  ('akvt', 'akv_pin', 'akv_rule')):
            expect = c.server_expectation(PINS, c.plan_servers(PINS, plan)[0], None, 'm', ['t'])
            self.assertEqual(expect[key], named[policy]['pin'], plan)

    def test_selection_parity_reads_the_registry(self):
        import selection_parity
        import verify_evidence
        self.assertEqual(selection_parity.replayed(REGISTRY_ROWS), dict(observation_query=None, random_fill='fill',
                                                                        ledger_query='hook',
                                                                        observation_query_markers='hook'))
        self.assertEqual(verify_evidence.parity_policies(PINS), dict(observation_query=None, random_fill='fill',
                                                                      ledger_query='hook',
                                                                      observation_query_markers='hook'))
        self.assertEqual(selection_parity.replayed(REGISTRY_ROWS_51B4085), dict(observation_query=None,
                                                                                random_fill='fill', ledger_query='hook'))
        self.assertEqual(selection_parity.replayed(REGISTRY_ROWS_755550E), dict(observation_query=None,
                                                                                random_fill='fill'))
        stats = dict(selective_turns=2, fast_equals_reference=2, equals_recorded=2)
        self.assertTrue(selection_parity.verdict(stats))  # observation_query: no own record
        self.assertTrue(selection_parity.verdict(dict(stats, own_record_present=2)))
        self.assertFalse(selection_parity.verdict(dict(stats, own_record_present=1)))  # random_fill without its fill


class StockThinking1112(unittest.TestCase):
    """2026-11-12: the stock control with thinking ON (config.stock.thinking, plans stock64t / stock32t): vLLM's
    --reasoning-parser qwen3, the agents' chat_template_kwargs enable_thinking true without a stop string, history
    reasoning_dropped; the login gate, the argv gate, the job's header probe and stock-thinking; every earlier stock
    plan unchanged (thinking off by contract)."""

    def test_the_contract_and_the_stock_servers(self):
        st = c.stock_thinking(PINS)
        self.assertEqual((st['reasoning_parser'], st['history'], st['stock_model_kwargs']),
                         ('qwen3', 'reasoning_dropped', dict(extra_body=dict(chat_template_kwargs=dict(enable_thinking=True)),
                                                             max_tokens=4096, temperature=0.0)))
        base = c.plan_servers(PINS, 'stock25')[0]
        for name, tasks in (('stock64t', 64), ('stock32t', 32), ('stock32ti', 32)):
            (srv,) = c.plan_servers(PINS, name)
            self.assertEqual((srv['kind'], srv[c.THINKING_KEY], srv['tasks'], srv[c.STOCK_SETTINGS_KEY]),
                             (c.STOCK, True, tasks, dict(kv_cache_gib=25, max_model_len=262144, preemption='finding')))
            self.assertTrue(c.thinking_on(PINS, srv))
            self.assertIsNone(c.thinking_record(PINS, srv))
            argv, plain = c.stock_argv(PINS, 8123, srv), c.stock_argv(PINS, 8123, base)
            self.assertEqual(argv.count('--reasoning-parser'), 1)
            at = argv.index('--reasoning-parser')
            self.assertEqual(argv[:at] + argv[at + 2:], plain)  # stock25's argv plus the parser, nothing else
            self.assertEqual(argv[at + 1], 'qwen3')
            expected = c.stock_expected(PINS, 8123, srv)
            self.assertEqual({k: v for k, v in expected.items() if c.stock_expected(PINS, 8123, base).get(k) != v},
                             dict(reasoning_parser='qwen3'))
            text = c.model_override_text(PINS, srv, 23004).splitlines()
            self.assertIn('    extra_body: {"chat_template_kwargs": {"enable_thinking": true}}', text)
            self.assertFalse([line for line in text if 'stop' in line or 'include_stop_str_in_output' in line])
            self.assertEqual([line for line in text if 'max_tokens' in line or 'temperature' in line],
                             ['    temperature: 0.0', '    max_tokens: 4096'])
        for name in ('stock', 'stocki', 'stock25', 'stock25x'):  # thinking off by contract, unchanged
            for srv in c.plan_servers(PINS, name):
                self.assertNotIn(c.THINKING_KEY, srv, name)
                self.assertFalse(c.thinking_on(PINS, srv))
                self.assertNotIn('--reasoning-parser', c.stock_argv(PINS, 8123, srv))
                self.assertNotIn('reasoning_parser', c.stock_expected(PINS, 8123, srv))
                self.assertIn('    stop: ["</mswea_bash_command>"]', c.model_override_text(PINS, srv, 23004).splitlines())

    def test_the_refusals(self):
        for change, pattern in (
                (lambda p: p['config']['stock']['thinking']['stock_model_kwargs'].update(stop=['</mswea_bash_command>']),
                 'no stop'),
                (lambda p: p['config']['stock']['thinking'].update(history='kept'), 'history reasoning_dropped'),
                (lambda p: p['config']['stock']['thinking'].update(reasoning_parser='deepseek_r1'), 'reasoning_parser'),
                (lambda p: p['config']['stock']['thinking']['stock_model_kwargs'].update(max_tokens=1536), 'max_tokens'),
                (lambda p: p['config']['stock'].pop('thinking'), 'no thinking contract'),
                (lambda p: p['config']['plans']['stock64t']['servers'].append(dict(
                    label='obs64x', policy='observation_query', step_limit=120, agent_cap_seconds=100,
                    agent_min_seconds=50)), 'runtime servers only or of stock servers only')):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, 'stock64t')

    def test_time_cost_and_estimate(self):
        # the thinking factor of the user's choice (11-12): our obs262t / obs262 span ratio, 2,503 / 1,766 s = 1.417
        for name, partition, time, expected, cap, minimum, wait, hours in (
                ('stock64t', 'ghx4', '02:39:00', 7216, 7680, 5410, 1200, (2.65, 2.004)),
                ('stock32t', 'ghx4', '01:24:00', 3397, 3600, 2290, 600, (1.4, 0.944)),
                ('stock32ti', 'ghx4-interactive', '01:24:00', 3397, 3600, 2290, 600, (2.8, 1.887))):
            plan, (srv,) = c.cfg(PINS)['plans'][name], c.plan_servers(PINS, name)
            fit, cost = c.validate_plan(PINS, name), c.plan_cost(PINS, name)
            self.assertEqual((plan['time'], plan['partition'], plan['expected_seconds'], srv['agent_cap_seconds'],
                              srv['agent_min_seconds'], plan['eval_wait_seconds'],
                              (cost['gpu_hours_limit'], cost['gpu_hours_expected'])),
                             (time, partition, expected, cap, minimum, wait, hours), name)
            self.assertIn('ESTIMATE', plan['about'])
            self.assertIn('NOT GPU-validated', plan['about'])
            self.assertIn('1.417', plan['about'])
            self.assertEqual(c.memory_fit(PINS, name)['need_gib'], 32.0)
            self.assertLessEqual(cost['limit_over_expected'], c.MAX_LIMIT_OVER_EXPECTED)
            self.assertEqual(fit['time_seconds'] - fit['worst_case_seconds'], 600 if name == 'stock64t' else 180)
            self.assertTrue(fit['cap_after_wait_fits'] or name == 'stock64t', name)
        self.assertEqual(c.cfg(PINS)['plans']['stock32ti']['variant_of'], 'stock32t')
        self.assertEqual(c.plan_servers(PINS, 'stock32ti'), c.plan_servers(PINS, 'stock32t'))
        self.assertEqual(c.model_override_text(PINS, c.plan_servers(PINS, 'stock32ti')[0], 23004),
                         c.model_override_text(PINS, c.plan_servers(PINS, 'stock32t')[0], 23004))
        self.assertNotIn('stock64ti', c.cfg(PINS)['plans'])

    def test_the_probe_and_the_check(self):
        import stock
        contract = c.stock_thinking(PINS)

        def server(on, off, http=200):
            def post(url, body, timeout):
                thinking = body['chat_template_kwargs']['enable_thinking']
                self.assertEqual((url, body['max_tokens'], body['temperature']), ('http://h/v1/chat/completions', 1, 0.0))
                message = dict(content=None, reasoning='Okay') if thinking else dict(content='THOUGHT')
                return http, json.dumps(dict(choices=[dict(message=message)],
                                             usage=dict(prompt_tokens=on if thinking else off))).encode()
            return post
        good = stock.thinking_probe('http://h', 'selective-prefix', contract, post=server(30, 32))
        self.assertEqual((good['status'], good['difference'], good['on']['reasoning'], good['off']['content']),
                         ('PASS', -2, 'Okay', 'THOUGHT'))
        for bad in (server(32, 32), server(31, 32), server(30, 32, http=400)):
            self.assertEqual(stock.thinking_probe('http://h', 'selective-prefix', contract, post=bad)['status'], 'FAIL')
        line = ("(APIServer pid=1) INFO [api_utils.py:273] non-default args: {'host': '127.0.0.1', "
                "'reasoning_parser': 'qwen3', 'max_model_len': 262144}\n")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaisesRegex(Exception, 'did not probe'):
                stock.check_thinking(out, line, contract)
            (out / stock.THINKING_PROBE).write_text(json.dumps(good), encoding='utf-8')
            detail = stock.check_thinking(out, line, contract)
            self.assertEqual((detail['reasoning_parser'], detail['history'], detail['prompt_tokens_on'],
                              detail['prompt_tokens_off']), ('qwen3', 'reasoning_dropped', 30, 32))
            with self.assertRaisesRegex(Exception, 'reasoning parser None'):
                stock.check_thinking(out, line.replace("'reasoning_parser': 'qwen3', ", ''), contract)
            (out / stock.THINKING_PROBE).write_text(json.dumps(dict(good, status='FAIL', reason='x')), encoding='utf-8')
            with self.assertRaisesRegex(Exception, 'stock thinking probe'):
                stock.check_thinking(out, line, contract)

    def test_the_template_gives_the_header_and_drops_earlier_reasoning(self):
        """The model's own template (fixtures/qwen3.6-27b-chat_template.jinja), as the login gate renders it on DeltaAI:
        enable_thinking true ends in THINKING_HEADER (false: STOCK_HEADER), and an earlier reply renders without its
        reasoning even when the reasoning reaches the template (stock vLLM 0.24 does not even pass it)."""
        try:
            import jinja2
        except ImportError:
            self.skipTest('jinja2 not importable')
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])

        def fail(message):
            raise ValueError(message)
        env.globals['raise_exception'] = fail
        template = env.from_string((ROOT / 'fixtures' / 'qwen3.6-27b-chat_template.jinja').read_text(encoding='utf-8'))
        messages = [dict(role='system', content='s'), dict(role='user', content='u')]
        render = lambda m, on: template.render(messages=m, add_generation_prompt=True, enable_thinking=on)
        self.assertTrue(render(messages, True).endswith(c.THINKING_HEADER))
        self.assertTrue(render(messages, False).endswith(c.STOCK_HEADER))
        history = render(messages + [dict(role='assistant', content='CONTENT_7f3a', reasoning_content='REASONING_7f3a'),
                                     dict(role='user', content='u2')], True)
        self.assertEqual((history.count('CONTENT_7f3a'), history.count('REASONING_7f3a')), (1, 0))
        self.assertIn("'thinking'", c.STOCK_LOGIN_PROBE.replace('"', "'"))


class Proxy64More1112(unittest.TestCase):
    """2026-11-12 (later): recent64 (recent_fill), akv64 (akv_rule with --release-idle-pages: the capacity analysis),
    rfill64 (random_fill) and rfill64i at 64 tasks, thinking ON, obs64's settings; times ESTIMATED (recentt, akvsi
    without its straggler plus the straggler allowance, obst); no recent64i / akv64i (02:00:00 is below their straggler
    bounds); every earlier plan unchanged."""

    def test_the_new_plans(self):
        plans = c.cfg(PINS)['plans']
        self.assertEqual(ORDER1112[-len(NEW1112):], NEW1112)
        self.assertEqual(PLAN_ORDER[-len(NEW1116 + LEDGER1114 + PHASEB + P150):],
                         NEW1116 + LEDGER1114 + PHASEB + P150)  # (11-17: the 150-task plans last)
        recentt, akvt, obs64 = (c.plan_servers(PINS, n)[0] for n in ('recentt', 'akvt', 'obs64'))
        wanted = dict(recent64=('recent64t', 'recent_fill', recentt, 'ghx4', '02:13:00', 5946, 5640, 3970, False,
                                (2.217, 1.652)),
                      akv64=('akv64tr', 'akv_rule', akvt, 'ghx4', '02:14:00', 6249, 5700, 2880, True, (2.233, 1.736)),
                      rfill64=('rfill64t', 'random_fill', None, 'ghx4', '02:13:00', 5343, 5640, 3450, False,
                               (2.217, 1.484)),
                      rfill64i=('rfill64t', 'random_fill', None, 'ghx4-interactive', '02:00:00', 5343, 4860, 3450, False,
                                (4.0, 2.968)))
        for name, (label, policy, base, partition, time, expected, cap, minimum, release, hours) in wanted.items():
            plan, (srv,) = plans[name], c.plan_servers(PINS, name)
            cost = c.plan_cost(PINS, name)
            self.assertEqual((srv['label'], srv['policy'], srv['tasks'], plan['partition'], plan['time'],
                              plan['expected_seconds'], srv['agent_cap_seconds'], srv['agent_min_seconds'],
                              plan['eval_wait_seconds'], c.release_idle_on(PINS, srv), srv[c.THINKING_KEY],
                              (cost['gpu_hours_limit'], cost['gpu_hours_expected'])),
                             (label, policy, 64, partition, time, expected, cap, minimum, 1200, release, True, hours),
                             name)
            self.assertEqual(c.server_cfg(PINS, srv), dict(c.server_cfg(PINS, obs64), **(
                {c.RELEASE_KEY: True} if release else {})), name)
            fit = c.validate_plan(PINS, name)
            self.assertEqual(fit['time_seconds'] - fit['worst_case_seconds'], 600, name)
            self.assertEqual(c.memory_fit(PINS, name)['need_gib'], 105.29)
            self.assertIn('ESTIMATE', plan['about'] if 'variant_of' not in plan else plans[plan['variant_of']]['about'])
            args = c.launch_arguments(PINS, srv, REWRITER)
            self.assertEqual((args.count('--enable-thinking'), args.count('--release-idle-pages')),
                             (1, 1 if release else 0), name)
            self.assertEqual(args.count('--active-token-budget'), 0 if policy == 'akv_rule' else 1, name)
            if base is not None:  # the 32-task thinking-on server of the same policy but for label, cap, tasks, release
                mine, theirs = dict(srv), dict(base)
                for k in ('label', 'agent_cap_seconds', 'agent_min_seconds', 'tasks', 'settings'):
                    mine.pop(k), theirs.pop(k)
                self.assertEqual(mine, theirs, name)
                self.assertEqual([a for a in args if a != '--release-idle-pages'],
                                 c.launch_arguments(PINS, base, REWRITER), name)
        self.assertEqual(plans['rfill64i']['variant_of'], 'rfill64')
        for absent in ('recent64i', 'akv64i', 'stock64ti'):
            self.assertNotIn(absent, plans)
        self.assertIn('183 of 200', plans['akv64']['about'])  # the capacity analysis behind its idle release

    def test_every_earlier_plan_is_unchanged(self):
        recorded = json.loads((ROOT / 'fixtures' / 'swe32-plans-4f5f4e7.json').read_bytes())
        import plan_fingerprint
        for name, parts in recorded['plans'].items():
            self.assertEqual(plan_fingerprint.fingerprint(pins_51b4085_rows(), name), parts, name)
        for name in PLANS[:PLANS.index('fulltri') + 1]:  # no earlier stock server runs thinking on
            for s in c.plan_servers(PINS, name):
                self.assertFalse(s['kind'] == c.STOCK and c.thinking_on(PINS, s), name)



class StockKeepReasoning1116(unittest.TestCase):
    """2026-11-16: stockkeep32t / stockkeepsync32t, the stock control with the earlier reasoning KEPT: stock32t's server
    plus --default-chat-template-kwargs '{"preserve_thinking": true}' (argv, the argv gate's expectation), the agents'
    keep model class (tbench/agent_ext/swe32_keep_reasoning.py, pinned in config.stock.thinking.kept), the login probe's
    kept render, the tbench gate's module SHA-256, the job's keep probe and stock-thinking's kept checks;
    stockkeepsync32t also --no-async-scheduling; every other stock plan's contract, argv and override unchanged."""

    def test_the_plans_differ_from_stock32t_only_as_decided(self):
        base = server('stock32t')
        for name, sync in (('stockkeep32t', False), ('stockkeepsync32t', True)):
            mine = server(name)
            self.assertEqual(mine['stock_settings'], dict(kv_cache_gib=25, max_model_len=262144, preemption='finding',
                                                          history='reasoning_kept',
                                                          **(dict(async_scheduling=False) if sync else {})))
            argv, plain = c.stock_argv(PINS, 8123, mine), c.stock_argv(PINS, 8123, base)
            extra = ['--no-async-scheduling'] if sync else []
            at = argv.index('--reasoning-parser') + 2
            self.assertEqual(argv[at:at + 2], ['--default-chat-template-kwargs', '{"preserve_thinking": true}'])
            self.assertEqual([a for a in argv[:at] + argv[at + 2:] if a not in extra], plain)
            self.assertEqual(argv.count('--no-async-scheduling'), int(sync))
            expected, before = c.stock_expected(PINS, 8123, mine), c.stock_expected(PINS, 8123, base)
            self.assertEqual(expected.pop('default_chat_template_kwargs'), {'preserve_thinking': True})
            self.assertEqual(expected.pop('async_scheduling'), False if sync else None)
            before.pop('async_scheduling')
            self.assertEqual(expected, before)
            text, plain_text = c.model_override_text(PINS, mine, 24001), c.model_override_text(PINS, base, 24001)
            self.assertEqual(text.replace('model_class: swe32_keep_reasoning.KeepReasoningTextbasedModel',
                                          'model_class: litellm_textbased'), plain_text)
            self.assertEqual(c.stock_contract(PINS, mine)['history'], 'reasoning_kept')
            self.assertEqual(c.stock_config(PINS, mine)['async_scheduling'], not sync)
            self.assertEqual((c.stock_config(PINS, mine)['enable_prefix_caching'],
                              c.stock_config(PINS, mine)['mamba_cache_mode']), (True, 'align'))  # as stock32t
        for name in PLAN_ORDER:  # every other stock server: config.stock.thinking itself, no kwargs
            for s in c.plan_servers(PINS, name):
                if s['kind'] == c.STOCK and name not in KEPT:
                    self.assertIsNone(c.stock_kept(PINS, s), name)
                    self.assertNotIn(c.STOCK_DEFAULT_KWARGS_ARG, c.stock_argv(PINS, 8000, s), name)
                    self.assertNotIn('default_chat_template_kwargs', c.stock_expected(PINS, 8000, s), name)
                    if c.thinking_on(PINS, s):
                        self.assertIs(c.stock_contract(PINS, s), c.stock_thinking(PINS), name)

    def test_the_time_plan(self):
        for name, (time, expected, cap, minimum) in dict(stockkeep32t=('01:09:00', 2765, 2880, 1985),
                                                         stockkeepsync32t=('01:14:00', 2998, 3180, 2180)).items():
            plan = c.cfg(PINS)['plans'][name]
            (s,) = plan['servers']
            self.assertEqual((plan['partition'], plan['time'], plan['expected_seconds'], s['agent_cap_seconds'],
                              s['agent_min_seconds'], plan['eval_wait_seconds']),
                             ('ghx4', time, expected, cap, minimum, 600), name)
            fit = c.validate_plan(PINS, name)
            self.assertTrue(fit['fits'])
            self.assertLessEqual(c.plan_cost(PINS, name)['limit_over_expected'], c.MAX_LIMIT_OVER_EXPECTED)
            self.assertEqual(c.memory_fit(PINS, name)['fits'], True)

    def test_the_agent_module(self):
        kept = c.cfg(PINS)['stock']['thinking']['kept']
        path = ROOT / kept['agent_module']['source']
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), kept['agent_module']['sha256'])
        self.assertEqual(kept['agent_module']['tbench_path'], '/home/ubuntu/swe32/upstream/swe32_keep_reasoning.py')
        import importlib.util
        spec = importlib.util.spec_from_file_location('swe32_keep_reasoning_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # no Mini-SWE-Agent here: only with_reasoning
        messages = [dict(role='system', content='s'), dict(role='user', content='u'),
                    dict(role='assistant', content='C', reasoning_content='R', provider_specific_fields={'reasoning': 'R'}),
                    dict(role='assistant', content='D', reasoning_content=None), dict(role='user', content='v')]
        out = module.with_reasoning(messages)
        self.assertEqual(out[2], dict(messages[2], reasoning='R'))
        self.assertEqual([m for i, m in enumerate(out) if i != 2], [m for i, m in enumerate(messages) if i != 2])
        self.assertNotIn('reasoning', messages[2])  # the input (the trajectory's messages) is not changed
        # the tbench half's manifest does not name it (the queued jobs' deployed half stays as pinned)
        self.assertFalse(any('agent_ext' in line or 'swe32_keep_reasoning' in line
                             for line in (ROOT / 'tbench' / 'deploy_tbench.ps1').read_text(encoding='utf-8').splitlines()
                             if not line.lstrip().startswith('#')))

    def test_the_tbench_gate_checks_the_module(self):
        class Link:
            def __init__(self, sha):
                self.sha, self.calls = sha, []

            def preflight(self, instances, statements, agent):
                return dict(status='TBENCH_PREFLIGHT_PASS', preflight=dict(manifest_sha256=PINS['tbench_half'][
                    'manifest_sha256']))

            def call(self, label, remote, timeout):
                self.calls.append(remote)
                return dict(returncode=0, stdout='%s  %s\n' % (self.sha, remote[1]), stderr='')
        pinned = c.cfg(PINS)['stock']['thinking']['kept']['agent_module']['sha256']
        with tempfile.TemporaryDirectory(prefix='swe32-keepgate-') as tmp:
            link = Link(pinned)
            value = c.tbench_gate(tmp, PINS, dict(plan='stockkeep32t'), link)
            self.assertEqual(value['agent_module']['sha256'], pinned)
            self.assertEqual(link.calls, [['sha256sum', '/home/ubuntu/swe32/upstream/swe32_keep_reasoning.py']])
        with tempfile.TemporaryDirectory(prefix='swe32-keepgate-') as tmp:
            with self.assertRaisesRegex(RuntimeError, 'deploy_agent_ext'):
                c.tbench_gate(tmp, PINS, dict(plan='stockkeepsync32t'), Link('0' * 64))
        with tempfile.TemporaryDirectory(prefix='swe32-keepgate-') as tmp:
            link = Link('0' * 64)
            c.tbench_gate(tmp, PINS, dict(plan='stock32t'), link)  # a reasoning-dropped plan never asks
            self.assertEqual(link.calls, [])

    def test_the_keep_probe_and_the_check(self):
        import stock
        contract = c.stock_contract(PINS, server('stockkeep32t'))

        def post(base, prompt):
            def call(url, body, timeout):
                n = prompt(body)
                return 200, json.dumps(dict(choices=[dict(message=dict(content=None, reasoning='x'))],
                                            usage=dict(prompt_tokens=n))).encode()
            return call

        def real(body):  # the pinned tokenizer's numbers (test_real_tokenizer.StockKeep checks them)
            kwargs = dict(contract['kept']['default_chat_template_kwargs'], **body['chat_template_kwargs'])
            reply = body['messages'][2]
            if not kwargs['preserve_thinking']:
                return 44
            return 62 if 'reasoning' in reply else 48
        value = stock.keep_probe('http://x', 'selective-prefix', contract, post=post(None, real))
        self.assertEqual((value['status'], value['differences']), ('PASS', stock.KEEP_PROBE_DIFFERENCES))
        self.assertEqual([value[v]['prompt_tokens'] for v in ('kept', 'content_only', 'not_preserved')], [62, 48, 44])
        dropped = stock.keep_probe('http://x', 'selective-prefix', contract, post=post(None, lambda body: 44))
        self.assertEqual(dropped['status'], 'FAIL')  # a server without preserve_thinking: every variant the same
        no_field = stock.keep_probe('http://x', 'selective-prefix', contract,
                                    post=post(None, lambda body: 44 if not body['chat_template_kwargs'].get(
                                        'preserve_thinking', True) else 48))
        self.assertEqual(no_field['status'], 'FAIL')  # the default applied but `reasoning` not read
        with tempfile.TemporaryDirectory(prefix='swe32-keepcheck-') as tmp:
            out = Path(tmp)
            (out / stock.THINKING_PROBE).write_text(json.dumps(dict(
                status='PASS', on=dict(prompt_tokens=34), off=dict(prompt_tokens=36),
                kwargs_on=contract['stock_model_kwargs']['extra_body']['chat_template_kwargs'])), encoding='utf-8')
            text = ("non-default args: {'host': '127.0.0.1', 'reasoning_parser': 'qwen3', "
                    "'default_chat_template_kwargs': {'preserve_thinking': True}, 'seed': 42}\n")
            with self.assertRaisesRegex(Exception, 'keep probe|did not probe'):
                stock.check_thinking(out, text, contract)
            (out / stock.KEEP_PROBE).write_text(json.dumps(value), encoding='utf-8')
            got = stock.check_thinking(out, text, contract)
            self.assertEqual((got['history'], got['keep_probe_prompt_tokens']), ('reasoning_kept', [62, 48, 44]))
            with self.assertRaisesRegex(Exception, 'default chat template kwargs'):
                stock.check_thinking(out, text.replace("'default_chat_template_kwargs': {'preserve_thinking': True}, ",
                                                       ''), contract)
            # the reasoning-dropped contract never reads the keep probe (stock32t's check unchanged)
            plain = stock.check_thinking(out, text, c.stock_contract(PINS, server('stock32t')))
            self.assertNotIn('keep_probe_prompt_tokens', plain)

    def test_the_login_gate_needs_the_kept_render(self):
        calls = []

        def fake_run(stage, name, argv, timeout, env=None):
            calls.append(argv)
            keep = argv[-1] == 'keep'
            line = dict(version='0.24.0', file='/s/vllm/__init__.py', VLLM_USE_FLASHINFER_SAMPLER=False,
                        generation_prompt_tail=c.STOCK_HEADER, thinking_prompt_tail=c.THINKING_HEADER, history_content=1,
                        history_reasoning=0, history_tail=c.THINKING_HEADER,
                        **(dict(kept_content=1, kept_reasoning=1, kept_block=1, kept_tail=c.THINKING_HEADER)
                           if keep else {}))
            (Path(stage) / ('%s.log' % name)).write_text(json.dumps(line) + '\n', encoding='utf-8')
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(prefix='swe32-keeplogin-') as tmp, \
                patch.object(c, 'run', fake_run), patch.object(c, 'gate_env', lambda stage, pins: {}):
            value = c.stock_gate(Path(tmp), PINS, 'py', '/s', server('stockkeep32t'))
            self.assertEqual(calls[-1][-2:], ['thinking', 'keep'])
            self.assertIn('--default-chat-template-kwargs', value['argv'])
            c.stock_gate(Path(tmp), PINS, 'py', '/s', server('stock32t'))
            self.assertEqual(calls[-1][-1], 'thinking')


class AsyncPair1116(unittest.TestCase):
    """2026-11-16: rfillasync32t / rfillsync32t = recentt's recent_fill server on 4ebff97, the plan switch async_scheduling
    true / false: every derived value equal but the async switch (serve.py --async-scheduling, the expectation's
    async_scheduling); the same task list, settings, override and pool spec; the async plan needs all three async fixes."""

    def test_the_pair_differs_only_in_async(self):
        a, s, r = server('rfillasync32t'), server('rfillsync32t'), server('recentt')
        mask = lambda v: {k: x for k, x in v.items() if k not in ('label', c.ASYNC_KEY)}  # noqa: E731
        self.assertEqual(mask(a), mask(s))
        self.assertEqual((a[c.ASYNC_KEY], s[c.ASYNC_KEY]), (True, False))
        self.assertEqual({k: x for k, x in mask(a).items() if k not in ('agent_cap_seconds', 'agent_min_seconds')},
                         {k: x for k, x in r.items() if k not in ('label', 'agent_cap_seconds', 'agent_min_seconds')})
        la, ls = c.launch_arguments(PINS, a, REWRITER), c.launch_arguments(PINS, s, REWRITER)
        self.assertEqual([x for x in la if x != c.ASYNC_SWITCH], ls)
        self.assertEqual(la.count(c.ASYNC_SWITCH), 1)
        self.assertEqual(ls, c.launch_arguments(PINS, r, REWRITER))  # recentt's arguments on this pin
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m/qwen3-0.6b', config_sha256='0' * 64)
        ea, es = (dict(c.server_expectation(PINS, x, gate, 'm', ['t0']), label=None) for x in (a, s))
        self.assertEqual((ea.pop(c.ASYNC_KEY), es.pop(c.ASYNC_KEY)), (True, False))
        self.assertEqual(ea, es)
        self.assertEqual(c.model_override_text(PINS, a, 24001), c.model_override_text(PINS, s, 24001))
        pa, ps = c.pool_spec(PINS, a, 24001, 1800000000)[1], c.pool_spec(PINS, s, 24001, 1800000000)[1]
        self.assertEqual((pa.pop('label'), ps.pop('label')), ('recentasync262t', 'recentsync262t'))
        self.assertEqual(pa, ps)
        pa_, ps_ = (dict(c.cfg(PINS)['plans'][n]) for n in ('rfillasync32t', 'rfillsync32t'))
        for v in (pa_, ps_):
            v.pop('about'), v.pop('authorization'), v.pop('servers')
        self.assertEqual((pa_.pop(c.ASYNC_KEY), ps_.pop(c.ASYNC_KEY)), (True, False))
        self.assertEqual(pa_, ps_)
        self.assertEqual(c.memory_fit(PINS, 'rfillasync32t'), c.memory_fit(PINS, 'rfillsync32t'))
        self.assertEqual(c.plan_fit(PINS, 'rfillasync32t'), c.plan_fit(PINS, 'rfillsync32t'))

    def test_the_time_plan_and_the_guard(self):
        for name in ASYNC1116:
            plan = c.cfg(PINS)['plans'][name]
            (s,) = plan['servers']
            self.assertEqual((plan['partition'], plan['time'], plan['expected_seconds'], s['agent_cap_seconds'],
                              s['agent_min_seconds'], plan['eval_wait_seconds'], plan['thinking']),
                             ('ghx4', '01:22:00', 3296, 3180, 2165, 600, True), name)
            self.assertTrue(c.validate_plan(PINS, name)['fits'])
            self.assertIsNone(c.plan_blocked(PINS, name), name)
        self.assertIn('13ee618', PINS['runtime_features'][c.ASYNC_FIX_KEY]['resolve'])



def plan_diff(a, b, masked=()):
    """The keys whose values differ between two plans' config entries ('about' always masked)."""
    a, b = (dict(c.cfg(PINS)['plans'][n]) for n in (a, b))
    return sorted(k for k in set(a) | set(b) if k not in ('about',) + tuple(masked) and a.get(k) != b.get(k))


class InteractivePair1116(unittest.TestCase):
    """2026-11-16 (round 2): rfillasync32ti / rfillsync32ti = the async pair on ghx4-interactive: each differs from its
    batch plan only in partition, role, variant_of and the phrase (the same servers, settings, async switch, task list
    and --time); the two interactive plans differ only in the async switch (and their labels / phrase)."""

    def test_the_variants(self):
        for name in TI1116:
            base = name[:-1]
            self.assertEqual(plan_diff(name, base), ['authorization', 'partition', 'role', 'variant_of'], name)
            plan = c.cfg(PINS)['plans'][name]
            self.assertEqual((plan['partition'], plan['role'], plan['variant_of']), ('ghx4-interactive', 'interactive',
                                                                                     base))
            self.assertEqual(c.plan_servers(PINS, name), c.plan_servers(PINS, base))
            self.assertEqual(c.plan_fit(PINS, name), c.plan_fit(PINS, base))
            cost = c.plan_cost(PINS, name)
            self.assertEqual((cost['billing_factor'], cost['gpu_hours_limit'], cost['gpu_hours_expected']),
                             (2, 2.733, 1.831))
            self.assertIsNone(c.plan_blocked(PINS, name))
        self.assertEqual(plan_diff('rfillasync32ti', 'rfillsync32ti'), ['async_scheduling', 'authorization', 'servers',
                                                                        'variant_of'])
        a, s = (c.plan_servers(PINS, n)[0] for n in TI1116)
        self.assertEqual({k: v for k, v in a.items() if k not in ('label', c.ASYNC_KEY)},
                         {k: v for k, v in s.items() if k not in ('label', c.ASYNC_KEY)})


class SecondHalf1116(unittest.TestCase):
    """2026-11-16 (round 2): the second half, `tasks` "h2" = tasks/swe64h2-tasks.json = instances 33-64 of the sealed
    64 list (disjoint from the 32 list; the two halves are the 64 list; the first half is the 32 list), statements
    tasks/tasks-pool-h2.json = the 64 pool's last 32 entries byte for byte; each plan is its first-32 plan's settings
    except the task list, its labels, its time plan and (runtime) the async switch; each async plan's sync twin differs
    only in the async switch."""
    BASE = dict(fullh2a='fulltr', fullh2s='fulltr', recenth2a='recentt', recenth2s='recentt', obsh2a='obst',
                obsh2s='obst', goalh2a='goalt', goalh2s='goalt', stockh2t='stock32t', stockkeeph2t='stockkeep32t')

    def test_the_sealed_half(self):
        whole, first, half = (c.instance_ids(PINS, n) for n in ('obs64', None, 'fullh2a'))
        self.assertEqual((len(whole), len(first), len(half)), (64, 32, 32))
        self.assertFalse(set(first) & set(half))
        self.assertEqual(first + half, whole)
        self.assertEqual(first, PINS['tasks']['instances'])
        self.assertEqual(half, json.loads((ROOT / 'tasks' / 'swe64h2-tasks.json').read_bytes())['instances'])
        pool64 = json.loads((ROOT / 'tasks' / 'tasks-pool-64.json').read_bytes())
        self.assertEqual((ROOT / 'tasks' / 'tasks-pool-h2.json').read_bytes(), c.tasks_bytes(pool64[32:]))
        value = PINS[c.TASK_SETS_KEY]['h2']
        self.assertEqual((value['pool']['file'], value['provisional'], value['list_file']),
                         ('tasks-pool-h2.json', False, 'tasks/swe64h2-tasks.json'))
        payload = json.loads((ROOT / 'dist' / 'payload.json').read_bytes())
        self.assertEqual(payload['files']['tasks-pool-h2.json']['sha256'], value['pool']['sha256'])
        for name in H2:
            self.assertEqual(c.plan_task_count(PINS, name), 'h2')
            self.assertEqual(c.pool_file(PINS, name), 'tasks-pool-h2.json')
            self.assertEqual([s['tasks'] for s in c.plan_servers(PINS, name)], [32])

    def test_each_plan_is_its_first_32_plan(self):
        for name, base in self.BASE.items():
            mine, theirs = c.plan_servers(PINS, name)[0], c.plan_servers(PINS, base)[0]
            masked = ('label', 'agent_cap_seconds', 'agent_min_seconds', c.ASYNC_KEY)
            self.assertEqual({k: v for k, v in mine.items() if k not in masked},
                             {k: v for k, v in theirs.items() if k not in masked}, name)
            self.assertEqual(plan_diff(name, base), sorted(
                ['authorization', 'expected_seconds', 'servers', 'tasks', 'time'] + (
                    [c.ASYNC_KEY] if name in H2RUNTIME else [])),
                name)
            if mine['kind'] == 'runtime':
                args = c.launch_arguments(PINS, mine, REWRITER)
                self.assertEqual([a for a in args if a != c.ASYNC_SWITCH], c.launch_arguments(PINS, theirs, REWRITER))
                self.assertEqual(c.ASYNC_SWITCH in args, name.endswith('a'), name)
            else:
                self.assertEqual(c.stock_argv(PINS, 8000, mine), c.stock_argv(PINS, 8000, theirs), name)
                self.assertEqual(c.model_override_text(PINS, mine, 24001), c.model_override_text(PINS, theirs, 24001))
            self.assertTrue(c.validate_plan(PINS, name)['fits'])
            self.assertIsNone(c.plan_blocked(PINS, name), name)
            self.assertEqual(c.cfg(PINS)['plans'][name]['partition'], 'ghx4')

    def test_the_sync_twins(self):
        for a in H2RUNTIME[0::2]:
            s = a[:-1] + 's'
            self.assertEqual(plan_diff(a, s), ['async_scheduling', 'authorization', 'servers'], a)
            sa, ss = c.plan_servers(PINS, a)[0], c.plan_servers(PINS, s)[0]
            self.assertEqual((sa[c.ASYNC_KEY], ss[c.ASYNC_KEY]), (True, False))
            self.assertEqual({k: v for k, v in sa.items() if k not in ('label', c.ASYNC_KEY)},
                             {k: v for k, v in ss.items() if k not in ('label', c.ASYNC_KEY)})
            la, ls = c.launch_arguments(PINS, sa, REWRITER), c.launch_arguments(PINS, ss, REWRITER)
            self.assertEqual([x for x in la if x != c.ASYNC_SWITCH], ls)
            self.assertEqual(c.model_override_text(PINS, sa, 24001), c.model_override_text(PINS, ss, 24001))
            pa, ps = c.pool_spec(PINS, sa, 24001, 1800000000)[1], c.pool_spec(PINS, ss, 24001, 1800000000)[1]
            pa.pop('label'), ps.pop('label')
            self.assertEqual(pa, ps)
            self.assertEqual(c.plan_fit(PINS, a), c.plan_fit(PINS, s))

    def test_the_time_plans(self):
        want = dict(fullh2a=('01:06:00', 2663, 2220, 1584), recenth2a=('01:28:00', 3626, 3540, 2380),
                    obsh2a=('01:27:00', 3605, 3480, 2340), goalh2a=('01:34:00', 4249, 3900, 2614),
                    stockh2t=('01:01:00', 2449, 2400, 1679), stockkeeph2t=('01:15:00', 3042, 3240, 2182))
        for name in H2:
            plan = c.cfg(PINS)['plans'][name]
            (s,) = plan['servers']
            key = name if name in want else name[:-1] + 'a'
            self.assertEqual((plan['time'], plan['expected_seconds'], s['agent_cap_seconds'], s['agent_min_seconds'],
                              plan['eval_wait_seconds']), want[key] + (600,), name)
            self.assertLessEqual(c.plan_cost(PINS, name)['limit_over_expected'], c.MAX_LIMIT_OVER_EXPECTED)



def thinking_row(turn, history, local, rebuild, generated=3, **extra):
    """A runtime row's stage_b and thinking record (keep-rebuild-1116 tests)."""
    value = dict(reasoning_tokens=1, think_end_tokens=1, content_tokens=generated - 2, closed=True, reasoning_chars=1,
                 content_chars=1, history=history, **extra)
    return dict(request_id='r%d' % turn, turn=turn, stage_b=dict(local_turn=local, rebuild=rebuild, prompt_end=10,
                                                                 computed_end=10 + generated - 1, computed_start=0),
                thinking=value)


class FunctionalJobs1116(unittest.TestCase):
    """2026-11-16 (the user's decision): the functional verification jobs short, on ghx4-interactive: each batch plan's
    *i variant (the same servers), one job per OFF / ON pair (two servers, the ON one's own gdn_inplace_crossing),
    --time the harness's worst case rounded up to the minute, the scripted servers' cap 900 s (1,080 s at K = 24)."""

    def test_the_variants_and_the_times(self):
        cfg = c.cfg(PINS)['plans']
        for name in ['keepedge', 'keepedgesync'] + [n for n in FUNC if not n.endswith('i')]:
            plan, variant = cfg[name], cfg[name + 'i']
            self.assertEqual((variant['variant_of'], variant['partition'], variant['role'], variant['servers'],
                              variant['time']), (name, 'ghx4-interactive', 'interactive', plan['servers'], plan['time']),
                             name)
            fit = c.plan_fit(PINS, name)
            worst = max(fit['worst_case_seconds'], fit['with_eval_wait_seconds'])
            self.assertEqual(c.time_seconds(plan['time']), -(-worst // 60) * 60, name)  # the tightest whole minute
            self.assertEqual(c.plan_cost(PINS, name + 'i')['billing_factor'], 2)
            caps = {s['agent_cap_seconds'] for s in plan['servers']}
            self.assertEqual(caps, {720} if name == 'sess24r' else {1080} if c.plan_running(PINS, name) == 24
                             else {900}, name)  # (11-17: sess24r's 720 s, sess24's scripted run plus readmissions)

    def test_the_k24_smoke_and_the_server_overlap(self):
        """2026-11-17: sess24r(i) = sess24's servers with --release-idle-pages and --admission-overlap 24 (a server's
        own admission_overlap over its policy's 16); goal64ak24 runs 24 too; sess24 (the record of 3613537) stays."""
        (mine,), (base,) = c.plan_servers(PINS, 'sess24r'), c.plan_servers(PINS, 'sess24')
        args, old = c.launch_arguments(PINS, mine, REWRITER), c.launch_arguments(PINS, base, REWRITER)
        self.assertEqual((args[args.index('--admission-overlap') + 1], old[old.index('--admission-overlap') + 1]),
                         ('24', '16'))
        self.assertIn(c.RELEASE_SWITCH, args)
        self.assertNotIn(c.RELEASE_SWITCH, old)
        self.assertEqual(c.server_admission(PINS, mine)['admission_overlap'], c.overlap_record(24))
        strip = lambda a: [x for i, x in enumerate(a) if x != c.RELEASE_SWITCH and not (  # noqa: E731
            x == '--admission-overlap' or (i and a[i - 1] == '--admission-overlap'))]
        self.assertEqual(strip(args), strip(old))
        self.assertEqual(c.plan_servers(PINS, 'goal64ak24')[0]['admission'], ['admission_overlap=24'])
        self.assertEqual(c.plan_servers(PINS, 'goal64a')[0]['admission'], ['admission_overlap=16'])
        for bad, text in ((1, 'admission_overlap'), (33, 'admission_overlap'), ('24', 'admission_overlap')):
            pins = copy.deepcopy(PINS)
            pins['config']['plans']['sess24r']['servers'][0]['admission_overlap'] = bad
            with self.assertRaisesRegex(RuntimeError, text):
                c.validate_plan(pins, 'sess24r')
        pins = copy.deepcopy(PINS)  # a policy without the overlap
        pins['config']['plans']['gdnobs']['servers'][0]['admission_overlap'] = 16
        with self.assertRaisesRegex(RuntimeError, 'admission_overlap'):
            c.validate_plan(pins, 'gdnobs')

    def test_the_pairs(self):
        for name in PAIRS:
            off, on = c.plan_servers(PINS, name)
            self.assertEqual((c.inplace_on(PINS, off), c.inplace_on(PINS, on)), (False, True), name)
            self.assertEqual({k: v for k, v in off.items() if k not in ('label', 'gdn_inplace_crossing', 'settings')},
                             {k: v for k, v in on.items() if k not in ('label', 'gdn_inplace_crossing', 'settings')},
                             name)
            self.assertEqual(on['label'], off['label'] + 'on')
            args_off, args_on = (c.launch_arguments(PINS, s, REWRITER) for s in (off, on))
            self.assertEqual([a for a in args_on if a != c.INPLACE_SWITCH], args_off, name)
            K = c.server_running(PINS, on)
            expect = c.server_expectation(PINS, on, dict(status='x'), 'm', ['t'])
            self.assertEqual((expect['layout']['record']['mamba_slots'], expect[c.INPLACE_KEY]),
                             (K + 1, PINS['runtime_features'][c.INPLACE_KEY]), name)
        bad = copy.deepcopy(PINS)
        bad['config']['plans']['gdnobsp']['servers'][1]['gdn_inplace_crossing'] = 1
        with self.assertRaisesRegex(RuntimeError, 'true only'):
            c.validate_plan(bad, 'gdnobsp')
        bad = copy.deepcopy(PINS)
        bad['runtime_features'].pop(c.INPLACE_KEY)
        with self.assertRaisesRegex(RuntimeError, 'gdn-inplace-crossing'):
            c.validate_plan(bad, 'gdnobsp')


class GoalRollback1116(unittest.TestCase):
    """2026-11-16 (phase B; goal-rollback-1116 f9ada25 / a39dd08 through its fixture only, REVIEW.md finding 1): the
    feature from query_selection._rollback_carry (absent on a2f9e36: no expectation key, every check as before), the
    plans keepedgegoal / keepedgegoalov (blocked until the pin has it), a carried rollback never parks and records no
    Goal timing (strict), the goal-carry check of the rows' carry records, the decision histogram apart."""
    FIXTURE = json.loads((ROOT / 'fixtures' / 'goal-rollback-43ff7b0.json').read_text())

    def pins(self):
        pins = copy.deepcopy(PINS)
        pins['runtime_features'][c.GOAL_ROLLBACK_KEY] = self.FIXTURE['feature']
        return pins

    def test_the_feature_the_plans_and_the_block(self):
        # 7a8c36e carries it (the same feature as 43ff7b0's fixture)
        self.assertEqual(PINS['runtime_features'][c.GOAL_ROLLBACK_KEY], self.FIXTURE['feature'])
        self.assertIsNone(c.goal_rollback_feature(ROOT.parents[2] / 'worktrees' / 'no-such-tree'))
        without = copy.deepcopy(PINS)
        without['runtime_features'].pop(c.GOAL_ROLLBACK_KEY)
        tree = ROOT.parents[2] / 'worktrees' / 'goal-rollback-1116'
        if (tree / c.GOAL_ROLLBACK_MODULE).is_file():
            self.assertEqual(c.goal_rollback_feature(tree), self.FIXTURE['feature'])
        self.assertEqual(c.goal_policies(PINS), ('goal_query', 'goal_query_markers'))
        pins = self.pins()
        for name in GOALEDGE:
            self.assertIn('goal-rollback-1116', c.plan_blocked(without, name))
            self.assertIsNone(c.plan_blocked(pins, name))
            for srv in c.plan_servers(pins, name):  # (keepedgegoalp(i): both in one job, --goal-overlap off, on)
                self.assertEqual((srv['policy'], srv['client'], c.goal_overlap_on(pins, srv)),
                                 ('goal_query', 'edge_script', srv['label'] == 'keepedgegoalov'))
                expect = c.server_expectation(pins, srv, dict(status='x'), 'm', ['t'])
                self.assertEqual(expect[c.GOAL_ROLLBACK_KEY], dict(self.FIXTURE['feature'], strict=True,
                                                                   policies=['goal_query', 'goal_query_markers']))
        for name in PLANS:  # no other plan is blocked by it; without it no expectation carries the key
            if name not in GOALEDGE:
                self.assertIsNone(c.goal_rollback_block(without, name), name)
            for srv in c.plan_servers(without, name):
                if srv['kind'] == 'runtime':
                    self.assertNotIn(c.GOAL_ROLLBACK_KEY, c.server_expectation(without, srv, dict(status='x'), 'm',
                                                                               ['t']))
        (obs,) = c.plan_servers(pins, 'gdnobs')  # with the feature: every runtime server, strict with the client
        self.assertIs(c.server_expectation(pins, obs, dict(status='x'), 'm', ['t'])[c.GOAL_ROLLBACK_KEY]['strict'], True)
        def obs(p):
            srv = p['config']['plans']['keepedgegoal']['servers'][0]
            for key in ('goal_rewriter_engine', 'goal_rewriter_guided_backend', 'goal_rewriter_vllm_kv_gib',
                        'goal_rewriter_batch_invariant'):
                srv.pop(key, None)
            srv['policy'] = 'observation_query'
        for plan, change, text in (
                ('keepedgegoal', lambda p: p['config']['plans']['keepedgegoal'].update(goal_rollback=1), 'true only'),
                ('keepedgegoal', obs, 'Goal policy'),
                ('keepedge', lambda p: p['config']['plans']['keepedge'].update(goal_rollback=True), 'Goal policy')):
            bad = self.pins()
            change(bad)
            with self.assertRaisesRegex(RuntimeError, text):
                c.validate_plan(bad, plan)

    @staticmethod
    def carry_rows(tmp, carries, policy='goal_query', history=True):
        """One session's rows: request 0 fresh, then per item a request whose carry is the item (None: an append)."""
        out = Path(tmp)
        rows = []
        for t, carry in enumerate([None] + list(carries)):
            # 43ff7b0's carried row: KEEP / KEEP, no rewriter, no cache hit, no forward, cold_start, the from_turn
            # row's Goal (payload_sha256, goal_hash)
            origin = carry['from_turn'] if isinstance(carry, dict) and carry.get('carried') else t
            goal = dict(gate_decision='KEEP' if carry else 'PATCH', applied_decision='KEEP' if carry else 'PATCH',
                        slm_attempted=False, slm_succeeded=False, fallback_reason=None, cache_hit=False,
                        forward=False, payload_sha256='p%s' % origin, **({'carry': carry} if carry is not None else {}))
            row = dict(request_id='r%d' % t, turn=t, session_index=0, policy=policy,
                       query=dict(goal=goal, goal_hash='h%s' % origin, selection=dict(
                           reason='cold_start' if carry is not None or t == 0 else None)),
                       stage_b=dict(local_turn=0 if carry is not None or t == 0 else t, rebuild=carry is not None))
            if history:
                row['thinking'] = dict(history='rollback' if carry is not None else ('fresh' if t == 0 else 'append'))
            rows.append(row)
            sc.native_path(out, row['request_id']).write_text(json.dumps(dict(query=dict(body=dict(goal=goal)))))
        return rows

    def test_the_carry_records(self):
        feature = dict(self.FIXTURE['feature'], policies=['goal_query', 'goal_query_markers'], strict=True)
        expect = {c.GOAL_ROLLBACK_KEY: feature}
        yes = lambda frm, gt: dict(carried=True, reason=None, from_turn=frm, goal_turn=gt)  # noqa: E731
        with tempfile.TemporaryDirectory() as tmp:
            # a rollback of request 1's reply carries from 1; a second in a row keeps that source
            rows = self.carry_rows(tmp, [None, yes(1, 2), yes(1, 2)])
            value = swe_checks.check_goal_carry(tmp, rows, 'goal_query', expect)
            self.assertEqual((value['carried'], value['second_in_a_row'], value['not_carried']), (2, 1, 0))
        bad = [
            ([None, yes(0, 2)], 'goal_query', 'its source is turn 1'),  # the wrong source
            ([None, yes(1, 2), yes(1, 3)], 'goal_query', 'its source is turn 1'),  # the chain's Goal turn changed
            ([None, dict(yes(1, 2), extra=1)], 'goal_query', 'carry keys'),
            ([None, dict(carried=False, reason=None, from_turn=None, goal_turn=None)], 'goal_query', 'a scripted'),
            ([None, yes(1, 2)], 'observation_query', 'records a carry'),
            ([None, dict(yes(1, 2), goal_turn=0)], 'goal_query', 'carried record')]
        for carries, policy, text in bad:
            with tempfile.TemporaryDirectory() as tmp, self.subTest(text=text):
                with self.assertRaisesRegex(sc.CheckFailed, text):
                    swe_checks.check_goal_carry(tmp, self.carry_rows(tmp, carries, policy), policy, expect)
        with tempfile.TemporaryDirectory() as tmp:  # not strict: a refused carry with its reason is a plain rebuild
            loose = {c.GOAL_ROLLBACK_KEY: dict(feature, strict=False)}
            rows = self.carry_rows(tmp, [None, dict(carried=False, reason='messages', from_turn=None, goal_turn=None)])
            self.assertEqual(swe_checks.check_goal_carry(tmp, rows, 'goal_query', loose)['reasons'], {'messages': 1})
            rows[2]['stage_b']['rebuild'] = False  # a carry on a request that is no rebuild
            with self.assertRaisesRegex(sc.CheckFailed, 'no rollback'):
                swe_checks.check_goal_carry(tmp, rows, 'goal_query', loose)
        with tempfile.TemporaryDirectory() as tmp:  # the native record differs from the row
            rows = self.carry_rows(tmp, [None, yes(1, 2)])
            sc.native_path(tmp, 'r2').write_text(json.dumps(dict(query=dict(body=dict(goal={})))))
            with self.assertRaisesRegex(sc.CheckFailed, 'native query.body.goal.carry'):
                swe_checks.check_goal_carry(tmp, rows, 'goal_query', expect)
        # 43ff7b0's shape: every deviation of a carried row is refused
        for change, text in ((lambda r: r['query']['goal'].update(cache_hit=True), 'carried Goal record'),
                             (lambda r: r['query']['goal'].update(forward=True), 'carried Goal record'),
                             (lambda r: r['query']['goal'].update(payload_sha256='p0'), 'not the Goal of turn 1'),
                             (lambda r: r['query'].update(goal_hash='h0'), 'not the Goal of turn 1'),
                             (lambda r: r['query']['selection'].update(reason='below_switch_threshold'), 'cold_start')):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(text=text):
                rows = self.carry_rows(tmp, [None, yes(1, 2)])
                change(rows[2])
                native = dict(query=dict(body=dict(goal=rows[2]['query']['goal'])))
                sc.native_path(tmp, 'r2').write_text(json.dumps(native))
                with self.assertRaisesRegex(sc.CheckFailed, text):
                    swe_checks.check_goal_carry(tmp, rows, 'goal_query', expect)
        for field in sc.GOAL_CARRY_ABSENT + ('frame',):  # the six native fields and the goal_forward frame
            with tempfile.TemporaryDirectory() as tmp, self.subTest(field=field):
                rows = self.carry_rows(tmp, [None, yes(1, 2)])
                native = dict(query=dict(body=dict(goal=rows[2]['query']['goal'])))
                if field == 'frame':
                    native['worker_frames'] = [dict(operation=sc.GOAL_FORWARD_OP)]
                else:
                    native[field] = 0.0
                sc.native_path(tmp, 'r2').write_text(json.dumps(native))
                with self.assertRaisesRegex(sc.CheckFailed, 'native record of a carried rollback'):
                    swe_checks.check_goal_carry(tmp, rows, 'goal_query', expect)
        decision = dict(gate_decision='KEEP', applied_decision='KEEP', triggers=['rollback'], changed_fields=[],
                        slm_attempted=False, slm_succeeded=False, fallback_reason=None)
        for item, ok in ((dict(decision=decision, rewrites=[]), True),
                         (dict(decision=dict(decision, triggers=[]), rewrites=[]), False),
                         (dict(decision=decision, rewrites=[dict(request='x')]), False), (None, False)):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(item=item):
                rows = self.carry_rows(tmp, [None, yes(1, 2)])
                lines = [dict(request_id='r1', carry=None)] + ([dict(item, request_id='r2', carry=yes(1, 2))]
                                                              if item else [])
                (Path(tmp) / 'goals.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in lines))
                if ok:
                    self.assertEqual(swe_checks.check_goal_carry(tmp, rows, 'goal_query', expect)['carried'], 1)
                else:
                    with self.assertRaisesRegex(sc.CheckFailed, 'goals.jsonl'):
                        swe_checks.check_goal_carry(tmp, rows, 'goal_query', expect)

    def test_carried_and_the_histogram(self):
        row = dict(query=dict(goal=dict(carry=dict(carried=True))))
        self.assertTrue(sc.carried(row, {c.GOAL_ROLLBACK_KEY: {}}) is False)  # an empty feature: off
        self.assertTrue(sc.carried(row, {c.GOAL_ROLLBACK_KEY: dict(policies=['goal_query'])}))
        self.assertFalse(sc.carried(row, {}))  # without the feature: a row as before (it parks or fails)
        self.assertFalse(sc.carried(dict(query=dict(goal=dict(carry=dict(carried=False)))),
                                    {c.GOAL_ROLLBACK_KEY: dict(policies=['goal_query'])}))
        records = [dict(session=0, turn=t, gate_decision=g, slm_attempted=False, **extra)
                   for t, (g, extra) in enumerate([('INITIALIZE', {}), ('REWRITE', {}), ('KEEP', dict(goal_carried=True)),
                                                   ('PATCH', dict(goal_carried=False))])]
        goal = swe_checks.summarize(records, {})['goal']
        self.assertEqual((goal['decisions'], goal['rollback_carried'], goal['rewrite_share']),
                         ({'INITIALIZE': 1, 'REWRITE': 1, 'PATCH': 1}, 1, 0.3333))
        plain = swe_checks.summarize([dict(r) for r in records[:2]], {})['goal']
        self.assertNotIn('rollback_carried', plain)  # before the feature: the same keys as before


class KeepRebuild1116(unittest.TestCase):
    """2026-11-16 (phase A of integrate-1116; keep-rebuild-1116 51af079 through fixtures only): the rollback history and
    the two row keys, the rollback's kept_tokens rule, the close's rebuilds / poisoned / holding (release gate 3), the
    keep render in the join (release gate 1) and its pre-fix note, the steady state, the scripted edge-case client and
    plan keepedge (blocked on 4ebff97), the max_tokens plan server key; every earlier plan unchanged."""
    EXPECT = dict(thinking=dict(header_text='<|im_start|>assistant\n<think>\n', think_end='</think>'))

    def test_old_rows_keep_their_detail(self):
        rows = [thinking_row(0, 'fresh', 0, False), thinking_row(1, 'append', 1, False),
                thinking_row(2, 'rebuild', 0, True)]
        info = swe_checks.check_thinking(rows, self.EXPECT)
        self.assertEqual(set(info), {'thinking', 'rows', 'histories', 'unclosed', 'reasoning_tokens',
                                     'mean_reasoning_tokens'})

    def test_rollback_and_rebuild_rows(self):
        rows = [thinking_row(0, 'fresh', 0, False), thinking_row(1, 'rollback', 0, True, kept_tokens=9,
                                                                 reencoded_replies=0),
                thinking_row(2, 'append', 1, False),
                thinking_row(3, 'rebuild', 0, True, kept_tokens=0, reencoded_replies=1)]
        info = swe_checks.check_thinking(rows, dict(self.EXPECT, keep_rebuild=dict(history='rollback')))
        self.assertEqual((info['rollbacks'], info['rebuilds'], info['kept_tokens'], info['reencoded_replies']),
                         (1, 1, 9, 1))
        bad = [
            [thinking_row(0, 'fresh', 0, False), thinking_row(1, 'rollback', 0, True)],  # no keys
            [thinking_row(0, 'fresh', 0, False), thinking_row(1, 'rollback', 0, True, kept_tokens=9,
                                                              reencoded_replies=2)],  # a re-encoded reply
            [thinking_row(0, 'fresh', 0, False), thinking_row(1, 'rebuild', 0, True, kept_tokens=4,
                                                              reencoded_replies=1)],  # a rebuild keeps nothing
            [thinking_row(0, 'fresh', 0, False, kept_tokens=0, reencoded_replies=0)],  # never on fresh / append
            [thinking_row(0, 'fresh', 0, False), thinking_row(1, 'rollback', 1, False, kept_tokens=9,
                                                              reencoded_replies=0)],  # not re-prefilled
            [thinking_row(0, 'rollback', 0, True, kept_tokens=9, reencoded_replies=0)],  # a first request
            [thinking_row(0, 'fresh', 0, False), thinking_row(1, 'rebuild', 0, True),  # with and without the keys
             thinking_row(2, 'rebuild', 0, True, kept_tokens=0, reencoded_replies=1)]]
        for case in bad:
            with self.assertRaises(Exception, msg=case):
                swe_checks.check_thinking(case, self.EXPECT)
        with self.assertRaises(Exception):  # the feature requires the keys on every rebuild row
            swe_checks.check_thinking([thinking_row(0, 'fresh', 0, False), thinking_row(1, 'rebuild', 0, True)],
                                      dict(self.EXPECT, keep_rebuild=dict(history='rollback')))

    def test_the_kept_tokens_rule(self):
        previous = dict(prompt_token_ids=list(range(20)))
        row = thinking_row(1, 'rollback', 0, True, kept_tokens=15, reencoded_replies=0)
        native = dict(prompt_token_ids=list(range(15)) + [90, 91, 92] + [1, 2, 3, 4, 5], stable_end=18, rebuild=True,
                      local_turn=0)
        self.assertEqual(swe_checks.check_rollback(row, native, previous, 5)['kept_tokens'], 15)
        for change in (dict(kept_tokens=14), dict(reencoded_replies=1)):
            wrong = thinking_row(1, 'rollback', 0, True, **dict(dict(kept_tokens=15, reencoded_replies=0), **change))
            with self.assertRaises(Exception):
                swe_checks.check_rollback(wrong, native, previous, 5)
        with self.assertRaises(Exception):  # the prompt does not start with the previous prompt
            swe_checks.check_rollback(row, dict(native, prompt_token_ids=[7] + native['prompt_token_ids'][1:]),
                                      previous, 5)

    def test_last_turn_holding(self):
        rows = [dict(stage_b=dict(retain=[0, 1], restore=[2], candidate_end=48, sealed_end=96, computed_end=100))]
        self.assertEqual(swe_checks.last_turn_holding(rows, 16), (3 + 3, True))
        self.assertIsNone(swe_checks.last_turn_holding([], 16))

    def test_steady_state(self):
        import join_swe32
        records = [dict(start=0, end=10), dict(start=1, end=6), dict(start=6, end=12)]
        steady, tail = join_swe32.steady_state(records, [(0.5, 7), (2, 10), (5, 20), (11, 30)], 2)
        self.assertEqual((steady['seconds'], steady['tokens'], steady['calls'], steady['tok_s']), (9, 30, 2, 3.33))
        self.assertEqual((tail['seconds'], tail['tokens']), (2, 30))
        self.assertEqual(join_swe32.steady_state(records, [], 5), (None, None))

    def test_keep_render_note_on_a_prefix_runtime(self):
        import join_swe32
        value = dict(instances=[dict(label='a', instance_id='i', checks=[])], checks=[])
        join_swe32.keep_render_apply(value, ['a'], {'a/i': [dict(messages=[], prompt=[])]},
                                     dict(status='NOT_AVAILABLE'))
        self.assertEqual((value['keep_render'], value['checks'], value['instances'][0]['checks']),
                         ({'a': join_swe32.KEEP_NOT_AVAILABLE}, [], []))
        self.assertEqual(join_swe32.KEEP_NOT_AVAILABLE, 'not available (pre-fix runtime)')
        self.assertEqual(join_swe32.keep_render_sessions(ROOT / 'no-such-tool.py', None, None, '', None, {})['status'],
                         'NOT_AVAILABLE')

    def test_the_feature_and_the_block(self):
        fixture = json.loads((ROOT / 'fixtures' / 'keep-rebuild-51af079.json').read_text())
        # 2026-11-16 (phase B): the pin a2f9e36 carries keep-rebuild-1116 (51af079's feature, the same tool)
        self.assertEqual(PINS['runtime_features'][c.KEEP_REBUILD_KEY], fixture['feature'])
        without = copy.deepcopy(PINS)
        without['runtime_features'].pop(c.KEEP_REBUILD_KEY)
        tree = ROOT.parents[2] / 'worktrees' / 'keep-rebuild-1116'
        if (tree / 'tools' / 'keep_render_check.py').is_file() and subprocess.run(
                ['git', '-C', str(tree), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() == \
                fixture['commit']:
            self.assertEqual(c.keep_rebuild_feature(tree), fixture['feature'])
        self.assertEqual(fixture['feature']['tool_sha256'], hashlib.sha256(
            (ROOT / 'fixtures' / 'keep_render_check-51af079.py').read_bytes()).hexdigest())
        for name in CLIENT_PLANS:  # (11-16 phase B: and the gdn pairs, the same scripted client)
            self.assertIn('keep-rebuild-1116', c.keep_rebuild_block(without, name))
            self.assertIsNone(c.keep_rebuild_block(PINS, name))
            for s in c.plan_servers(PINS, name):
                expect = c.server_expectation(PINS, s, dict(status='x'), 'm', ['t'])
                self.assertEqual((expect[c.KEEP_REBUILD_KEY], expect['client'], expect['max_tokens']),
                                 (fixture['feature'], 'edge_script', 1536))
        for name in PLAN_ORDER:  # no other plan is blocked by it
            if name not in CLIENT_PLANS:
                self.assertIsNone(c.keep_rebuild_block(without, name), name)

    def test_the_edge_plans(self):
        a, s_ = c.plan_servers(PINS, 'keepedge')[0], c.plan_servers(PINS, 'keepedgesync')[0]
        self.assertEqual((a['client'], a['tasks'], c.plan_running(PINS, 'keepedge'), c.async_scheduling_on(PINS, a),
                          c.async_scheduling_on(PINS, s_)), ('edge_script', 16, 16, True, False))
        text = c.model_override_text(PINS, a, 24001)
        self.assertIn('model_class: swe32_edge_script.EdgeScriptModel', text)
        self.assertIn('    max_tokens: 1536\n', text)
        edge = c.cfg(PINS)['agent']['edge_script']['agent_module']
        self.assertEqual(edge['sha256'], hashlib.sha256((ROOT / 'tbench' / 'agent_ext' / edge['file']).read_bytes())
                         .hexdigest())
        bad = copy.deepcopy(PINS)
        bad['config']['plans']['keepedge']['servers'][0]['client'] = 'agent'
        with self.assertRaises(Exception):
            c.validate_plan(bad, 'keepedge')
        for name in GDN + FUNC:  # the gdn plans and the functional jobs (11-16): keepedge's client and bound
            for srv in c.plan_servers(PINS, name):
                self.assertEqual(srv['client'], 'edge_script', name)
                self.assertIn('    max_tokens: 1536\n', c.model_override_text(PINS, srv, 24001))
        for name in PLAN_ORDER:  # every other runtime server: 4096 and litellm_textbased, as before
            for srv in c.plan_servers(PINS, name):
                if srv['kind'] == 'runtime' and name not in CLIENT_PLANS:
                    text = c.model_override_text(PINS, srv, 24001)
                    self.assertIn('    max_tokens: 4096\n', text)
                    self.assertIn('model_class: litellm_textbased', text)

    def test_the_script(self):
        sys.path.insert(0, str(ROOT / 'tbench' / 'agent_ext'))
        import swe32_edge_script as edge
        sent = []

        def post(body):
            k = len(sent)
            sent.append(body)
            cut = k == 6  # the first long-reasoning request is cut inside its reasoning
            return dict(id='c%d' % k, usage=dict(prompt_tokens=10, completion_tokens=1536 if cut else 20),
                        choices=[dict(finish_reason='length' if cut else 'stop',
                                      message=dict(content='' if cut else 'C%d' % k, reasoning_content='R%d' % k))])
        settings = edge.settings_from('openai/selective-prefix', dict(max_tokens=1536, api_base='http://x/v1',
                                                                      extra_body=dict(chat_template_kwargs=dict(
                                                                          enable_thinking=True))),
                                      observation_lines=3, long_items=2)
        record = edge.run_session(post, [dict(role='system', content='S'), dict(role='user', content='T')], settings)
        self.assertIsNone(record['error'])
        self.assertEqual([c_['expected_history'] for c_ in record['calls']],
                         ['fresh', 'rollback', 'append', 'append', 'rollback', 'rollback', 'append', 'rollback',
                          'append', 'rebuild', 'append'])
        self.assertEqual(record['coverage'], dict(first_reply_dropped=1, rollback_after_closed=1, double_rollback=1,
                                                  rollback_after_cut=1, cut_attempts=1, resend=1))
        self.assertEqual(sent[9]['messages'], sent[8]['messages'])  # the resend repeats the request
        self.assertEqual(sent[1]['messages'], sent[0]['messages'] + [dict(role='user', content=edge.FORMAT_ERROR)])
        self.assertEqual(sent[3]['messages'][-2], dict(role='assistant', content='C2', reasoning_content='R2'))
        self.assertTrue(all(b['max_tokens'] == 1536 and b['temperature'] == 0.0 and b['model'] == 'selective-prefix'
                            and b['chat_template_kwargs'] == dict(enable_thinking=True) for b in sent))
        import join_swe32
        calls = join_swe32.script_calls(record)
        self.assertEqual([x['kind'] for x in calls].count('format_error'), 4)
        self.assertEqual([x['kind'] for x in calls].count('resend'), 1)
class Ledger1114(unittest.TestCase):
    """2026-11-14 (perf-swe32-ledger): ledger_query of ledger-int-1112 51b4085, the release plug-in of the query hook:
    accepted from describe() (its row: kind query, the observation vector, hooked, under hook, the XML protocol only;
    the POLICIES lines keep the eight built-in policies), the plan key ledger_cap_percent (--ledger-cap-percent N,
    default 50 without a flag, 1-100, ledger_query only), the plans ledgert / ledger / ledger64 / ledgerti, the
    per-request checks of the hook record (swe_checks.check_hook against the sealed plan's own records), the
    selection parity's recorded fill, ledger_verify's history, coverage, streaks and record counts."""

    @staticmethod
    def modules():
        import fake_server
        return fake_server.ledger_modules()

    def test_the_row_the_feature_and_the_builtin_lists(self):
        named = {r['name']: r for r in PINS['runtime_features'][c.REGISTRY_KEY]}
        row = named['ledger_query']
        self.assertEqual((row['kind'], row['query_vector'], row['hooked'], row['evidence_key'], row['protocols'],
                          row['budget'], row['selector_v2'], row['builtin'], row['goal_engine'], row['pin_options']),
                         ('query', 'observation', True, 'hook', ['mswea_xml_v1'], 'fixed', True, False, False,
                          ['ledger_cap_percent']))
        self.assertEqual(row['pin'], c.LEDGER_PIN)
        self.assertEqual(row['rewriter_unused'], c.GOAL_REWRITER_UNUSED_LEDGER)
        self.assertEqual(c.registry_disagrees(row, 'ledger_query'), [])
        self.assertIsNone(c.policy_problem(PINS, 'ledger_query'))
        self.assertEqual(c.hooked_policies(PINS), HOOKED)  # (a2f9e36: and the marker plug-ins, no plan runs them)
        self.assertEqual([n for n, r in named.items() if r['builtin']], BUILTIN)  # the eight built-in policies
        self.assertTrue(all(r['hooked'] is False and r['protocols'] is None for n, r in named.items() if r['builtin']))
        self.assertEqual(PINS['runtime_features']['ledger'], dict(
            policy='ledger_query', serve_arg='--ledger-cap-percent', config_key='ledger_cap_percent',
            module='source-overlay/page_policies/ledger_query.py', limits=[1, 100], pin=c.LEDGER_PIN,
            protocols=['mswea_xml_v1'], evidence_key='hook'))
        import smoke_checks as sc
        self.assertEqual((sc.LEDGER_PIN, sc.GOAL_REWRITER_UNUSED_LEDGER, sc.HOOKED),
                         (c.LEDGER_PIN, c.GOAL_REWRITER_UNUSED_LEDGER, (c.LEDGER,)))
        self.assertIn('ledger_query', sc.SELECTIVE)
        tree = ROOT.parents[2] / 'worktrees' / 'integrate-1116'  # the pinned commit's worktree, when at the pin
        if (tree / 'serve.py').is_file() and subprocess.run(['git', '-C', str(tree), 'rev-parse', 'HEAD'],
                                                            capture_output=True, text=True).stdout.strip() == \
                PINS['commit']:  # the POLICIES lines of serve.py / config.py: still the eight built-ins
            serve = (tree / 'serve.py').read_text(encoding='utf-8')
            config = (tree / 'source-overlay/remote_ra_http/config.py').read_text(encoding='utf-8')
            self.assertEqual(list(c.serve_constant(serve, 'POLICIES')), BUILTIN)
            self.assertEqual(list(c.serve_constant(config, 'POLICIES')), BUILTIN)
            self.assertEqual(c.ledger_feature(tree, serve, named), PINS['runtime_features']['ledger'])
            for name in ('policy_registry.py', 'page_policies/__init__.py', 'page_policies/ledger_query.py',
                         'page_policies/marker_pages.py'):
                self.assertEqual((ROOT / 'fixtures' / 'ledger-a2f9e36' / name).read_bytes(),
                                 (tree / 'source-overlay' / name).read_bytes(), name)  # the fake's vendored copy

    def test_describe_adds_hooked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'source-overlay').mkdir()
            module = root / 'source-overlay' / 'policy_registry.py'
            rows = [dict(r) for r in REGISTRY_ROWS]
            for r in rows:
                r.pop('hooked')
            body = ('ROWS = %r\nclass P:\n    hooked = True\nclass Q:\n    pass\n'
                    'def describe():\n    return [dict(r) for r in ROWS]\n'
                    'def get(name):\n    return P() if name in %r else Q()\n' % (rows, HOOKED))
            module.write_text(body, encoding='utf-8')
            got = c.registry_rows(root)
            self.assertEqual(got, REGISTRY_ROWS)  # hooked from each policy's own attribute (False where it has none)
            module.write_text('ROWS = %r\ndef describe():\n    return [dict(r) for r in ROWS]\n' % (rows,), encoding='utf-8')
            self.assertTrue(all(r['hooked'] is False for r in c.registry_rows(root)))  # a registry without get()
            self.assertEqual(list(root.rglob('__pycache__')), [])

    def test_the_plans(self):
        cfg = c.cfg(PINS)['plans']
        want = dict(ledgert=('ledger262t', 'ghx4', '01:39:00', 3978, 600, 4200, 2100, True, None, 'validation',
                             (1.65, 1.105)),
                    ledger=('ledger262', 'ghx4', '01:18:00', 3138, 600, 2940, 1500, None, None, 'validation',
                            (1.3, 0.872)),
                    ledger64=('ledger64t', 'ghx4', '02:13:00', 5343, 1200, 5640, 3450, True, 64, 'validation',
                              (2.217, 1.484)),
                    ledgerti=('ledger262t', 'ghx4-interactive', '01:39:00', 3978, 600, 4200, 2100, True, None,
                              'interactive', (3.3, 2.21)))
        for name, (label, partition, time, expected, wait, cap, minimum, thinking, tasks, role, hours) in want.items():
            plan, (server,) = cfg[name], c.plan_servers(PINS, name)
            cost, fit = c.plan_cost(PINS, name), c.validate_plan(PINS, name)
            self.assertEqual((server['label'], server['policy'], plan['partition'], plan['time'], plan['expected_seconds'],
                              plan['eval_wait_seconds'], server['agent_cap_seconds'], server['agent_min_seconds'],
                              plan.get('thinking'), plan.get('tasks'), plan['role'],
                              (cost['gpu_hours_limit'], cost['gpu_hours_expected'])),
                             (label, 'ledger_query', partition, time, expected, wait, cap, minimum, thinking, tasks,
                              role, hours), name)
            self.assertLessEqual(cost['limit_over_expected'], c.MAX_LIMIT_OVER_EXPECTED, name)
            self.assertTrue(fit['fits'], name)
            self.assertEqual(c.memory_fit(PINS, name)['need_gib'], 105.29)
            self.assertNotIn(c.LEDGER_KEY, server)  # the sealed default 50, no flag
            self.assertNotIn(c.LEDGER_SWITCH, c.launch_arguments(PINS, server, REWRITER))
            self.assertEqual(c.ledger_cap(PINS, server), 50)
            self.assertIn('ESTIMATE', plan['about'])
            self.assertIn('NOT GPU-validated', cfg[name]['about'] if name != 'ledgerti' else cfg['ledgert']['about'])
            self.assertTrue(plan['authorization'].startswith('AUTHORIZE_SWE32_%s_%dTASKS_16SESSIONS_262K_KV25_THINKING_%s_'
                                                             'LEDGER_QUERY_' % (name.upper(), tasks or 32,
                                                                                'ON' if thinking else 'OFF')), name)
        # the baselines' servers with the policy ledger_query: obst's obs262t, ctx25's obs262, obs64's obs64t
        for name, (reference, index) in (('ledgert', ('obst', 0)), ('ledger', ('ctx25', 1)), ('ledger64', ('obs64', 0))):
            mine, theirs = c.plan_servers(PINS, name)[0], c.plan_servers(PINS, reference)[index]
            self.assertEqual({k: v for k, v in mine.items() if k not in ('label', 'policy', 'agent_cap_seconds',
                                                                         'agent_min_seconds')},
                             {k: v for k, v in theirs.items() if k not in ('label', 'policy', 'agent_cap_seconds',
                                                                           'agent_min_seconds')}, name)
            self.assertEqual(c.launch_arguments(PINS, mine, REWRITER), c.launch_arguments(PINS, theirs, REWRITER), name)
        self.assertEqual(cfg['ledgerti']['variant_of'], 'ledgert')
        self.assertEqual(c.plan_servers(PINS, 'ledgerti'), c.plan_servers(PINS, 'ledgert'))
        self.assertEqual([n for n in PLAN_ORDER if n not in PHASEB + P150][-len(LEDGER1114):],
                         LEDGER1114)

    def test_the_cap_key_and_its_refusals(self):
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['ledgert']['servers'][0][c.LEDGER_KEY] = 75
        (server,) = c.plan_servers(pins, 'ledgert')
        args = c.launch_arguments(pins, server, REWRITER)
        at = args.index(c.LEDGER_SWITCH)
        self.assertEqual(args[at - 2:at + 2], ['--active-token-budget', '8192', '--ledger-cap-percent', '75'])
        self.assertTrue(c.validate_plan(pins, 'ledgert')['fits'])
        gate = dict(status='GOAL_REWRITER_GATE_SKIPPED')
        expect = c.server_expectation(pins, server, gate, 'm', ['t0'])
        self.assertEqual(expect['ledger_pin'], dict(c.LEDGER_PIN, ledger_cap_percent=75))
        import smoke_checks as sc
        contract = sc.wanted_contract('ledger_query', expect)
        self.assertEqual((contract['ledger_query'], contract['goal_rewriter'], contract['active_token_budget']),
                         (dict(c.LEDGER_PIN, ledger_cap_percent=75), None, 8192))
        self.assertEqual(sc.wanted_rewriter_record('ledger_query', expect),
                         dict(used=False, reason=c.GOAL_REWRITER_UNUSED_LEDGER, requested_model=None))
        default = c.server_expectation(PINS, c.plan_servers(PINS, 'ledgert')[0], gate, 'm', ['t0'])
        self.assertEqual(default['ledger_pin'], c.LEDGER_PIN)
        self.assertNotIn('ledger_pin', c.server_expectation(PINS, c.plan_servers(PINS, 'obst')[0], gate, 'm', ['t0']))

        def refused(change, pattern, plan='ledgert'):
            bad = copy.deepcopy(PINS)
            change(bad)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(bad, plan)
        for value in (0, 101, True, 50.0, '50', None):
            refused(lambda p, v=value: p['config']['plans']['ledgert']['servers'][0].update(ledger_cap_percent=v),
                    'must be an integer 1-100')
        refused(lambda p: p['config']['plans']['obst']['servers'][0].update(ledger_cap_percent=50),
                'applies to ledger_query only', 'obst')
        refused(lambda p: p['config']['plans']['stock32t']['servers'][0].update(ledger_cap_percent=50),
                'on a stock server', 'stock32t')
        refused(lambda p: p['runtime_features'].update(ledger=None), 'does not serve ledger_query with')
        refused(lambda p: next(r for r in p['runtime_features'][c.REGISTRY_KEY] if r['name'] == 'ledger_query').update(
            protocols=['chat_text_v1']), 'serves the protocols')
        with self.assertRaisesRegex(RuntimeError, 'run ledger_query or name ledger_cap_percent'):
            c.check_features(dict(PINS['runtime_features'], ledger=None), PINS['config'])

    def synthetic_turn(self, thinking=False, cap=50):
        """One active ledger_query turn through the sealed plan and fill (the vendored 51b4085 code): a history of
        five replies (a read of a.py, an edit of a.py, a read of b.py, a re-read of a.py, a re-read of b.py, the
        newest) and their observations; returns (selection, native, contract, n, mandatory, capacity, plain)."""
        R, L = self.modules()
        P, initial = 16, 160
        spans = [('assistant', 160, 260, 'cat a.py'), ('user', 260, 900, None),
                 ('assistant', 900, 1000, "sed -i 's/x/y/' a.py"), ('user', 1000, 1100, None),
                 ('assistant', 1100, 1200, 'cat b.py'), ('user', 1200, 2400, None),
                 ('assistant', 2400, 2500, 'cat a.py'), ('user', 2500, 3000, None),
                 ('assistant', 3000, 3100, 'cat b.py')]
        messages = []
        for role, a, b, command in spans:
            if role == 'assistant':
                call = a + 30 if thinking else a + 5
                messages.append(R.HistoryMessage('assistant', a, b, call, 'THOUGHT: x.\n\n<mswea_bash_command>%s'
                                                 '</mswea_bash_command>' % command, 'r %d' % a if thinking else None))
            else:
                messages.append(R.HistoryMessage('user', a, b, None, '<returncode>0</returncode>\n<output>x</output>', None))
        n, mandatory, capacity = 3100 // P, initial // P, 60
        recent = list(range(n - 8, n))
        fixed = set(range(mandatory)) | set(recent)
        candidates = [p for p in range(n) if p not in fixed]
        scores = {p: round(0.9 - (p * 37 % 23) / 50.0, 8) for p in range(n)}
        order = sorted(candidates, key=lambda p: (-scores[p], p))
        room = capacity - mandatory - len(recent)
        plain = frozenset(fixed) | frozenset(order[:room])
        pin = dict(c.LEDGER_PIN, ledger_cap_percent=cap)
        context = R.FillContext(policy='ledger_query', num_pages=n, page_tokens=P, mandatory=mandatory,
                                recent=tuple(recent), candidates=tuple(candidates), capacity=capacity, local_turn=5,
                                turn=5, session_index=0, turn_ends=[], observation_spans=[], stable_end=3100, pin=pin,
                                messages=tuple(messages), state={})
        hook = L.LedgerQuery().query_hook(context)
        taken, rest, fill = R.hooked_fill(order, room, hook)
        selected = sorted(fixed | set(taken))
        record = json.loads(json.dumps(hook.record(frozenset(selected), plain, fill)))
        selection = dict(recent_pages=recent, retrieved_pages=list(taken), selected_pages=selected, hook=record)
        native = dict(stable_end=3100, query=dict(body=dict(selection=dict(scores={str(p): scores[p] for p in range(n)}),
                                                           ranking=list(range(mandatory)) + recent + list(taken) + rest)))
        return selection, native, {'ledger_query': pin}, n, mandatory, capacity, initial

    def check(self, turn, thinking=False):
        selection, native, contract, n, mandatory, capacity, initial = turn
        return swe_checks.check_hook('ledger_query', {}, native, selection, contract, n, mandatory, capacity, 16,
                                     initial, thinking=thinking)

    def test_check_hook_accepts_the_policys_own_records(self):
        for thinking, cap in ((False, 50), (True, 50), (False, 10), (False, 100)):
            turn = self.synthetic_turn(thinking, cap)
            info = self.check(turn, thinking)
            self.assertEqual(info['hook_cap_pages'], info['hook_room'] * cap // 100)
            self.assertEqual(info['hook_reasoning'] == 0, not thinking or info['hook_reasoning'] == 0)
            items = turn[0]['hook']['items']
            self.assertEqual(items[0][0], 'edit')
            if cap == 10:  # a small cap: the ledger skips the reads (whole or nothing for the cosine fill)
                self.assertTrue(info['hook_items']['skipped'] + info['hook_items']['filled_whole'])
        summary = swe_checks.hook_summary([dict(selection=self.check(self.synthetic_turn()))])
        self.assertEqual(summary['turns'], 1)

    def test_check_hook_refusals(self):
        def refused(change, pattern, thinking=False, cap=50):
            turn = self.synthetic_turn(thinking, cap)
            change(turn[0], turn[1])
            with self.assertRaisesRegex(Exception, pattern):
                self.check(turn, thinking)
        refused(lambda s, nv: s['hook'].update(ledger_cap_percent=75), 'differs from the contract')
        refused(lambda s, nv: s['hook'].pop('fill'), 'is not the ledger record')
        refused(lambda s, nv: s['hook'].update(ledger_cap_pages=s['hook']['ledger_cap_pages'] + 1), 'hook room')
        refused(lambda s, nv: s['retrieved_pages'].reverse(), 'do not start with')
        refused(lambda s, nv: s['hook']['items'][0].__setitem__(3, 99), 'candidate Pages')
        refused(lambda s, nv: s['hook']['fill'].update(unfilled=s['hook']['fill']['unfilled'] + 1), 'hook fill')
        refused(lambda s, nv: s['hook'].update(reserved_items=s['hook']['reserved_items'] + 1), 'reserved_items')
        refused(lambda s, nv: s['hook']['shares'].update(reasoning=0.1, code=s['hook']['shares']['code'] - 0.1),
                'reasoning share with thinking off')
        refused(lambda s, nv: s['hook']['commands'].update(edits=99), 'hook commands')
        refused(lambda s, nv: nv['query']['body']['selection']['scores'].pop('0'), 'score of every candidate')

        def partly(s, nv):  # a skipped item filled in part: one of its Pages left out (filled whole) or added (skipped)
            item = next(i for i in s['hook']['items'] if i[4] in ('skipped', 'filled_whole'))
            pages = [p for p in range(item[1] // 16, -(-item[2] // 16)) if p in s['retrieved_pages']] \
                if item[4] == 'filled_whole' else \
                [p for p in range(item[1] // 16, -(-item[2] // 16)) if p not in s['selected_pages']]
            if item[4] == 'filled_whole':
                s['selected_pages'].remove(pages[-1])
                s['retrieved_pages'].remove(pages[-1])
            else:
                s['selected_pages'] = sorted(s['selected_pages'] + pages[:1])
                s['retrieved_pages'].append(pages[0])
        refused(partly, 'filled in part')
        refused(partly, 'filled in part', cap=10)

    def test_inactive_turn_without_a_hook(self):
        import smoke_checks as sc
        contract = dict(sc.CONTRACT, ledger_query=dict(c.LEDGER_PIN))
        base = dict(contract_sha256=sc.contract_digest(contract), reason='cold_start', candidate_pages=0,
                    history_tokens=0, capacity_pages=0, selected_pages=[], mandatory_pages=[], recent_pages=[],
                    retrieved_pages=[], projection_sha256=None, selector_config_hash=None)
        native = dict(local_turn=0, stable_end=160, plan=dict(selected=[], candidate_end=0, policy='ledger_query'),
                      query=dict(body=dict(selection=dict(base, initial_prompt_end=160), local_turn=0)))
        row = dict(candidate_page_count=0, selected_logical_pages=[], query=dict(selection=dict(base, hook=None)))
        info = swe_checks.check_selection('ledger_query', row, native, contract, 160, 16)
        self.assertEqual(info['reason'], 'cold_start')
        row['query']['selection']['hook'] = native['query']['body']['selection']['hook'] = dict(order='x')
        with self.assertRaisesRegex(Exception, 'names a hook record'):
            swe_checks.check_selection('ledger_query', row, native, contract, 160, 16)

    def test_selection_parity_replays_up_to_the_hook(self):
        import selection_parity
        hooked = selection_parity.recorded_fill(dict(retrieved_pages=[7, 3], hook=dict(fill=dict(room=4))))
        taken, rest, record = hooked(None, None, 10, 1, 2, 6, [9], [3, 5, 7, 8, 4], 4, {}, frozenset())
        self.assertEqual((taken, rest, record(frozenset())), ([7, 3], [5, 8, 4], dict(fill=dict(room=4))))
        self.assertEqual(selection_parity.replayed(REGISTRY_ROWS)['ledger_query'], 'hook')

    def test_ledger_verify_helpers(self):
        import ledger_verify as lv
        self.assertEqual(lv.attended_fraction(0, 32, {0}, 1000), 0.5)
        self.assertEqual(lv.attended_fraction(16, 48, set(), 32), 0.5)  # [32, 48) past the candidate end
        self.assertEqual(lv.attended_fraction(5, 5, set(), 0), 1.0)
        steps = [dict(kind='action', cmd=x) for x in ['ls'] + ['cat a.py'] * 10 + ['cat b.py']]
        inside, lengths = lv.streaks(steps)
        self.assertEqual((lengths, sorted(inside)), ([10], list(range(1, 11))))
        self.assertEqual(lv.special_ids(True), (1, 5))
        self.assertEqual(lv.special_ids(False), (248045, 248069))
        rows = [dict(query=dict(selection=dict(hook=self.synthetic_turn()[0]['hook']))), dict(query=dict(selection={}))]
        counts = lv.hook_counts(rows)
        self.assertEqual((counts['requests'], counts['edit_item_requests']), (1, 1))
        self.assertIsNone(lv.hook_counts([dict(query=None)]))
        # the history of a request from its prompt's <|im_start|> positions and the trajectory (fake ids: 1 = <|im_start|>,
        # 5 = </think>): two messages of system and task, a rebuilt reply (call at its start), then a reply since the
        # rebuild (its call after the header, thinking on: after </think>) and the live observation
        R, _ = self.modules()
        ids = [1, 0, 0, 1, 0, 0, 1, 9, 9, 9, 1, 9, 9, 1, 2, 2, 2, 5, 3, 3, 1, 7, 7, 1, 8, 8, 8]
        messages = [dict(role='system', content='s'), dict(role='user', content='t'),
                    dict(role='assistant', content='rebuilt'), dict(role='user', content='obs'),
                    dict(role='assistant', content='c', reasoning_content='r'), dict(role='user', content='live')]
        st = lv.starts_of(ids, 1)
        got = lv.history(R, dict(prompt_token_ids=ids), messages, 6, 4, st, True, 5)
        # (ledger v2, a2f9e36: a rebuilt reply renders its reasoning in keep mode; without </think> it was cut while
        # thinking and its call starts at its end)
        self.assertEqual([tuple(h) for h in got], [('assistant', 6, 10, 10, 'rebuilt', None),
                                                   ('user', 10, 13, None, 'obs', None),
                                                   ('assistant', 13, 20, 18, 'c', 'r')])
        off = lv.history(R, dict(prompt_token_ids=ids), messages, 6, 4, st, False, 5)
        self.assertEqual(off[-1], R.HistoryMessage('assistant', 13, 20, 13 + len(ids) - st[-1], 'c', None))


class Loops1115(unittest.TestCase):
    """2026-11-15 (independent audit of gmark32ti job 3602728, D:\\Remote-RA-temp\\scratch\\gmark32ti-audit-1114\\AUDIT.md
    row 7b): ledger_verify.endpoints reported loop_instances = sum(1 for x in loops if x) over the streak lengths, so it
    counted STREAKS (>= 10 identical consecutive commands), not instances (job 3602728: 6 = 6 streaks in 5 instances,
    astropy-13453 has two). Now `loops` (the lengths, kept), `loop_streaks`, `loop_instances` (instances with at least
    one streak) and `loops_by_instance`, in the JSON and the endpoint lines; a record written before the fix prints
    loop_instances n/a."""

    TASKS = ['twice__twice-1', 'once__once-2', 'none__none-3', 'short__short-4']

    @staticmethod
    def trajectory(name, commands):
        """A mini-swe-agent trajectory: system, task, then per command an assistant call and its observation (a
        command None: a FormatError turn, which a streak skips)."""
        messages = [dict(role='system', content='s'), dict(role='user', content='task')]
        for k, command in enumerate(commands):
            if command is None:
                messages.append(dict(role='user', content='format error', extra=dict(interrupt_type='FormatError')))
                continue
            messages.append(dict(role='assistant', content='step %d\n<mswea_bash_command>%s</mswea_bash_command>'
                                 % (k, command)))
            messages.append(dict(role='user', content='<returncode>0</returncode><output>ok %d</output>' % k,
                                 extra=dict(returncode=0, raw_output='ok %d\n' % k)))
        return dict(instance_id=name, messages=messages, info=dict(exit_status='Submitted'), trajectory_format='x')

    def endpoints(self, tmp):
        """One server's evidence (no runtime rows: no session joins) and pool: instance 1 two streaks (10, then 11
        with a FormatError turn inside), instance 2 one (12), instance 3 none, instance 4 nine identical (no streak)."""
        import ledger_verify as lv
        out, pool = Path(tmp) / 'out', Path(tmp) / 'pool'
        out.mkdir()
        (out / 'runtime.jsonl').write_text('', encoding='utf-8')
        runs = {self.TASKS[0]: ['ls'] + ['cat a.py'] * 10 + ['ls'] + ['git diff'] * 5 + [None] + ['git diff'] * 6
                + ['submit'],
                self.TASKS[1]: ['ls', 'pwd'] + ["grep -n 'x' b.py"] * 12,
                self.TASKS[2]: ['ls', 'cat a.py', 'cat b.py', 'git diff'],
                self.TASKS[3]: ['cat c.py'] * 9 + ['ls']}
        for name, commands in runs.items():
            (pool / name).mkdir(parents=True)
            (pool / name / (name + '.traj.json')).write_text(json.dumps(self.trajectory(name, commands)),
                                                             encoding='utf-8')
        return lv.endpoints(out, pool, [dict(instance_id=n) for n in self.TASKS], lv.special_ids(True))

    def test_streaks_and_instances_apart(self):
        import ledger_verify as lv
        with tempfile.TemporaryDirectory() as tmp:
            value = self.endpoints(tmp)
        self.assertEqual((value['instances'], value['loops'], value['loop_streaks'], value['loop_instances']),
                         (4, [12, 11, 10], 3, 2))
        self.assertEqual(value['loops_by_instance'], {self.TASKS[0]: [10, 11], self.TASKS[1]: [12]})
        self.assertNotEqual(value['loop_instances'], sum(1 for x in value['loops'] if x))  # the old formula: 3
        self.assertEqual(sorted(x for v in value['loops_by_instance'].values() for x in v), sorted(value['loops']))
        line = lv.lines('srv', value)[0]
        self.assertTrue(line.startswith('LEDGER_ENDPOINTS srv '), line)
        self.assertIn(' loops=[12, 11, 10] loop_streaks=3 loop_instances=2 resolved=', line)
        self.assertEqual(lv.loop_counts(value), (3, 2))

    def test_job_3602728s_shape_and_an_old_record(self):
        """6 streaks in 5 instances (the audit's gmark32ti: astropy-13453 has 53 and 11); a record written before the
        fix (loop_instances = the streak count, no loop_streaks) prints its streaks and loop_instances n/a."""
        import ledger_verify as lv
        new = dict(loops=[75, 70, 58, 53, 35, 11], loop_streaks=6, loop_instances=5)
        self.assertIn('loops=[75, 70, 58, 53, 35, 11] loop_streaks=6 loop_instances=5', lv.loop_text(new))
        old = dict(loops=[75, 70, 58, 53, 35, 11], loop_instances=6)
        self.assertEqual(lv.loop_counts(old), (6, None))
        self.assertEqual(lv.loop_text(old), 'loops=[75, 70, 58, 53, 35, 11] loop_streaks=6 loop_instances=n/a')
        self.assertEqual(lv.loop_text(dict(loops=[], loop_streaks=0, loop_instances=0)),
                         'loops=[] loop_streaks=0 loop_instances=0')

    def test_the_baselines_carry_both(self):
        """ledger_baselines.json recomputed 2026-11-15 with the fixed code: obst 3539260 2 streaks in 2 instances
        (82 in astropy-13977, 20 in django-10554), ctx25 3461703 none; the BASELINE lines print both."""
        import ledger_verify as lv
        runs = c.load(ROOT / 'postrun' / 'ledger_baselines.json')['runs']
        obst, ctx25 = runs['obst 3539260']['endpoints'], runs['ctx25 3461703']['endpoints']
        self.assertEqual((obst['loops'], obst['loop_streaks'], obst['loop_instances'], obst['loops_by_instance']),
                         ([82, 20], 2, 2, {'astropy__astropy-13977': [82], 'django__django-10554': [20]}))
        self.assertEqual((ctx25['loops'], ctx25['loop_streaks'], ctx25['loop_instances'], ctx25['loops_by_instance']),
                         ([], 0, 0, {}))
        self.assertIn(' loops=[82, 20] loop_streaks=2 loop_instances=2 ',
                      lv.lines('obst 3539260', obst, 'BASELINE ')[0])


class Plans150_1117(unittest.TestCase):
    """2026-11-17 (integrate-1117 2fb7f62): the group's 150-instance runs at 24 sessions: the sealed 150 list
    (tasks/swe150-tasks.json, statements tasks/tasks-pool-150.json, a payload member) and full150k24 / ledger150k24 /
    stockkeep150k24 (the runtime README's 24-session configuration)."""

    def test_the_sealed_150_task_list(self):
        value = PINS[c.TASK_SETS_KEY]['150']
        ids = value['instances']
        self.assertEqual((len(ids), len(set(ids))), (150, 150))
        self.assertEqual(ids[:64], PINS[c.TASK_SETS_KEY]['64']['instances'])
        self.assertEqual(ids[:32], PINS['tasks']['instances'])
        self.assertEqual((ids[64], ids[-1]), ('django__django-11880', 'django__django-14315'))
        self.assertEqual(value['pool']['file'], 'tasks-pool-150.json')
        raw = (ROOT / 'tasks' / 'tasks-pool-150.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), value['pool']['sha256'])
        tasks = json.loads(raw)
        self.assertEqual([t['instance_id'] for t in tasks], ids)
        self.assertEqual(c.tasks_bytes(tasks[:64]), (ROOT / 'tasks' / 'tasks-pool-64.json').read_bytes())
        self.assertEqual(c.tasks_bytes(tasks[:32]), (ROOT / 'tasks' / 'tasks-pool.json').read_bytes())
        self.assertEqual((value['provisional'], value['dataset'], value['split']),
                         (False, 'princeton-nlp/SWE-Bench_Verified', 'test'))
        self.assertIn('CONFIRMED', value['confirmed'])
        payload = json.loads((ROOT / 'dist' / 'payload.json').read_bytes())
        self.assertEqual(payload['files']['tasks-pool-150.json']['sha256'], value['pool']['sha256'])
        self.assertLessEqual(150, PINS['runtime_features']['session_pool']['max_tasks'])

    def test_the_150_task_plans(self):
        plans = c.cfg(PINS)['plans']
        self.assertEqual(PLAN_ORDER[-3:], P150)
        for name, base in (('full150k24', 'full64ak24'), ('ledger150k24', 'ledger64ak24'),
                           ('stockkeep150k24', 'stockkeep64k24')):
            plan, old = plans[name], plans[base]
            self.assertEqual(plan['tasks'], 150)
            self.assertEqual((plan['partition'], plan['role']), ('ghx4', 'validation'))
            self.assertEqual({k: plan.get(k) for k in ('server', 'stock', 'thinking', 'async_scheduling')},
                             {k: old.get(k) for k in ('server', 'stock', 'thinking', 'async_scheduling')})
            (srv,) = c.plan_servers(PINS, name)
            (was,) = c.plan_servers(PINS, base)
            self.assertEqual({k: v for k, v in srv.items() if k not in ('label', 'tasks', 'agent_cap_seconds',
                                                                          'agent_min_seconds')},
                             {k: v for k, v in was.items() if k not in ('label', 'tasks', 'agent_cap_seconds',
                                                                          'agent_min_seconds')})
            self.assertEqual((srv['tasks'], srv['step_limit'], c.server_running(PINS, srv)), (150, 120, 24))
            self.assertEqual(c.server_instances(PINS, srv, name), PINS[c.TASK_SETS_KEY]['150']['instances'])
            self.assertEqual(c.client_timeout(PINS, srv), 3600 if srv['kind'] == c.STOCK else 4400)
            fit = c.validate_plan(PINS, name)
            self.assertTrue(fit['fits'])
            self.assertEqual(plan['eval_wait_seconds'], 2760)
            self.assertIn('_150TASKS_24SESSIONS_', plan['authorization'])
            self.assertTrue(plan['authorization'].endswith('_%dMIN_NO_RETRY' % (fit['time_seconds'] // 60)))
            memory = c.memory_fit(PINS, name)
            self.assertTrue(memory['fits'])
        for name in P150RUNTIME:
            server = c.server_cfg(PINS, c.plan_servers(PINS, name)[0])
            self.assertEqual((server['kv_cache_gib'], server['max_model_len'], server['dram_arena_gib'],
                              server[c.RELEASE_KEY], server['max_running_sessions']), (25, 262144, 68, True, 24))
            self.assertTrue(plans[name]['thinking'] and plans[name]['async_scheduling'])
        (ledger,) = c.plan_servers(PINS, 'ledger150k24')
        self.assertEqual(ledger['policy'], 'ledger_query')
        self.assertNotIn(c.LEDGER_KEY, ledger)  # the default cap 50 %: no --ledger-cap-percent
        self.assertEqual(plans['stockkeep150k24']['stock']['max_num_seqs'], 24)
        self.assertEqual(plans['stockkeep150k24']['stock']['history'], 'reasoning_kept')
        self.assertEqual([plans[n]['time'] for n in P150], ['03:15:00', '04:02:00', '05:06:00'])


class Gate2Parity1117(unittest.TestCase):
    """2026-11-17, release gate 2 (the runtime's docs/release-gates.md section 3, check 2: full against stock, token
    for token): plan parity150 (our full server with full150k24's settings, then stock vLLM with stockkeep150k24's,
    both with the parity_probe client), the probes' records, the check (parity.evaluate) and its refusals."""

    def settings(self):
        import parity
        return parity.settings_of(c.cfg(PINS)['plans']['parity150'])

    def test_the_plan(self):
        import parity
        plans = c.cfg(PINS)['plans']
        plan, full, stock = plans['parity150'], plans['full150k24'], plans['stockkeep150k24']
        self.assertEqual(c.cfg(PINS)['plan_order'][-1:], GATE2)
        self.assertEqual((plan['role'], plan['partition'], plan['time'], plan['expected_seconds'], plan['tasks'],
                          plan['thinking'], plan['async_scheduling']),
                         ('release-gate', 'ghx4', '00:55:00', 1500, 150, True, True))
        self.assertEqual((plan['server'], plan['stock']), (full['server'], stock['stock']))
        ours, theirs = c.plan_servers(PINS, 'parity150')
        (f,), (s,) = c.plan_servers(PINS, 'full150k24'), c.plan_servers(PINS, 'stockkeep150k24')
        skip = ('label', 'step_limit', 'agent_cap_seconds', 'agent_min_seconds', 'client')
        self.assertEqual({k: v for k, v in ours.items() if k not in skip}, {k: v for k, v in f.items() if k not in skip})
        self.assertEqual({k: v for k, v in theirs.items() if k not in skip + ('settings', 'thinking')},
                         {k: v for k, v in s.items() if k not in skip + ('settings', 'thinking')})
        self.assertEqual(c.launch_arguments(PINS, ours, REWRITER), c.launch_arguments(PINS, f, REWRITER))
        self.assertEqual(c.stock_argv(PINS, 8000, theirs), c.stock_argv(PINS, 8000, s))  # the same stock server
        self.assertEqual((ours['client'], theirs['client']), (c.PARITY_CLIENT, c.PARITY_CLIENT))
        settings = self.settings()
        self.assertEqual({k: settings[k] for k in ('prompts', 'window', 'logprobs', 'near_tie_nats', 'resyncs')},
                         dict(prompts=8, window=256, logprobs=5, near_tie_nats=0.1, resyncs=3))
        raw = (ROOT / 'parity-prompts.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), settings['prompts_sha256'])
        prompts = parity.load_prompts(ROOT / 'parity-prompts.json', settings,
                                      json.loads((ROOT / 'tasks' / 'tasks-pool-150.json').read_bytes()))
        self.assertEqual([p['instance_id'] for p in prompts], c.instance_ids(PINS, 'parity150')[:32:4])
        fit, cost = c.validate_plan(PINS, 'parity150'), c.plan_cost(PINS, 'parity150')
        self.assertEqual((fit['worst_case_seconds'], fit['time_seconds'], fit['spare_seconds']), (3300, 3300, 0))
        self.assertEqual(c.max_limit_over_expected(plan), 2.25)
        self.assertEqual(c.memory_fit(PINS, 'parity150'), c.memory_fit(PINS, 'full150k24'))
        self.assertIsNone(c.plan_blocked(PINS, 'parity150'))
        self.assertIn('parity.py', c.load(ROOT / 'dist' / 'payload.json')['files'])
        self.assertIn('parity-prompts.json', c.load(ROOT / 'dist' / 'payload.json')['files'])
        self.assertGreater(cost['gpu_hours_limit'], 0)

    def test_refusals(self):
        def refused(change, pattern, plan='parity150'):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex((RuntimeError, ValueError), pattern):
                c.validate_plan(pins, plan)
        p = lambda pins: pins['config']['plans']['parity150']
        refused(lambda pins: p(pins)['parity'].update(window=8), 'window must be')
        refused(lambda pins: p(pins)['parity'].update(prompts_sha256='0' * 64), 'differs from the plan')
        refused(lambda pins: p(pins)['parity'].update(extra=1), 'must name exactly')
        refused(lambda pins: p(pins)['parity'].update(prompts=33), 'prompts must be')
        refused(lambda pins: p(pins)['servers'].reverse(), 'exactly our full server then the stock control')
        refused(lambda pins: p(pins)['servers'][0].update(policy='observation_query'), 'exactly our full server')
        refused(lambda pins: p(pins)['servers'][1].pop('client'), 'run the client parity_probe|exactly our full')
        refused(lambda pins: p(pins).update(role='validation'), 'role release-gate')
        refused(lambda pins: p(pins).update(thinking=False), 'on a plan with thinking true')  # (the kept history)
        refused(lambda pins: (p(pins).update(thinking=False), p(pins)['stock'].pop('history')),
                'role release-gate and thinking true')
        refused(lambda pins: p(pins).pop('parity'), 'thinking must be true or false|own server settings')
        # the probe client on any other plan, the role without the settings
        refused(lambda pins: pins['config']['plans']['full150k24']['servers'][0].update(client=c.PARITY_CLIENT),
                'servers of a gate-2 plan only', 'full150k24')
        refused(lambda pins: pins['config']['plans']['full150k24'].update(role='release-gate'),
                'without parity settings', 'full150k24')

    def test_the_check(self):
        import parity
        settings = dict(self.settings(), window=8, resyncs=2)
        tasks = [dict(instance_id='a', problem_statement='A'), dict(instance_id='b', problem_statement='B')]
        prompts = [dict(instance_id='a', messages=[]), dict(instance_id='b', messages=[])]
        p0, g = [1, 2, 3], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

        def rows(index, prompt, generated, rebuild=False):
            return [dict(session_index=index, turn=0, prompt_token_ids=prompt,
                         stage_b=dict(prompt_end=len(prompt), computed_end=len(prompt) + len(generated) - 1)),
                    dict(session_index=index, turn=1, prompt_token_ids=prompt + generated + [2, 7, 8],
                         stage_b=dict(rebuild=rebuild))]

        def window(instance, start, ids, top=None):
            top = top or [{t: -0.01} for t in ids]
            return dict(instance_id=instance, step='window', start=start, http=200, response=dict(choices=[dict(
                finish_reason='length', logprobs=dict(tokens=['token_id:%d' % t for t in ids],
                                                      top_logprobs=[{'token_id:%d' % k: v for k, v in d.items()}
                                                                    for d in top]))]))

        def tokenize(instance, ids):
            return dict(instance_id=instance, step='tokenize', http=200, response=dict(tokens=ids))
        runtime = rows(0, p0, g) + rows(1, [4, 5], g)
        equal = [tokenize('a', p0), window('a', 0, g[:8]), tokenize('b', [4, 6]), window('b', 0, g[:8])]
        value = parity.evaluate(prompts, runtime, tasks, equal, settings)
        self.assertEqual((value['verdict'], value['equal_windows'], value['tokens_compared'], value['render_equal']),
                         (parity.PASS, 2, 16, 1))
        self.assertEqual(value['results'][1]['render'], dict(equal=False, first_difference=1, stock_tokens=2,
                                                            runtime_tokens=2))
        # a near tie at step 3 (margin 0.05 < 0.1), resynced, the rest equal
        tie = [window('a', 0, g[:3] + [99], [{t: -0.01} for t in g[:3]] + [{99: -0.3, g[3]: -0.35}]),
               window('a', 4, g[4:8]), window('b', 0, g[:8])]
        value = parity.evaluate(prompts, runtime, tasks, tie, settings)
        self.assertEqual((value['verdict'], value['near_tie_windows'], value['divergences'], value['max_margin_nats']),
                         (parity.NEAR_TIE, 1, 1, 0.05))
        self.assertEqual(value['results'][0]['equal_prefix'], 3)
        # the runtime's token outside stock's top-k: DIFFERS with a lower bound
        far = [window('a', 0, [99], [{99: -0.01, 98: -4.0}]), window('a', 1, g[1:8]), window('b', 0, g[:8])]
        value = parity.evaluate(prompts, runtime, tasks, far, settings)
        self.assertEqual(value['verdict'], parity.DIFFERS)
        self.assertEqual((value['results'][0]['divergences'][0]['margin_lower_bound'],
                          value['results'][0]['divergences'][0]['margin_nats']), (True, 3.99))
        # resyncs exhausted: the window ends early, still a verdict (two near ties and one more)
        many = [window('a', 0, [90], [{90: -0.3, g[0]: -0.31}]), window('a', 1, [91], [{91: -0.3, g[1]: -0.31}]),
                window('a', 2, [92], [{92: -0.3, g[2]: -0.31}]), window('b', 0, g[:8])]
        value = parity.evaluate(prompts, runtime, tasks, many, settings)
        self.assertEqual((value['verdict'], value['results'][0]['uncompared']), (parity.NEAR_TIE, 5))
        # incomplete: no window, an HTTP error, a short window, out of order, no turn-1 extension
        for records in ([window('b', 0, g[:8])],
                        [dict(window('a', 0, g[:8]), http=500), window('b', 0, g[:8])],
                        [window('a', 0, g[:5]), window('b', 0, g[:8])],
                        [window('a', 2, g[2:8]), window('b', 0, g[:8])],
                        [window('a', 0, g[:3] + [99], [{t: -0.01} for t in g[:3]] + [{99: -0.3, g[3]: -0.35}]),
                         window('b', 0, g[:8])]):
            self.assertEqual(parity.evaluate(prompts, runtime, tasks, records, settings)['verdict'], parity.INCOMPLETE)
        rebuilt = rows(0, p0, g, rebuild=True) + rows(1, [4, 5], g)
        value = parity.evaluate(prompts, rebuilt, tasks, equal, settings)
        self.assertEqual((value['verdict'], value['results'][0]['reason']),
                         (parity.INCOMPLETE, 'turn 1 does not extend turn 0 (a rebuild or another prompt)'))
        self.assertIn('GATE2 GATE2_INCOMPLETE prompts=2', parity.line(value))

    def test_the_probes(self):
        import parity
        settings = dict(self.settings(), window=4, resyncs=1)
        tasks = [dict(instance_id='a', problem_statement='A')]
        prompts = [dict(instance_id='a', messages=[dict(role='system', content='s'), dict(role='user', content='A')])]
        sent = []

        def post(url, body, timeout):
            sent.append((url.rsplit('/', 1)[-1], body))
            if url.endswith('/chat/completions'):
                return 200, json.dumps(dict(choices=[dict(message=dict(content='C', reasoning_content='R'))])).encode()
            if url.endswith('/close'):
                return 200, b'{"object": "session.close"}'
            if url.endswith('/tokenize'):
                return 200, b'{"tokens": [1, 2]}'
            ids = ([10, 99, 12, 13] if len(body['prompt']) == 2 else [12, 13])[:body['max_tokens']]
            top = [{'token_id:%d' % t: -0.1} for t in ids]
            return 200, json.dumps(dict(choices=[dict(logprobs=dict(tokens=['token_id:%d' % t for t in ids],
                                                                     top_logprobs=top))])).encode()
        clock = iter(range(1000)).__next__
        with tempfile.TemporaryDirectory(prefix='swe32-gate2-') as tmp:
            agent, instances = parity.runtime_probe('http://x', tmp, prompts, model='selective-prefix',
                                                    extra_body=dict(chat_template_kwargs=dict(enable_thinking=True)),
                                                    max_tokens=4096, deadline=500, post=post, clock=clock)
            self.assertEqual((agent['outcome'], instances['a']['status'], instances['a']['api_calls']),
                             ('completed', 'done', 2))
            self.assertEqual([s[0] for s in sent], ['completions', 'completions', 'close'])
            turn1 = sent[1][1]['messages']
            self.assertEqual(turn1[2:], [dict(role='assistant', content='C', reasoning_content='R'),
                                         dict(role='user', content=parity.OBSERVATION)])
            self.assertEqual((sent[0][1]['chat_template_kwargs'], sent[0][1]['temperature']),
                             (dict(enable_thinking=True), 0.0))
            runtime = [dict(session_index=0, turn=0, prompt_token_ids=[1, 2], stage_b=dict(prompt_end=2,
                                                                                          computed_end=5)),
                       dict(session_index=0, turn=1, prompt_token_ids=[1, 2, 10, 11, 12, 13, 2], stage_b={})]
            sent.clear()
            agent, instances = parity.stock_probe('http://x', tmp, runtime, tasks, prompts, settings,
                                                  model='selective-prefix', chat_kwargs=dict(enable_thinking=True),
                                                  deadline=500, post=post, clock=clock)
            self.assertEqual((agent['outcome'], instances['a']['api_calls']), ('completed', 3))
            windows = [b for name, b in sent if name == 'completions']
            self.assertEqual([(w['prompt'], w['max_tokens'], w['cache_salt']) for w in windows],
                             [([1, 2], 4, 'gate2-a-0'), ([1, 2, 10, 11], 2, 'gate2-a-2')])
            self.assertTrue(all(w['temperature'] == 0.0 and w['logprobs'] == 5 and w['return_tokens_as_token_ids']
                                for w in windows))
            value = parity.evaluate(prompts, runtime, tasks, parity.read_records(Path(tmp) / parity.STOCK_RECORD),
                                    settings)
            self.assertEqual((value['verdict'], value['results'][0]['divergences'][0]['position']),
                             (parity.DIFFERS, 1))  # (99 outside the top-1 the fake returned: a lower bound)
            # the deadline: a prompt after it is truncated, the outcome agent_timeout
            agent, instances = parity.runtime_probe('http://x', tmp, prompts, model='m', extra_body={}, max_tokens=4096,
                                                    deadline=0, post=post, clock=clock)
            self.assertEqual((agent['outcome'], instances['a']['status']), ('agent_timeout', 'truncated'))

    def test_the_job_and_the_verifier(self):
        import job
        results = [dict(label='g2full', client=c.PARITY_CLIENT, agent=dict(outcome='completed')),
                   dict(label='g2stock', client=c.PARITY_CLIENT, agent=dict(outcome='completed'))]
        self.assertEqual(job.launch_evaluation(None, PINS, results, object(), 'swe32-1-r1'),
                         dict(status='EVAL_NOT_NEEDED', labels=[]))
        with tempfile.TemporaryDirectory(prefix='swe32-gate2-') as tmp:
            self.assertIsNone(job.gate2_result(Path(tmp), PINS, 'full150k24'))
            value = job.gate2_result(Path(tmp), PINS, 'parity150')  # nothing ran: INCOMPLETE, recorded
            self.assertEqual(value['verdict'], 'GATE2_INCOMPLETE')
            self.assertEqual(c.load(Path(tmp) / 'gate2.json'), value)
        verify = (ROOT / 'postrun' / 'verify_evidence.py').read_text(encoding='utf-8')
        remote = (ROOT / 'postrun' / 'tbench_verify' / 'remote_verify.py').read_text(encoding='utf-8')
        self.assertIn("report.run('gate2-recomputed', gate2_check)", verify)
        self.assertIn("if report.get('gate2') is not None:", remote)


if __name__ == '__main__':
    unittest.main()
