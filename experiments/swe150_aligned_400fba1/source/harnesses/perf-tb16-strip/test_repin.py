"""Local check of the runtime re-pin path of perf-tb16 in disposable copies of the harness (never in place). No step
ever starts a real submission of a PINNED copy (only -DryRun, which makes no connection).

1. `build.py --commit <the pinned commit>` again (the runtime repository; the copy keeps the pinned release-check
   receipt): the same public package, BUILD_PASS, RUNTIME_PIN=PINNED, the chat protocol's module the accepted one; the
   entry script's DryRun of every plan without NOT_SUBMITTABLE (a blocked plan: only NOT_SUBMITTABLE blocked_plan; none
   was blocked from the integrate-1021 9ef0d6d re-pin on; ctx25 was on 4f5f4e7 by the Qwen KV sizes, not on cf2b28d);
2. `build.py --commit <integrate-1015 0702cab>` (the share repository: no chat protocol; the copy's config without the
   two switches it lacks, --decode-fastpath3 and --deferred-completion-io): RUNTIME_PIN=PLACEHOLDER and the entry script
   refusing a real submission (before any connection: the placeholder check);
3. the pinned commit with controller.INTERFACE_ACCEPTED = None in the copy (a module the harness has not accepted):
   PLACEHOLDER, DryRun NOT_SUBMITTABLE, a real submission refused;
4. `build.py --commit <integrate-1016 0d37346>` (the Qwen chat protocol without Gemma's turns: the pin before the Gemma
   plans; another module and no release-check receipt of it): RUNTIME_PIN=PLACEHOLDER, a Gemma plan's and a Qwen
   plan's DryRun NOT_SUBMITTABLE, a real submission of dryo refused;
5. the light login gates' receipt: the pinned commit with the receipt removed from the copy's pins.json
   (`build.py --reuse-bundle`): PLACEHOLDER naming the release check, DryRun NOT_SUBMITTABLE, a real submission
   refused; and `build.py --release-receipt` with a receipt whose log SHA-256 is not its log's: the build refused;
6. the previous pin, integrate-1018 f9322a6 (2026-10-21): with the sealed config (the optimized flags) the build is
   refused (no --deferred-completion-io); with that switch off in the copy: PLACEHOLDER (another chat module, no
   goal_query on chat-text, no receipt of it), the goal plan's and dryo's DryRun NOT_SUBMITTABLE, a real submission
   refused;
7. the previous pin, integrate-1019-tbgoal 120bc3c (2026-10-22; the Qwen goal dry run's runtime): PLACEHOLDER (its chat
   module 8e1412bc..., the Gemma launcher without goal_query, the third fast path or the deferred files), the Gemma
   goal and obs dry plans' and goaldry's DryRun NOT_SUBMITTABLE, a real ggoaldry submission refused;
8. the pin before the engine-stall fix, integrate-1021 9ef0d6d with its own receipt (the same chat module): the build
   refused, the goal plans need the linear-time quoted-literal pass (job 3429670);
9. the pin before the release smoke, thinking-strip 8ca8511 with its receipt (2026-11-03): PINNED (its chat module
   d9028de5... accepted in the copy: the accepted module is f5a5c36's since 2026-11-09), the smoke plans that name the
   vLLM Goal rewriter, --goal-overlap or --async-scheduling blocked by themselves (smoke_block), strip25 and smokegh
   runnable;
10. integrate-1103 ea18256 with its receipt (the release smoke's, frozen in perf-tb16-strip-44e60bc5; 2026-11-09):
   PLACEHOLDER (its chat module d9028de5... is not the accepted one), gctx25's DryRun NOT_SUBMITTABLE, a real gctx25
   submission refused;
11. gemma-tb-262k f5a5c36 with its receipt (gctx25's, job 3560498; frozen in perf-tb16-strip-3acc490a; its chat module
   2b492e67... accepted in the copy: the accepted one is integrate-1112's since 2026-11-12): PINNED, its Terminal-Bench
   client writes the Gemma job at 262,144 (decision D-A): gctx25 runnable (no NOT_SUBMITTABLE); smokeqt and smokeq
   blocked by the async guard (no async resolve fix: async-maxtok-1112 13ee618, job 3568311);
12. the async-fix guard (decision D-G): recent-fill-1103 f4ccd4b with its receipt (chat module 2b492e67... accepted in
   the copy, neither async fix): PINNED, the async plans smokeq and smokeqt blocked by themselves (async_fix_block:
   NOT_SUBMITTABLE blocked_plan naming the async fixes), smokeqs (async off) runnable; gctx25 blocked (no client);
13. the previous pin, main ba4d1ba with its receipt (2026-11-09 to 11-12, frozen in perf-tb16-strip-58de72a9; chat
   module 2b492e67... accepted in the copy): PINNED, smokeqt and smokeq blocked by the async guard (the resolve fix
   missing: perf-mab smqt 3568311 failed there), smokeqs runnable, gctx25 blocked by the client block (Gemma 65,536).
Prints REPIN_PASS (exit 0) or REPIN_FAIL.
Usage: python -B test_repin.py [--work <new dir under D:>] [--runtime-repo DIR]
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402

SHARE = ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies'
# the release smoke (2026-11-03): integrate-1103 ea18256 = thinking-strip 8ca8511 merged with the vLLM Goal rewriter,
# --goal-overlap, async-c / async-int (--async-scheduling, opt-in), defaults-on (the speed defaults) and parallel-check
# (perf-tb16-strip until 2026-11-03: thinking-strip 8ca8511, frozen in perf-tb16-strip-16803f97)
# (gctx25, 2026-11-09: gemma-tb-262k f5a5c36 = main 1684a46 + the Terminal-Bench client's Gemma context 262,144 and its
# docs; the release smoke's integrate-1103 ea18256 is the pin before it, step 10)
# (the release smoke matrix re-pin, 2026-11-09: main ba4d1ba, read-only detached worktree pin-ba4d1ba; gemma-tb-262k
# f5a5c36 is the previous pin, step 11; the async-fix guard on recent-fill-1103 f4ccd4b, step 12)
# (the integrate-1112 re-pin, 2026-11-12: 4ebff97, read-only detached worktree pin-4ebff97; main ba4d1ba is the
# previous pin, step 13; every earlier step's copy accepts its own chat module where it needs one)
RUNTIME = ROOT.parents[2] / 'worktrees' / 'pin-4ebff97'
PREVIOUS_MAIN = ROOT.parents[2] / 'worktrees' / 'pin-ba4d1ba'
PREVIOUS_MAIN_COMMIT = 'ba4d1ba4fc37ed9d3d593cdc6333b43628b36903'
PREVIOUS_MAIN_RECEIPT = r'D:\Remote-RA-temp\scratch\main-ba4d1ba\release-receipt-ba4d1ba.json'
MAIN_MODULE = '2b492e676bb5807e101e84f8ce66afa4ff8b58220aee05294b338e3a7236c8f8'  # f5a5c36 / f4ccd4b / ba4d1ba's
PREVIOUS_262K = ROOT.parents[2] / 'worktrees' / 'gemma-tb-262k'
PREVIOUS_262K_COMMIT = 'f5a5c369f7183989d116a6588c91b86c608a1026'
PREVIOUS_262K_RECEIPT = r'D:\Remote-RA-temp\scratch\gemma-tb-262k\receipt\release-receipt-f5a5c36.json'
NO_FIX = ROOT.parents[2] / 'worktrees' / 'recent-fill-1103'
NO_FIX_COMMIT = 'f4ccd4bce7af10f3e143a835d49b61b731ba0542'  # recent-fill-1103: chat module 2b492e67..., no async fix
NO_FIX_RECEIPT = r'D:\Remote-RA-temp\scratch\recent-fill\release-receipt-f4ccd4b.json'
STALL = ROOT.parents[2] / 'worktrees' / 'integrate-1021'
STALL_COMMIT = '9ef0d6d721d6f82a5f724d3f37965869092f8bf4'  # integrate-1021: before the quoted-literal fix
STALL_RECEIPT = r'D:\Remote-RA-temp\scratch\integrate-1021\release-receipt-9ef0d6d.json'
TBGOAL = ROOT.parents[2] / 'worktrees' / 'integrate-1019-tbgoal'
TBGOAL_COMMIT = '120bc3c0a6db281eeba21f541957c3ae6941542d'  # integrate-1019-tbgoal: the pin before 2026-10-22
PREVIOUS = ROOT.parents[2] / 'worktrees' / 'integrate-1018'
PREVIOUS_COMMIT = 'f9322a65477deb7a6660c08a316f2d4c26644c26'  # integrate-1018: the pin before 2026-10-21
OLD_COMMIT = '0702cab5940795a9c319af0da274dd902d6a81e1'  # integrate-1015: mswea-xml and kv-session only
QWEN_ONLY = ROOT.parents[2] / 'worktrees' / 'integrate-1016'
QWEN_ONLY_COMMIT = '0d3734641ffd061272905d40d67f41adbd98c231'  # integrate-1016: chat-text without Gemma's turns
SKIP = {'validation', '__pycache__', 'dist'}
# the local check_git_release receipt of the pinned commit (its log next to it; read, never changed)
RECEIPT = r'D:\Remote-RA-temp\scratch\integrate-1112\receipt\release-receipt-4ebff97.json'
# the pin before gctx25 (perf-tb16-strip 2026-11-03, frozen in perf-tb16-strip-44e60bc5): integrate-1103 ea18256
SMOKE = ROOT.parents[2] / 'worktrees' / 'integrate-1103'
SMOKE_COMMIT = 'ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46'
SMOKE_RECEIPT = r'D:\Remote-RA-temp\scratch\parallel-check\release-receipt-ea18256.json'
SMOKE_MODULE = 'd9028de5ab83a8ce4fe6c3c48a2f1a2575737efe86701c7dbaaf06b2a80fd9bf'  # 8ca8511 / ea18256's chat module
# the pin before the release smoke (perf-tb16-strip 2026-10-30, frozen in perf-tb16-strip-16803f97)
STRIP = ROOT.parents[2] / 'worktrees' / 'thinking-strip'
STRIP_COMMIT = '8ca8511be72b5fdb9b02df8d2e7b03ca43f5bd4c'
STRIP_RECEIPT = r'D:\Remote-RA-temp\scratch\thinking-strip\release-receipt-8ca8511.json'


def switches_off(target, keys=('decode_fastpath3', 'deferred_completion_io')):
    """The copy's tb16-config.json without switches an older runtime lacks (the plans otherwise as sealed)."""
    path = target / 'tb16-config.json'
    value = json.loads(path.read_bytes())
    for key in keys:
        value['server'][key] = False
    path.write_bytes((json.dumps(value, indent=1, ensure_ascii=False) + '\n').encode())


def accept(target, module):
    """The copy's controller.INTERFACE_ACCEPTED set to another chat module (a previous pin's own)."""
    path = target / 'controller.py'
    source = path.read_text(encoding='utf-8')
    start = source.index('\nINTERFACE_ACCEPTED = ')
    end = source.index('\n', start + 1)
    path.write_text(source[:start] + "\nINTERFACE_ACCEPTED = '%s'" % module + source[end:], encoding='utf-8', newline='\n')


def copy_harness(work):
    target = work / 'perf-tb16'
    shutil.copytree(ROOT, target, ignore=lambda d, names: [n for n in names if n in SKIP or n.endswith('.pyc')])
    (target / 'dist').mkdir()
    return target


def run(command, cwd, timeout=1800):
    done = subprocess.run([str(x) for x in command], cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    return done.returncode, done.stdout + done.stderr


def line(text, prefix):
    found = [x for x in text.splitlines() if x.startswith(prefix)]
    return found[-1] if found else ''


def entry(target, plan, pins, dry=True):
    """The copy's entry script with the plan's phrase; `dry` False only on copies whose pin is not PINNED."""
    built = c.load(target / 'pins.json')
    if not dry:
        assert built['runtime_pin']['status'] != 'PINNED', 'a real submission is never started on a PINNED copy'
    phrase = pins['config']['plans'][plan]['authorization']
    command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', target / 'run_one_connection.ps1',
               '-Plan', plan, '-Authorize', '-Authorization', phrase] + (['-DryRun'] if dry else [])
    return run(command, target, 600)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work')
    parser.add_argument('--runtime-repo', default=str(RUNTIME))
    args = parser.parse_args(argv)
    work = Path(args.work or (r'D:\Remote-RA-temp\scratch\perf-tb16-strip\repin-' + time.strftime('%Y%m%dT%H%M%S')))
    work.mkdir(parents=True)
    problems = []
    pins = c.load(ROOT / 'pins.json')
    # 1. the pinned commit again
    target = copy_harness(work / 'same')
    code, text = run([sys.executable, '-B', 'build.py', '--repo', args.runtime_repo, '--commit', pins['commit']], target)
    print('SAME_COMMIT exit=%s %s | %s' % (code, line(text, 'RUNTIME_PIN='), line(text, 'BUILD_PASS')[:160]))
    if code != 0 or 'RUNTIME_PIN=PINNED' not in text or 'BUILD_PASS' not in text:
        problems.append('re-pin to the same commit: exit %s %s' % (code, text[-1500:]))
    else:
        repinned = c.load(target / 'pins.json')
        chat = repinned['runtime_features']['chat_text']
        if repinned['package_sha256'] != pins['package_sha256'] or chat['module_sha256'] != c.INTERFACE_ACCEPTED:
            problems.append('the same commit gave another package or chat module')
        if repinned.get('release_receipt') != pins.get('release_receipt') or not repinned['runtime_pin'].get(
                'release_receipt'):
            problems.append('the same commit lost its pinned release-check receipt')
        for plan in repinned['config']['plan_order']:
            code, text = entry(target, plan, repinned)
            blocked = c.plan_blocked(repinned, plan)  # (ctx25: the sealed Qwen KV sizes, controller.kv_block)
            refusals = [x for x in text.splitlines() if x.startswith('NOT_SUBMITTABLE')]
            if code != 0 or 'DRY_RUN connections=0' not in text or 'RELEASE_RECEIPT commit=' + pins['commit'] not in text \
                    or refusals != (['NOT_SUBMITTABLE blocked_plan=%s reason=%s' % (plan, blocked)] if blocked else []):
                problems.append('DryRun of %s on the PINNED copy: exit %s %s' % (plan, code, text[-600:]))
        print('SAME_COMMIT dry runs: %d plans' % len(repinned['config']['plan_order']))
    # 2. a commit without the chat protocol
    target = copy_harness(work / 'old')
    switches_off(target)
    code, text = run([sys.executable, '-B', 'build.py', '--repo', SHARE, '--commit', OLD_COMMIT], target)
    print('OLD_COMMIT exit=%s %s' % (code, line(text, 'RUNTIME_PIN=')[:220]))
    if code != 0 or 'RUNTIME_PIN=PLACEHOLDER' not in text:
        problems.append('integrate-1015 0702cab: exit %s %s' % (code, text[-1500:]))
    else:
        code, text = entry(target, 'full', pins, dry=False)
        print('OLD_COMMIT real submission refused=%s' % (code != 0))
        if code == 0 or 'nothing was submitted' not in text:
            problems.append('the entry script accepted a real submission on a PLACEHOLDER pin: %s' % text[-400:])
    # 3. the pinned commit with the module not accepted
    target = copy_harness(work / 'unaccepted')
    path = target / 'controller.py'
    source = path.read_text(encoding='utf-8')
    start = source.index('\nINTERFACE_ACCEPTED = ')
    end = source.index('\n', start + 1)
    path.write_text(source[:start] + '\nINTERFACE_ACCEPTED = None' + source[end:], encoding='utf-8', newline='\n')
    code, text = run([sys.executable, '-B', 'build.py', '--reuse-bundle'], target)
    print('UNACCEPTED exit=%s %s' % (code, line(text, 'RUNTIME_PIN=')[:220]))
    if code != 0 or 'RUNTIME_PIN=PLACEHOLDER' not in text or 'not adapted' not in text:
        problems.append('the unaccepted module did not keep the pin a PLACEHOLDER: %s' % text[-1500:])
    else:
        code, text = entry(target, 'dry', pins)
        if code != 0 or 'NOT_SUBMITTABLE runtime_pin=PLACEHOLDER' not in text:
            problems.append('DryRun on the unaccepted copy: exit %s %s' % (code, text[-600:]))
        code, text = entry(target, 'obs', pins, dry=False)
        print('UNACCEPTED real submission refused=%s' % (code != 0))
        if code == 0:
            problems.append('a real submission was accepted on the unaccepted copy')
    # 4. the Qwen-only chat protocol (the pin before the Gemma plans)
    target = copy_harness(work / 'qwen-only')
    switches_off(target)
    code, text = run([sys.executable, '-B', 'build.py', '--repo', QWEN_ONLY, '--commit', QWEN_ONLY_COMMIT], target)
    print('QWEN_ONLY exit=%s %s' % (code, line(text, 'RUNTIME_PIN=')[:220]))
    if code != 0 or 'RUNTIME_PIN=PLACEHOLDER' not in text:
        problems.append('integrate-1016 0d37346 did not keep the pin a PLACEHOLDER: %s' % text[-1500:])
    else:
        for plan in ('gdry', 'full'):
            code, text = entry(target, plan, pins)
            if code != 0 or 'NOT_SUBMITTABLE runtime_pin=PLACEHOLDER' not in text:
                problems.append('DryRun of %s on the Qwen-only copy: exit %s %s' % (plan, code, text[-600:]))
        code, text = entry(target, 'dryo', pins, dry=False)
        print('QWEN_ONLY real submission refused=%s' % (code != 0))
        if code == 0 or 'nothing was submitted' not in text:
            problems.append('a real dryo submission was accepted on the Qwen-only copy: %s' % text[-400:])
    # 5. the pinned commit without its release-check receipt; a receipt that does not match its log
    target = copy_harness(work / 'no-receipt')
    copied = c.load(target / 'pins.json')
    copied.pop('release_receipt')
    (target / 'pins.json').write_bytes((json.dumps(copied, sort_keys=True, indent=2) + '\n').encode())
    code, text = run([sys.executable, '-B', 'build.py', '--reuse-bundle'], target)
    print('NO_RECEIPT exit=%s %s' % (code, line(text, 'RUNTIME_PIN=')[:220]))
    if code != 0 or 'RUNTIME_PIN=PLACEHOLDER' not in text or 'release check' not in line(text, 'RUNTIME_PIN='):
        problems.append('the pinned commit without its receipt did not build a PLACEHOLDER: %s' % text[-1500:])
    else:
        code, text = entry(target, 'dryo', pins)
        if code != 0 or 'NOT_SUBMITTABLE runtime_pin=PLACEHOLDER' not in text:
            problems.append('DryRun without the receipt: exit %s %s' % (code, text[-600:]))
        code, text = entry(target, 'dryo', pins, dry=False)
        print('NO_RECEIPT real submission refused=%s' % (code != 0))
        if code == 0 or 'nothing was submitted' not in text:
            problems.append('a real submission was accepted without the receipt: %s' % text[-400:])
    # 6. the previous pin (integrate-1018 f9322a6): refused with the optimized flags, a PLACEHOLDER without the one it lacks
    target = copy_harness(work / 'previous')
    code, text = run([sys.executable, '-B', 'build.py', '--repo', PREVIOUS, '--commit', PREVIOUS_COMMIT], target)
    print('PREVIOUS_PIN exit=%s refused=%s' % (code, code != 0 and '--deferred-completion-io' in text))
    if code == 0 or '--deferred-completion-io' not in text:
        problems.append('f9322a6 with the optimized flags was not refused: exit %s %s' % (code, text[-1500:]))
    target = copy_harness(work / 'previous-off')
    switches_off(target, ('deferred_completion_io',))
    code, text = run([sys.executable, '-B', 'build.py', '--repo', PREVIOUS, '--commit', PREVIOUS_COMMIT], target)
    print('PREVIOUS_PIN_OFF exit=%s %s' % (code, line(text, 'RUNTIME_PIN=')[:220]))
    if code != 0 or 'RUNTIME_PIN=PLACEHOLDER' not in text:
        problems.append('f9322a6 without the deferred switch did not build a PLACEHOLDER: %s' % text[-1500:])
    else:
        for plan in ('goal', 'dryo'):
            code, text = entry(target, plan, pins)
            if code != 0 or 'NOT_SUBMITTABLE runtime_pin=PLACEHOLDER' not in text:
                problems.append('DryRun of %s on the previous-pin copy: exit %s %s' % (plan, code, text[-600:]))
        code, text = entry(target, 'goal', pins, dry=False)
        print('PREVIOUS_PIN_OFF real submission refused=%s' % (code != 0))
        if code == 0 or 'nothing was submitted' not in text:
            problems.append('a real goal submission was accepted on the previous-pin copy: %s' % text[-400:])
    # 7. the previous pin (integrate-1019-tbgoal 120bc3c): another chat module, no Gemma goal_query: a PLACEHOLDER
    target = copy_harness(work / 'tbgoal')
    code, text = run([sys.executable, '-B', 'build.py', '--repo', TBGOAL, '--commit', TBGOAL_COMMIT], target)
    print('TBGOAL_PIN exit=%s %s' % (code, line(text, 'RUNTIME_PIN=')[:220]))
    if code != 0 or 'RUNTIME_PIN=PLACEHOLDER' not in text:
        problems.append('120bc3c did not build a PLACEHOLDER: %s' % text[-1500:])
    else:
        for plan in ('ggoaldry', 'gobsdry', 'goaldry'):
            code, text = entry(target, plan, pins)
            if code != 0 or 'NOT_SUBMITTABLE runtime_pin=PLACEHOLDER' not in text:
                problems.append('DryRun of %s on the 120bc3c copy: exit %s %s' % (plan, code, text[-600:]))
        code, text = entry(target, 'ggoaldry', pins, dry=False)
        print('TBGOAL_PIN real submission refused=%s' % (code != 0))
        if code == 0 or 'nothing was submitted' not in text:
            problems.append('a real ggoaldry submission was accepted on the 120bc3c copy: %s' % text[-400:])
    # 8. the pin before the engine-stall fix (integrate-1021 9ef0d6d, its receipt): the goal plans refuse it. (perf-tb16-
    # strip: the accepted module is 8ca8511's now, so the copy accepts 9ef0d6d's own module cd131876..., the module of
    # every pin from 9ef0d6d to cf2b28d: the build is then held to the goal plans' requirements, as it was)
    target = copy_harness(work / 'stall')
    path = target / 'controller.py'
    source = path.read_text(encoding='utf-8')
    start = source.index('\nINTERFACE_ACCEPTED = ')
    end = source.index('\n', start + 1)
    path.write_text(source[:start] + "\nINTERFACE_ACCEPTED = 'cd13187612fe4e312b90450c95cc96763e0eed7a50b1d2adf878edd26d2533c7'"
                    + source[end:], encoding='utf-8', newline='\n')
    code, text = run([sys.executable, '-B', 'build.py', '--repo', STALL, '--commit', STALL_COMMIT, '--release-receipt',
                      STALL_RECEIPT], target)
    print('STALL_PIN exit=%s refused=%s' % (code, code != 0 and 'quoted-literal pass' in text))
    if code == 0 or 'quoted-literal pass' not in text:
        problems.append('9ef0d6d (without the quoted-literal fix) was not refused: exit %s %s' % (code, text[-1500:]))
    # 9. the previous pin, thinking-strip 8ca8511 with its receipt (the release smoke, 2026-11-03): PINNED (the same chat
    # module), the release smoke plans that name the vLLM rewriter, --goal-overlap or --async-scheduling blocked by
    # themselves (controller.smoke_block: NOT_SUBMITTABLE blocked_plan), strip25 and smokegh (the Hugging Face fallback
    # names none of them) runnable
    target = copy_harness(work / 'strip')
    accept(target, SMOKE_MODULE)  # (8ca8511's module, accepted in the copy: the accepted one is f5a5c36's now)
    code, text = run([sys.executable, '-B', 'build.py', '--repo', STRIP, '--commit', STRIP_COMMIT, '--release-receipt',
                      STRIP_RECEIPT], target)
    print('STRIP_PIN exit=%s %s' % (code, line(text, 'RUNTIME_PIN=')[:160]))
    if code != 0 or 'RUNTIME_PIN=PINNED' not in text:
        problems.append('8ca8511 did not build a PINNED copy: exit %s %s' % (code, text[-1500:]))
    else:
        repinned = c.load(target / 'pins.json')
        for plan in ('smokeq', 'smokeqs', 'smokeg', 'smokegh', 'strip25'):
            code, text = entry(target, plan, repinned)
            blocked = c.plan_blocked(repinned, plan)
            wanted = plan in ('smokeq', 'smokeqs', 'smokeg')
            refusals = [x for x in text.splitlines() if x.startswith('NOT_SUBMITTABLE')]
            if code != 0 or bool(blocked) is not wanted or refusals != (
                    ['NOT_SUBMITTABLE blocked_plan=%s reason=%s' % (plan, blocked)] if wanted else []):
                problems.append('DryRun of %s on the 8ca8511 copy: exit %s blocked %r %s' % (plan, code, blocked,
                                                                                            text[-600:]))
        print('STRIP_PIN smoke plans blocked: %s' % [p for p in ('smokeq', 'smokeqs', 'smokeg', 'smokegh')
                                                       if c.plan_blocked(repinned, p)])
    # 10. integrate-1103 ea18256 with its receipt (2026-11-09): its chat module is not the accepted one: a PLACEHOLDER,
    # gctx25's DryRun NOT_SUBMITTABLE, a real gctx25 submission refused
    target = copy_harness(work / 'smoke')
    code, text = run([sys.executable, '-B', 'build.py', '--repo', SMOKE, '--commit', SMOKE_COMMIT, '--release-receipt',
                      SMOKE_RECEIPT], target)
    print('SMOKE_PIN exit=%s %s' % (code, line(text, 'RUNTIME_PIN=')[:220]))
    if code != 0 or 'RUNTIME_PIN=PLACEHOLDER' not in text or 'not adapted' not in text:
        problems.append('ea18256 did not build a PLACEHOLDER: exit %s %s' % (code, text[-1500:]))
    else:
        code, text = entry(target, 'gctx25', pins)
        if code != 0 or 'NOT_SUBMITTABLE runtime_pin=PLACEHOLDER' not in text:
            problems.append('DryRun of gctx25 on the ea18256 copy: exit %s %s' % (code, text[-600:]))
        code, text = entry(target, 'gctx25', pins, dry=False)
        print('SMOKE_PIN real gctx25 submission refused=%s' % (code != 0))
        # (since the main ba4d1ba re-pin the block refuses it first: ea18256 has no Terminal-Bench client, client_block,
        # "... Nothing was submitted."; PowerShell wraps the message at the console width: joined before the search)
        if code == 0 or 'nothing was submitted' not in text.replace('\r', '').replace('\n', '').lower():
            problems.append('a real gctx25 submission was accepted on the ea18256 copy: %s' % text[-400:])
    # 11. gemma-tb-262k f5a5c36 with its receipt (gctx25's runtime, job 3560498; its module accepted in the copy): PINNED,
    # its client's Gemma default 262,144: gctx25 runnable; smokeqt / smokeq blocked (no async resolve fix)
    target = copy_harness(work / 'previous-262k')
    accept(target, MAIN_MODULE)
    code, text = run([sys.executable, '-B', 'build.py', '--repo', PREVIOUS_262K, '--commit', PREVIOUS_262K_COMMIT,
                      '--release-receipt', PREVIOUS_262K_RECEIPT], target)
    print('PREVIOUS_262K_PIN exit=%s %s' % (code, line(text, 'RUNTIME_PIN=')[:160]))
    if code != 0 or 'RUNTIME_PIN=PINNED' not in text:
        problems.append('f5a5c36 did not build a PINNED copy: exit %s %s' % (code, text[-1500:]))
    else:
        repinned = c.load(target / 'pins.json')
        if (repinned['runtime_features'].get(c.TB_CLIENT_KEY) or {}).get('context', {}).get(c.GEMMA_MODEL) != 262144 \
                or repinned['runtime_features'].get(c.ASYNC_FIX_KEY) is not None:
            problems.append('f5a5c36: the client %r / the async fixes %r' % (
                repinned['runtime_features'].get(c.TB_CLIENT_KEY), repinned['runtime_features'].get(c.ASYNC_FIX_KEY)))
        for plan, wanted in (('gctx25', None), ('smokeqt', 'async-maxtok-1112'), ('smokeq', 'async-maxtok-1112')):
            code, text = entry(target, plan, repinned)
            blocked = c.plan_blocked(repinned, plan)
            refusals = [x for x in text.splitlines() if x.startswith('NOT_SUBMITTABLE')]
            if code != 0 or bool(blocked) is not bool(wanted) or (wanted and wanted not in blocked) or refusals != (
                    ['NOT_SUBMITTABLE blocked_plan=%s reason=%s' % (plan, blocked)] if wanted else []) \
                    or 'DRY_RUN connections=0' not in text:
                problems.append('DryRun of %s on the f5a5c36 copy: exit %s blocked %r %s' % (plan, code, blocked,
                                                                                            text[-600:]))
        print('PREVIOUS_262K_PIN runnable: gctx25 %s, smokeqt %s' % (not c.plan_blocked(repinned, 'gctx25'),
                                                                     not c.plan_blocked(repinned, 'smokeqt')))
    # 12. the async-fix guard (D-G): recent-fill-1103 f4ccd4b with its receipt (the accepted chat module, neither async
    # fix): PINNED, smokeq / smokeqt blocked by async_fix_block, smokeqs runnable, gctx25 blocked (no client)
    target = copy_harness(work / 'no-async-fix')
    accept(target, MAIN_MODULE)
    code, text = run([sys.executable, '-B', 'build.py', '--repo', NO_FIX, '--commit', NO_FIX_COMMIT, '--release-receipt',
                      NO_FIX_RECEIPT], target)
    print('NO_ASYNC_FIX_PIN exit=%s %s' % (code, line(text, 'RUNTIME_PIN=')[:160]))
    if code != 0 or 'RUNTIME_PIN=PINNED' not in text:
        problems.append('f4ccd4b did not build a PINNED copy: exit %s %s' % (code, text[-1500:]))
    else:
        repinned = c.load(target / 'pins.json')
        if repinned['runtime_features'].get(c.ASYNC_FIX_KEY) is not None:
            problems.append('f4ccd4b reads as having the async fixes: %r' % repinned['runtime_features'][c.ASYNC_FIX_KEY])
        for plan, wanted in (('smokeq', 'async fixes'), ('smokeqt', 'async fixes'), ('smokeqs', None),
                             ('gctx25', 'Terminal-Bench client')):
            code, text = entry(target, plan, repinned)
            blocked = c.plan_blocked(repinned, plan)
            refusals = [x for x in text.splitlines() if x.startswith('NOT_SUBMITTABLE')]
            if code != 0 or bool(blocked) is not bool(wanted) or (wanted and wanted not in blocked) or refusals != (
                    ['NOT_SUBMITTABLE blocked_plan=%s reason=%s' % (plan, blocked)] if wanted else []):
                problems.append('DryRun of %s on the f4ccd4b copy: exit %s blocked %r %s' % (plan, code, blocked,
                                                                                            text[-600:]))
        print('NO_ASYNC_FIX_PIN blocked: %s' % [p for p in ('smokeq', 'smokeqs', 'smokeqt', 'gctx25')
                                                 if c.plan_blocked(repinned, p)])
    # 13. the previous pin, main ba4d1ba with its receipt (its module accepted in the copy): PINNED, smokeq / smokeqt
    # blocked by the async guard (no resolve fix: job 3568311), smokeqs runnable, gctx25 blocked by the client (65,536)
    target = copy_harness(work / 'previous-main')
    accept(target, MAIN_MODULE)
    code, text = run([sys.executable, '-B', 'build.py', '--repo', PREVIOUS_MAIN, '--commit', PREVIOUS_MAIN_COMMIT,
                      '--release-receipt', PREVIOUS_MAIN_RECEIPT], target)
    print('PREVIOUS_MAIN_PIN exit=%s %s' % (code, line(text, 'RUNTIME_PIN=')[:160]))
    if code != 0 or 'RUNTIME_PIN=PINNED' not in text:
        problems.append('ba4d1ba did not build a PINNED copy: exit %s %s' % (code, text[-1500:]))
    else:
        repinned = c.load(target / 'pins.json')
        if repinned['runtime_features'].get(c.ASYNC_FIX_KEY) is not None or c.REGISTRY_KEY in repinned['runtime_features']:
            problems.append('ba4d1ba reads as having the resolve fix or the registry: %r' % (
                repinned['runtime_features'].get(c.ASYNC_FIX_KEY),))
        for plan, wanted in (('smokeq', 'async-maxtok-1112'), ('smokeqt', 'async-maxtok-1112'), ('smokeqs', None),
                             ('gctx25', 'writes the Gemma job at 65536')):
            code, text = entry(target, plan, repinned)
            blocked = c.plan_blocked(repinned, plan)
            refusals = [x for x in text.splitlines() if x.startswith('NOT_SUBMITTABLE')]
            if code != 0 or bool(blocked) is not bool(wanted) or (wanted and wanted not in blocked) or refusals != (
                    ['NOT_SUBMITTABLE blocked_plan=%s reason=%s' % (plan, blocked)] if wanted else []):
                problems.append('DryRun of %s on the ba4d1ba copy: exit %s blocked %r %s' % (plan, code, blocked,
                                                                                            text[-600:]))
        print('PREVIOUS_MAIN_PIN blocked: %s' % [p for p in ('smokeq', 'smokeqs', 'smokeqt', 'gctx25')
                                                 if c.plan_blocked(repinned, p)])
    receipt = json.loads(Path(RECEIPT).read_bytes())
    receipt['log_sha256'] = '0' * 64
    bad = work / 'tampered-receipt.json'
    bad.write_text(json.dumps(receipt), encoding='utf-8')
    target = copy_harness(work / 'tampered-receipt')
    code, text = run([sys.executable, '-B', 'build.py', '--reuse-bundle', '--release-receipt', bad], target)
    print('TAMPERED_RECEIPT exit=%s refused=%s' % (code, code != 0 and 'does not match its log' in text))
    if code == 0 or 'does not match its log' not in text:
        problems.append('a receipt that does not match its log was pinned: exit %s %s' % (code, text[-800:]))
    print('REPIN_%s problems=%d work=%s' % ('PASS' if not problems else 'FAIL', len(problems), work))
    for problem in problems:
        print('  PROBLEM ' + problem[:3000])
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
