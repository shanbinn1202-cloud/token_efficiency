# perf-gemma-swe32 local validation: no connection to DeltaAI or tbench, no job submission, nothing on C:.
# Two builds (determinism), the unit tests, the shared tbench lockset (test_tbench_lockset.py: gswe32 beside swe32,
# swe16, swe10 and tb16), the sealed runtime's own serve.py over every planned server (release parity), the real Gemma 4
# tokenizer (tokenizer venv, REMOTE_RA_GEMMA4_METADATA = the five pinned metadata files), the goal-text probe (the pinned commit's own goals.jsonl writer), the
# gates (every plan's memory / time / cost and the refusals), the re-pin paths in a disposable copy, the synthetic
# stages (e2e_gswe32.py: the real job, tbench scripts and postrun on fakes, every scenario verified, joined, packaged),
# the eval pipeline reading five synthetic verified runs (pipeline_compat.py), the entry script's self-test and dry run
# of every plan plus refusals, the transport self-test, deploy_tbench.ps1 -SelfTest (manifest = pins), fetch_run.ps1
# -SelfTest, postrun\verify_on_tbench.py --self-test (2026-11-12), and the clean-up checks. Logs: validation\<stamp>\. After a re-pin (build.py --commit <hash>) run it
# again: LOCAL_VALIDATION_PASS names the commit.
# Usage: powershell.exe -NoProfile -ExecutionPolicy Bypass -File '<perf-gemma-swe32>\local_validation.ps1' [-Skip e2e,repin,tokenizer]
param([string[]]$Skip = @(), [string]$Python = 'D:\Anaconda\python.exe',
      [string]$TokenizerPython = 'D:\Remote-RA-temp\venvs\remote-ra-tok\Scripts\python.exe',
      [string]$GemmaMetadata = 'D:\Remote-RA-temp\scratch\gemma4-meta')
$ErrorActionPreference = 'Stop'
$Temp = 'D:\Remote-RA-temp\tmp\gswe32'
foreach ($Name in 'TEMP', 'TMP', 'TMPDIR') { [Environment]::SetEnvironmentVariable($Name, $Temp) }
[void](New-Item -ItemType Directory -Force -Path $Temp)
$Here = $PSScriptRoot
$Stamp = Get-Date -Format 'yyyyMMddTHHmmss'
$Logs = Join-Path $Here ('validation\' + $Stamp)
[void](New-Item -ItemType Directory -Force -Path $Logs)
$Scratch = 'D:\Remote-RA-temp\scratch\perf-gemma-swe32'
[void](New-Item -ItemType Directory -Force -Path $Scratch)
$CTemp = Join-Path $env:LOCALAPPDATA 'Temp'
$CBefore = @(Get-ChildItem -LiteralPath $CTemp -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'gswe32*' -or $_.Name -like 'sealed-git*' } | ForEach-Object { $_.FullName })
$Results = [ordered]@{}
function Run-Step([string]$Name, [string]$Log, [scriptblock]$Body) {
    $Started = Get-Date
    $Previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'  # native stderr lines (unittest, PowerShell children) are output, not errors
    $global:LASTEXITCODE = 0
    $Output = & $Body 2>&1 | ForEach-Object { "$_" }
    $Code = $LASTEXITCODE
    $ErrorActionPreference = $Previous
    [IO.File]::WriteAllLines((Join-Path $Logs $Log), [string[]]$Output)
    $Seconds = [math]::Round(((Get-Date) - $Started).TotalSeconds, 1)
    $script:Results[$Name] = @{code = $Code; seconds = $Seconds; log = $Log}
    Write-Output ('STEP ' + $Name + ' exit=' + $Code + ' seconds=' + $Seconds + ' log=' + $Log)
}
function Hash([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
Push-Location $Here
try {
    Run-Step 'build-1' 'build-1.log' { & $Python -B build.py --reuse-bundle }
    $First = @((Hash (Join-Path $Here 'dist\payload.json')), (Hash (Join-Path $Here 'run_one_connection.ps1')), (Hash (Join-Path $Here 'pins.json')))
    Run-Step 'build-2' 'build-2.log' { & $Python -B build.py --reuse-bundle }
    $Second = @((Hash (Join-Path $Here 'dist\payload.json')), (Hash (Join-Path $Here 'run_one_connection.ps1')), (Hash (Join-Path $Here 'pins.json')))
    Run-Step 'deterministic' 'deterministic.log' {
        Write-Output ('payload ' + $First[0] + ' / ' + $Second[0]); Write-Output ('entry ' + $First[1] + ' / ' + $Second[1]); Write-Output ('pins ' + $First[2] + ' / ' + $Second[2])
        $Same = ($First -join ',') -ceq ($Second -join ',')
        Write-Output ('DETERMINISTIC=' + $Same)
        $global:LASTEXITCODE = [int](-not $Same)
    }
    Run-Step 'unit-tests' 'unit-tests.log' { & $Python -B -m unittest test_gswe32 -v }
    Run-Step 'tbench-lockset' 'tbench-lockset.log' { & $Python -B -m unittest test_tbench_lockset -v }
    Run-Step 'release-parity' 'release-parity.log' { & $Python -B -m unittest test_release_parity -v }
    if ($Skip -notcontains 'tokenizer') {
        if (-not $env:REMOTE_RA_GEMMA4_METADATA) { $env:REMOTE_RA_GEMMA4_METADATA = $GemmaMetadata }
        Run-Step 'real-tokenizer' 'real-tokenizer.log' { & $TokenizerPython -B -m unittest test_real_tokenizer -v }
    }
    # goal_text.py against the pinned commit's own Goal-text writer (NOT_AVAILABLE on a commit without one). The 2026-11-09
    # re-pin: a pin whose launcher declares the Goal-text file must give GOAL_TEXT_PROBE=PASS (the runtime's tests moved
    # to tests/; a probe that cannot find them must not pass as NOT_AVAILABLE)
    Run-Step 'goal-text-probe' 'goal-text-probe.log' {
        $ProbeOut = @(& $Python -B goal_text_probe.py 2>&1 | ForEach-Object { "$_" })
        $ProbeCode = $LASTEXITCODE
        $ProbeOut
        $Declared = ([IO.File]::ReadAllText((Join-Path $Here 'pins.json')) | ConvertFrom-Json).runtime_features.goal_text
        if ($ProbeCode -eq 0 -and $Declared -and -not ($ProbeOut | Where-Object { $_ -like 'GOAL_TEXT_PROBE=PASS *' })) { Write-Output 'UNEXPECTED: the pin declares a Goal-text file but the probe did not PASS'; $ProbeCode = 1 }
        $global:LASTEXITCODE = $ProbeCode
    }
    Run-Step 'gate-checks' 'gate-checks.log' { & $Python -B gate_checks.py }
    if ($Skip -notcontains 'repin') {
        Run-Step 'repin' 'repin.log' { & $Python -B test_repin.py --work (Join-Path $Scratch ('repin-' + $Stamp)) }
    }
    $E2E = Join-Path $Scratch ('e2e-' + $Stamp)
    if ($Skip -notcontains 'e2e') {
        Run-Step 'synthetic-stages' 'synthetic-stages.log' { & $Python -B e2e_gswe32.py --work $E2E }
        Run-Step 'pipeline-compat' 'pipeline-compat.log' { & $Python -B pipeline_compat.py --e2e $E2E }
    }
    $Pins = [IO.File]::ReadAllText((Join-Path $Here 'pins.json')) | ConvertFrom-Json
    $Entry = Join-Path $Here 'run_one_connection.ps1'
    Run-Step 'entry' 'entry.log' {
        $Bad = 0
        $Capture = @()
        foreach ($Plan in $Pins.config.plan_order) {
            $Phrase = $Pins.config.plans.$Plan.authorization
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry -Plan $Plan -SelfTest
            if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output ('UNEXPECTED self-test failure: ' + $Plan) }
            $Lines = @(& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry -Plan $Plan -Authorize -Authorization $Phrase -DryRun 2>&1 | ForEach-Object { "$_" })
            $global:LASTEXITCODE = $LASTEXITCODE
            $Lines
            $Capture += $Lines
            $NotReady = [bool]$Pins.config.plans.$Plan.not_ready
            if ($NotReady) {
                # goal until its wave-2 rebuild: self-test only; its dry run (and so its submission) is refused
                if ($LASTEXITCODE -eq 0) { $Bad++; Write-Output ('UNEXPECTED dry run of the NOT READY plan ' + $Plan) } else { Write-Output ('REFUSED as expected: dry run of the NOT READY plan ' + $Plan) }
            } elseif ($LASTEXITCODE -ne 0) { $Bad++; Write-Output ('UNEXPECTED dry-run failure: ' + $Plan) }
            $Lines = @(& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry -Plan $Plan -AfterJob '1234567' -Authorize -Authorization $Phrase -DryRun 2>&1 | ForEach-Object { "$_" })
            $global:LASTEXITCODE = $LASTEXITCODE
            $Lines
            $Capture += $Lines
            if ($NotReady) {
                if ($LASTEXITCODE -eq 0) { $Bad++; Write-Output ('UNEXPECTED dry run of the NOT READY plan with -AfterJob: ' + $Plan) } else { Write-Output ('REFUSED as expected: dry run of the NOT READY plan with -AfterJob ' + $Plan) }
            } elseif ($Pins.config.plans.$Plan.partition -ceq 'ghx4-interactive') {
                # one submitted job per user there: -AfterJob is refused for the interactive variants
                if ($LASTEXITCODE -eq 0) { $Bad++; Write-Output ('UNEXPECTED acceptance of -AfterJob on ghx4-interactive: ' + $Plan) } else { Write-Output ('REFUSED as expected: -AfterJob on ghx4-interactive plan ' + $Plan) }
            } elseif ($LASTEXITCODE -ne 0) { $Bad++; Write-Output ('UNEXPECTED dry-run failure with -AfterJob: ' + $Plan) }
        }
        $Goal = $Pins.config.plans.goal.authorization
        $Refusals = @(
            @('-Plan', 'full', '-Authorize', '-Authorization', $Goal, '-DryRun'),
            @('-Plan', 'goal', '-Authorize', '-Authorization', $Goal.Replace('_BATCH_', '_INTERACTIVE_'), '-DryRun'),
            @('-Plan', 'goal', '-Authorize', '-Authorization', $Goal.Replace('_121MIN_', '_120MIN_'), '-DryRun'),
            @('-Plan', 'goal', '-Authorize', '-Authorization', $Goal.Replace('_GEMMA4_', '_'), '-DryRun'),
            @('-Plan', 'goalb4', '-SelfTest'),
            @('-Plan', 'goal', '-Authorize', '-Authorization', $Goal.ToLowerInvariant(), '-DryRun'),
            @('-Plan', 'goal', '-Authorize', '-Authorization', 'AUTHORIZE_SWE16_GOAL_32TASKS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY', '-DryRun'),
            @('-Plan', 'goal', '-Authorization', $Goal),
            @('-Plan', 'goal', '-AfterJob', 'x7', '-Authorize', '-Authorization', $Goal, '-DryRun'),
            @('-Plan', 'all', '-SelfTest'),
            @('-Plan', 'nope', '-SelfTest'),
            @('-Plan', 'full', '-RunLabel', 'x1', '-SelfTest'),
            @('-Plan', 'full', '-RunLabel', 'r100', '-SelfTest'),
            @('-Plan', 'obsi', '-Authorize', '-Authorization', $Pins.config.plans.obs.authorization, '-DryRun'),
            @('-Plan', 'obs', '-Authorize', '-Authorization', $Pins.config.plans.obsi.authorization, '-DryRun'),
            @('-Plan', 'obs4k', '-Authorize', '-Authorization', $Pins.config.plans.obs.authorization, '-DryRun'),
            @('-Plan', 'obs', '-Authorize', '-Authorization', $Pins.config.plans.obs4k.authorization, '-DryRun'),
            @('-Plan', 'obs4k', '-Authorize', '-Authorization', $Pins.config.plans.obs4k.authorization.Replace('_FORWARD4096_', '_'), '-DryRun'),
            @('-Plan', 'obs4ki', '-SelfTest'),
            @('-Plan', 'goali', '-SelfTest'),
            # the release smoke (2026-11-03): its own phrases, no interactive variant
            @('-Plan', 'smoke', '-Authorize', '-Authorization', $Pins.config.plans.smokehf.authorization, '-DryRun'),
            @('-Plan', 'smokehf', '-Authorize', '-Authorization', $Pins.config.plans.smoke.authorization, '-DryRun'),
            @('-Plan', 'smoke', '-Authorize', '-Authorization', $Pins.config.plans.smoke.authorization.Replace('_153MIN_', '_152MIN_'), '-DryRun'),
            @('-Plan', 'smoke', '-Authorize', '-Authorization', $Pins.config.plans.smoke.authorization.Replace('_BATCH_', '_INTERACTIVE_'), '-DryRun'),
            @('-Plan', 'smoke', '-Authorize', '-Authorization', $Pins.config.plans.smoke.authorization.Replace('_VLLM_REWRITER_', '_'), '-DryRun'),
            @('-Plan', 'smoke', '-Authorize', '-Authorization', $Pins.config.plans.obs4k.authorization, '-DryRun'),
            @('-Plan', 'smokei', '-SelfTest'),
            # the re-pin to main ba4d1ba (2026-11-09): plan smokea (--async-scheduling) has its own phrase
            @('-Plan', 'smokea', '-Authorize', '-Authorization', $Pins.config.plans.smoke.authorization, '-DryRun'),
            @('-Plan', 'smoke', '-Authorize', '-Authorization', $Pins.config.plans.smokea.authorization, '-DryRun'),
            @('-Plan', 'smokea', '-Authorize', '-Authorization', $Pins.config.plans.smokea.authorization.Replace('_ASYNC_', '_'), '-DryRun'),
            @('-Plan', 'smokea', '-Authorize', '-Authorization', $Pins.config.plans.smokea.authorization.Replace('_153MIN_', '_152MIN_'), '-DryRun'),
            @('-Plan', 'smokea', '-Authorize', '-Authorization', $Pins.config.plans.smokea.authorization.Replace('_BATCH_', '_INTERACTIVE_'), '-DryRun'),
            @('-Plan', 'smokeai', '-SelfTest'))
        if (-not $Pins.tasks.pool) { $Refusals += ,@('-Plan', 'full', '-Authorize', '-Authorization', $Pins.config.plans.full.authorization) }
        foreach ($A in $Refusals) {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry @A
            if ($LASTEXITCODE -eq 0) { $Bad++; Write-Output ('UNEXPECTED acceptance: ' + ($A -join ' ')) } else { Write-Output ('REFUSED as expected: ' + ($A -join ' ')) }
        }
        # every plan recorded before the release smoke keeps its DryRun lines (fixtures/gswe32-dryrun-cf2b28d.txt = the cf2b28d
        # validation's entry.log; the stage nonce, payload bytes and pin masked) and its derived values
        # (fixtures/gswe32-plans-cf2b28d.json, plan_fingerprint.py --write before any change, 2026-11-03)
        $CaptureFile = Join-Path $Logs 'dryrun-lines.txt'
        [IO.File]::WriteAllLines($CaptureFile, [string[]]$Capture)
        & $Python -I -B (Join-Path $Here 'plan_fingerprint.py') --dryrun (Join-Path $Here 'fixtures\gswe32-dryrun-cf2b28d.txt') $CaptureFile
        if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output 'UNEXPECTED: a recorded plan''s DryRun line changed' }
        & $Python -I -B (Join-Path $Here 'plan_fingerprint.py') --check (Join-Path $Here 'fixtures\gswe32-plans-cf2b28d.json')
        if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output 'UNEXPECTED: a recorded plan''s derived values changed' }
        # the re-pin to main ba4d1ba (2026-11-09): every plan of the ea18256 state (the frozen perf-gemma-swe32-aff65eeb's
        # record: fixtures/gswe32-plans-ea18256.json, its validation's entry.log as fixtures/gswe32-dryrun-ea18256.txt)
        # keeps its DryRun lines and derived values, except exactly plan smoke's GPU estimate (the vLLM child recalibrated
        # from cg r2 job 3538523: gpu_fit and the plan-table row; its DryRun gpu_peak_mib / gpu_free_mib / gpu_servers)
        & $Python -I -B (Join-Path $Here 'plan_fingerprint.py') --dryrun (Join-Path $Here 'fixtures\gswe32-dryrun-ea18256.txt') $CaptureFile --mask 'smoke:gpu_peak_mib+gpu_free_mib+gpu_servers'
        if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output 'UNEXPECTED: an ea18256-recorded plan''s DryRun line changed beyond smoke''s GPU estimate' }
        & $Python -I -B (Join-Path $Here 'plan_fingerprint.py') --check (Join-Path $Here 'fixtures\gswe32-plans-ea18256.json') --expect 'smoke:gpu_fit,smoke:plan_row'
        if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output 'UNEXPECTED: an ea18256-recorded plan''s derived values changed beyond smoke''s GPU estimate' }
        # 2026-11-12 (the policy registry readers, the evidence push, the re-pin to integrate-1112): every plan of the
        # ba4d1ba state (the frozen perf-gemma-swe32-b3ce5c3d's record: fixtures/gswe32-plans-b3ce5c3d.json, its
        # validation's entry.log as fixtures/gswe32-dryrun-b3ce5c3d.txt) keeps its DryRun lines (commit and the new
        # tbench_push field masked) and its derived values
        & $Python -I -B (Join-Path $Here 'plan_fingerprint.py') --dryrun (Join-Path $Here 'fixtures\gswe32-dryrun-b3ce5c3d.txt') $CaptureFile
        if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output 'UNEXPECTED: a b3ce5c3d-recorded plan''s DryRun line changed' }
        & $Python -I -B (Join-Path $Here 'plan_fingerprint.py') --check (Join-Path $Here 'fixtures\gswe32-plans-b3ce5c3d.json')
        if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output 'UNEXPECTED: a b3ce5c3d-recorded plan''s derived values changed' }
        # 2026-11-14 (the async guard's third fix, async-maxtok-1112 13ee618's resolve): every plan of the e5c93e1e state
        # (the frozen perf-gemma-swe32-e5c93e1e's record: fixtures/gswe32-plans-e5c93e1e.json, its validation's entry.log
        # as fixtures/gswe32-dryrun-e5c93e1e.txt) keeps its DryRun lines and its derived values
        & $Python -I -B (Join-Path $Here 'plan_fingerprint.py') --dryrun (Join-Path $Here 'fixtures\gswe32-dryrun-e5c93e1e.txt') $CaptureFile
        if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output 'UNEXPECTED: an e5c93e1e-recorded plan''s DryRun line changed' }
        & $Python -I -B (Join-Path $Here 'plan_fingerprint.py') --check (Join-Path $Here 'fixtures\gswe32-plans-e5c93e1e.json')
        if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output 'UNEXPECTED: an e5c93e1e-recorded plan''s derived values changed' }
        if (-not ($Capture | Where-Object { $_ -like 'DRY_RUN *' -and $_ -like '* tbench_push=/home/ubuntu/gswe32-verify:120s*' })) { $Bad++; Write-Output 'UNEXPECTED: no DryRun line names the evidence push' }
        Write-Output ('ENTRY_CHECKS bad=' + $Bad)
        $global:LASTEXITCODE = [int]($Bad -ne 0)
    }
    Run-Step 'transport' 'transport.log' { & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'test_transport.ps1') }
    Run-Step 'deploy-selftest' 'deploy-selftest.log' {
        $Out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'tbench\deploy_tbench.ps1') -SelfTest
        $Out
        $Want = 'MANIFEST_SHA256=' + $Pins.tbench_half.manifest_sha256
        if ($Out -contains $Want) { Write-Output ('MANIFEST_EQUALS_PIN ' + $Want); $global:LASTEXITCODE = 0 } else { Write-Output ('MANIFEST_DIFFERS want ' + $Want); $global:LASTEXITCODE = 1 }
    }
    Run-Step 'fetch-selftest' 'fetch-selftest.log' {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'postrun\fetch_run.ps1') -SelfTest
    }
    # 2026-11-12: the tbench verification's local entry point (no connection: inputs, the code tree with the Gemma 4
    # metadata, ssh / scp present)
    Run-Step 'tbench-verify-selftest' 'tbench-verify-selftest.log' {
        & $TokenizerPython -I -B (Join-Path $Here 'postrun\verify_on_tbench.py') --self-test --cache-root (Join-Path $Scratch 'tbench-code-cache')
    }
    Run-Step 'cleanup' 'cleanup.log' {
        $Bad = 0
        # this harness's fakes only (their command lines carry its absolute path): other harnesses' validations may run
        $Left = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($Here) -and $_.CommandLine -match 'fake_server|fake_tbench|e2e_gswe32|relay\.py|run_pool' })
        if ($Left.Count) { $Bad++; Write-Output ('LEFTOVER processes: ' + (($Left | ForEach-Object { [string]$_.ProcessId + ' ' + $_.CommandLine.Substring(0, [math]::Min(160, $_.CommandLine.Length)) }) -join ' | ')) }
        $Caches = @(Get-ChildItem -LiteralPath $Here -Recurse -Force -Directory -Filter '__pycache__')
        if ($Caches.Count) { $Bad++; Write-Output ('__pycache__ present: ' + ($Caches.FullName -join ', ')) }
        foreach ($File in Get-ChildItem -LiteralPath $Here -Recurse -File | Where-Object { $_.Extension -in '.py', '.json', '.ps1', '.sh', '.in', '.md', '.env' -and $_.FullName -notmatch '\\validation\\' -and $_.FullName -notmatch '\\dist\\' }) {
            if ([IO.File]::ReadAllText($File.FullName).Contains("`r")) { $Bad++; Write-Output ('CR line ending: ' + $File.FullName) }
        }
        $Synthetic = @(Get-ChildItem -LiteralPath $Temp -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'gswe32-fake-*' -or $_.Name -like 'gswe32-parity-*' -or $_.Name -like 'gswe32-tok-*' })
        if ($Synthetic.Count) { $Bad++; Write-Output ('synthetic leftovers in TEMP: ' + ($Synthetic.Name -join ', ')) }
        $CAfter = @(Get-ChildItem -LiteralPath $CTemp -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'gswe32*' -or $_.Name -like 'sealed-git*' } | ForEach-Object { $_.FullName })
        $New = @($CAfter | Where-Object { $CBefore -notcontains $_ })
        if ($New.Count) { $Bad++; Write-Output ('new entries on C: ' + ($New -join ', ')) }
        Write-Output ('CLEANUP bad=' + $Bad)
        $global:LASTEXITCODE = [int]($Bad -ne 0)
    }
} finally { Pop-Location }
$Failed = @($Results.Keys | Where-Object { $Results[$_].code -ne 0 })
if ($Pins.runtime_features.goal_text) { $GoalText = (@($Pins.runtime_features.goal_text.files) -join ',') } else { $GoalText = 'NOT_AVAILABLE' }
$Summary = [ordered]@{stamp = $Stamp; commit = $Pins.commit; payload_sha256 = $Second[0]; entry_script_sha256 = $Second[1]; pins_sha256 = $Second[2]; manifest_sha256 = $Pins.tbench_half.manifest_sha256; goal_text = $GoalText; steps = $Results; failed = $Failed}
[IO.File]::WriteAllText((Join-Path $Logs 'summary.json'), ($Summary | ConvertTo-Json -Depth 5))
if ($Failed.Count) { Write-Output ('LOCAL_VALIDATION_FAIL failed=' + ($Failed -join ',') + ' commit=' + $Pins.commit + ' logs=' + $Logs); exit 1 }
Write-Output ('LOCAL_VALIDATION_PASS commit=' + $Pins.commit + ' goal_text=' + $GoalText + ' payload=' + $Second[0] + ' logs=' + $Logs)
