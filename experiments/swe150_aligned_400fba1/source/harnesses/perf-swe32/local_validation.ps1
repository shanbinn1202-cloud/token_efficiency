# perf-swe32 local validation: no connection to DeltaAI or tbench, no job submission, nothing on C:.
# Two builds (determinism), the unit tests, the sealed runtime's own serve.py over every planned server (release
# parity), the real tokenizer (tokenizer venv), the goal-text probe (the pinned commit's own goals.jsonl writer), the
# gates (every plan's memory / time / cost and the refusals), the re-pin paths in a disposable copy, the synthetic
# stages (e2e_swe32.py: the real job, tbench scripts and postrun on fakes, every scenario verified, joined, packaged),
# the eval pipeline reading five synthetic verified runs (pipeline_compat.py), the entry script's self-test and dry run
# of every plan plus refusals, the transport self-test, deploy_tbench.ps1 -SelfTest (manifest = pins), fetch_run.ps1
# -SelfTest, and the clean-up checks. Logs: validation\<stamp>\. After a re-pin (build.py --commit <hash>) run it
# again: LOCAL_VALIDATION_PASS names the commit.
# Usage: powershell.exe -NoProfile -ExecutionPolicy Bypass -File '<perf-swe32>\local_validation.ps1' [-Skip e2e,repin,tokenizer] [-Jobs 6]
# -Jobs N (2026-11-10, default 6): the re-pin groups, the synthetic stages and the pipeline's fetches run N at a time
# (test_repin.py / e2e_swe32.py / pipeline_compat.py --jobs; each e2e case's transcript in validation\<stamp>\e2e\<case>.log
# and, in the usual order, in synthetic-stages.log). -Jobs 1 runs them one after the other as before.
# 2026-11-16 (the user's decision): after the two builds, synthetic-stages runs in the background (a PowerShell job)
# while the steps that do not read its output run one after the other (unit tests, release parity, the real
# tokenizer, the goal-text probe, the gates, the re-pin paths, the entry script, transport, deploy and fetch
# self-tests); then the job is awaited (its STEP line, log and exit code as before) and pipeline-compat (it reads the
# synthetic stages' archives) and the clean-up run. The STEP / LOCAL_VALIDATION_* lines and summary.json keep their
# format; a failing step, in the background or not, fails the validation and keeps its log.
param([string[]]$Skip = @(), [string]$Python = 'D:\Anaconda\python.exe',
      [string]$TokenizerPython = 'D:\Remote-RA-temp\venvs\remote-ra-tok\Scripts\python.exe', [int]$Jobs = 6)
$ErrorActionPreference = 'Stop'
$Temp = 'D:\Remote-RA-temp\tmp\swe32'
foreach ($Name in 'TEMP', 'TMP', 'TMPDIR') { [Environment]::SetEnvironmentVariable($Name, $Temp) }
[void](New-Item -ItemType Directory -Force -Path $Temp)
$Here = $PSScriptRoot
$Stamp = Get-Date -Format 'yyyyMMddTHHmmss'
$Logs = Join-Path $Here ('validation\' + $Stamp)
[void](New-Item -ItemType Directory -Force -Path $Logs)
$Scratch = 'D:\Remote-RA-temp\scratch\perf-swe32'
[void](New-Item -ItemType Directory -Force -Path $Scratch)
$CTemp = Join-Path $env:LOCALAPPDATA 'Temp'
$CBefore = @(Get-ChildItem -LiteralPath $CTemp -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'swe32*' -or $_.Name -like 'sealed-git*' } | ForEach-Object { $_.FullName })
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
function Start-Background([string]$Name, [string]$Log, [string[]]$Command) {
    # one step in a PowerShell job (its own process, this script's TEMP / TMP / TMPDIR and directory); Wait-Background
    # writes its log and STEP line exactly as Run-Step does
    $Job = Start-Job -ScriptBlock {
        param($Dir, $TempDir, $Exe, $Arguments)
        foreach ($N in 'TEMP', 'TMP', 'TMPDIR') { [Environment]::SetEnvironmentVariable($N, $TempDir) }
        Set-Location -LiteralPath $Dir
        $ErrorActionPreference = 'Continue'
        $global:LASTEXITCODE = 0
        & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
        '__STEP_EXIT__=' + $LASTEXITCODE
    } -ArgumentList $Here, $Temp, $Command[0], ([string[]]$Command[1..($Command.Count - 1)])
    return @{name = $Name; log = $Log; job = $Job; started = (Get-Date)}
}
function Wait-Background($Step) {
    [void](Wait-Job -Job $Step.job)
    $Output = @(Receive-Job -Job $Step.job 2>&1 | ForEach-Object { "$_" })
    $State = $Step.job.State
    Remove-Job -Job $Step.job -Force
    $Last = @($Output | Where-Object { $_ -like '__STEP_EXIT__=*' } | Select-Object -Last 1)
    $Code = if ($Last.Count) { [int]$Last[0].Substring(14) } else { 1 }  # no exit line: the job itself failed
    if ($State -ne 'Completed' -and $Code -eq 0) { $Code = 1 }
    $Output = @($Output | Where-Object { $_ -notlike '__STEP_EXIT__=*' })
    if (-not $Last.Count) { $Output += ('BACKGROUND_JOB_STATE=' + $State) }
    [IO.File]::WriteAllLines((Join-Path $Logs $Step.log), [string[]]$Output)
    $Seconds = [math]::Round(((Get-Date) - $Step.started).TotalSeconds, 1)
    $script:Results[$Step.name] = @{code = $Code; seconds = $Seconds; log = $Step.log}
    Write-Output ('STEP ' + $Step.name + ' exit=' + $Code + ' seconds=' + $Seconds + ' log=' + $Step.log)
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
    # 2026-11-16: the synthetic stages in the background (see the header)
    $E2E = Join-Path $Scratch ('e2e-' + $Stamp)
    $Synthetic = $null
    if ($Skip -notcontains 'e2e') {
        $Synthetic = Start-Background 'synthetic-stages' 'synthetic-stages.log' @($Python, '-B', 'e2e_swe32.py', '--work', $E2E,
                                                                                  '--jobs', [string]$Jobs, '--logs', (Join-Path $Logs 'e2e'))
    }
    Run-Step 'unit-tests' 'unit-tests.log' { & $Python -B -m unittest test_swe32 -v }
    Run-Step 'release-parity' 'release-parity.log' { & $Python -B -m unittest test_release_parity -v }
    if ($Skip -notcontains 'tokenizer') {
        Run-Step 'real-tokenizer' 'real-tokenizer.log' { & $TokenizerPython -B -m unittest test_real_tokenizer -v }
    }
    # goal_text.py against the pinned commit's own Goal-text writer (NOT_AVAILABLE on a commit without one)
    Run-Step 'goal-text-probe' 'goal-text-probe.log' { & $Python -B goal_text_probe.py }
    Run-Step 'gate-checks' 'gate-checks.log' { & $Python -B gate_checks.py }
    if ($Skip -notcontains 'repin') {
        Run-Step 'repin' 'repin.log' { & $Python -B test_repin.py --work (Join-Path $Scratch ('repin-' + $Stamp)) --jobs $Jobs }
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
            $Code = $LASTEXITCODE
            $Lines
            $Capture += ('== ' + $Plan); $Capture += $Lines
            if ($Code -ne 0) { $Bad++; Write-Output ('UNEXPECTED dry-run failure: ' + $Plan) }
            # a plan with its own settings (ctx24 / ctx25): NOT_SUBMITTABLE exactly while it is blocked (the plan table)
            $Blocked = [bool]($Pins.config.plans.$Plan.server -and @($Pins.runtime_features.qwen_kv_cache_gib) -notcontains [int]$Pins.config.plans.$Plan.server.kv_cache_gib)
            # 2026-11-09 (D-G): an async plan on a pin without the async fixes (runtime_features.async_fix) is blocked too
            if ($Pins.config.plans.$Plan.async_scheduling -eq $true -and -not $Pins.runtime_features.async_fix.resolve) { $Blocked = $true }  # 2026-11-16: incl. 13ee618
            # 2026-11-16 (later): a scripted edge-case plan on a pin without keep-rebuild-1116
            if (@($Pins.config.plans.$Plan.servers | Where-Object { $_.client }).Count -gt 0 -and -not $Pins.runtime_features.keep_rebuild) { $Blocked = $true }
            # 2026-11-16 (phase B): a plan that names goal_rollback on a pin without goal-rollback-1116
            if ($Pins.config.plans.$Plan.goal_rollback -eq $true -and -not $Pins.runtime_features.goal_rollback) { $Blocked = $true }
            if ($Blocked -ne [bool](@($Lines | Where-Object { $_ -like 'NOT_SUBMITTABLE blocked_plan=*' }).Count)) { $Bad++; Write-Output ('UNEXPECTED NOT_SUBMITTABLE state of ' + $Plan + ' (blocked=' + $Blocked + ')') }
            if ($Blocked) {
                # a real (non-dry) submission of a blocked plan is refused before any connection
                $Out = @(& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry -Plan $Plan -Authorize -Authorization $Phrase 2>&1 | ForEach-Object { "$_" })
                if ($LASTEXITCODE -eq 0 -or -not ($Out -join ' ').Contains('is blocked') -or ($Out -join ' ').Contains('PLANNED_STAGE')) { $Bad++; Write-Output ('UNEXPECTED: blocked plan ' + $Plan + ' not refused: ' + ($Out -join ' ')) } else { Write-Output ('REFUSED as expected: a real submission of blocked plan ' + $Plan) }
            }
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry -Plan $Plan -AfterJob '1234567' -Authorize -Authorization $Phrase -DryRun
            if ($Pins.config.plans.$Plan.partition -ceq 'ghx4-interactive') {
                # an interactive plan goes first in a tbench chain: -AfterJob is refused
                if ($LASTEXITCODE -eq 0) { $Bad++; Write-Output ('UNEXPECTED acceptance of -AfterJob on interactive plan ' + $Plan) } else { Write-Output ('REFUSED as expected: -AfterJob on interactive plan ' + $Plan) }
            } elseif ($LASTEXITCODE -ne 0) { $Bad++; Write-Output ('UNEXPECTED dry-run failure with -AfterJob: ' + $Plan) }
        }
        $Goal = $Pins.config.plans.goalb4.authorization
        $GoalI = $Pins.config.plans.goalb4i.authorization
        $Minutes = '_' + [int]([TimeSpan]::Parse($Pins.config.plans.goalb4.time).TotalMinutes) + 'MIN_'
        $Refusals = @(
            @('-Plan', 'full', '-Authorize', '-Authorization', $Goal, '-DryRun'),
            @('-Plan', 'goalb4', '-Authorize', '-Authorization', $Goal.Replace('_BATCH_', '_INTERACTIVE_'), '-DryRun'),
            @('-Plan', 'goalb4', '-Authorize', '-Authorization', $GoalI, '-DryRun'),
            @('-Plan', 'goalb4i', '-Authorize', '-Authorization', $Goal, '-DryRun'),
            @('-Plan', 'goalb4', '-Authorize', '-Authorization', $Goal.Replace($Minutes, '_120MIN_'), '-DryRun'),
            @('-Plan', 'goalb4', '-Authorize', '-Authorization', $Goal.ToLowerInvariant(), '-DryRun'),
            @('-Plan', 'goalb4', '-Authorize', '-Authorization', 'AUTHORIZE_SWE16_GOAL_32TASKS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY', '-DryRun'),
            @('-Plan', 'goalb4', '-Authorization', $Goal),
            @('-Plan', 'goalb4', '-AfterJob', 'x7', '-Authorize', '-Authorization', $Goal, '-DryRun'),
            @('-Plan', 'all', '-SelfTest'),
            @('-Plan', 'nope', '-SelfTest'),
            @('-Plan', 'full', '-RunLabel', 'x1', '-SelfTest'),
            @('-Plan', 'full', '-RunLabel', 'r100', '-SelfTest'))
        if (-not $Pins.tasks.pool) { $Refusals += ,@('-Plan', 'full', '-Authorize', '-Authorization', $Pins.config.plans.full.authorization) }
        foreach ($A in $Refusals) {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry @A
            if ($LASTEXITCODE -eq 0) { $Bad++; Write-Output ('UNEXPECTED acceptance: ' + ($A -join ' ')) } else { Write-Output ('REFUSED as expected: ' + ($A -join ' ')) }
        }
        # every plan recorded before ctx24 / ctx25 keeps its DryRun line (fixtures/swe32-dryrun-4f5f4e7.txt; the stage
        # nonce, payload bytes, commit and receipt masked)
        $CaptureFile = Join-Path $Logs 'dryrun-lines.txt'
        [IO.File]::WriteAllLines($CaptureFile, [string[]]$Capture)
        & $Python -I -B (Join-Path $Here 'plan_fingerprint.py') --dryrun (Join-Path $Here 'fixtures\swe32-dryrun-4f5f4e7.txt') $CaptureFile
        if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output 'UNEXPECTED: a recorded plan''s DryRun line changed' }
        # 2026-11-16 (phase B, a2f9e36): with 51b4085's registry rows (a2f9e36's adds the marker plug-ins: every runtime
        # server's rewriter / parking lists gain goal_query_markers, nothing else; test_swe32 Ctx262 checks that split)
        & $Python -I -B (Join-Path $Here 'plan_fingerprint.py') --check (Join-Path $Here 'fixtures\swe32-plans-4f5f4e7.json') --registry (Join-Path $Here 'fixtures\policy-registry-51b4085.json') --drop-feature goal_rollback
        if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output 'UNEXPECTED: a recorded plan''s derived values changed' }
        Write-Output ('ENTRY_CHECKS bad=' + $Bad)
        $global:LASTEXITCODE = [int]($Bad -ne 0)
    }
    Run-Step 'transport' 'transport.log' { & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'test_transport.ps1') }
    Run-Step 'deploy-selftest' 'deploy-selftest.log' {
        $Out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'tbench\deploy_tbench.ps1') -SelfTest
        $Out
        $Want = 'MANIFEST_SHA256=' + $Pins.tbench_half.manifest_sha256
        if ($Out -contains $Want) { Write-Output ('MANIFEST_EQUALS_PIN ' + $Want); $Code = 0 } else { Write-Output ('MANIFEST_DIFFERS want ' + $Want); $Code = 1 }
        # 2026-11-16: the keep-reasoning agents' model class (outside the manifest) equals its config pin
        $Ext = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'tbench\agent_ext\deploy_agent_ext.ps1') -SelfTest
        $Ext
        if (-not ($Ext -contains 'DEPLOY_AGENT_EXT_SELFTEST_PASS connections=0')) { $Code = 1 }
        $global:LASTEXITCODE = $Code
    }
    Run-Step 'fetch-selftest' 'fetch-selftest.log' {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'postrun\fetch_run.ps1') -SelfTest
    }
    if ($null -ne $Synthetic) {
        Wait-Background $Synthetic
        $Synthetic = $null
        Run-Step 'pipeline-compat' 'pipeline-compat.log' { & $Python -B pipeline_compat.py --e2e $E2E --jobs $Jobs }
    }
    Run-Step 'cleanup' 'cleanup.log' {
        $Bad = 0
        # this harness's fakes only (their command lines carry its absolute path): other harnesses' validations may run
        $Left = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($Here) -and $_.CommandLine -match 'fake_server|fake_tbench|e2e_swe32|relay\.py|run_pool' })
        if ($Left.Count) { $Bad++; Write-Output ('LEFTOVER processes: ' + (($Left | ForEach-Object { [string]$_.ProcessId + ' ' + $_.CommandLine.Substring(0, [math]::Min(160, $_.CommandLine.Length)) }) -join ' | ')) }
        $Caches = @(Get-ChildItem -LiteralPath $Here -Recurse -Force -Directory -Filter '__pycache__')
        if ($Caches.Count) { $Bad++; Write-Output ('__pycache__ present: ' + ($Caches.FullName -join ', ')) }
        foreach ($File in Get-ChildItem -LiteralPath $Here -Recurse -File | Where-Object { $_.Extension -in '.py', '.json', '.ps1', '.sh', '.in', '.md', '.env' -and $_.FullName -notmatch '\\validation\\' -and $_.FullName -notmatch '\\dist\\' }) {
            if ([IO.File]::ReadAllText($File.FullName).Contains("`r")) { $Bad++; Write-Output ('CR line ending: ' + $File.FullName) }
        }
        $Synthetic = @(Get-ChildItem -LiteralPath $Temp -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'swe32-fake-*' -or $_.Name -like 'swe32-parity-*' -or $_.Name -like 'swe32-tok-*' })
        if ($Synthetic.Count) { $Bad++; Write-Output ('synthetic leftovers in TEMP: ' + ($Synthetic.Name -join ', ')) }
        $CAfter = @(Get-ChildItem -LiteralPath $CTemp -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'swe32*' -or $_.Name -like 'sealed-git*' } | ForEach-Object { $_.FullName })
        $New = @($CAfter | Where-Object { $CBefore -notcontains $_ })
        if ($New.Count) { $Bad++; Write-Output ('new entries on C: ' + ($New -join ', ')) }
        Write-Output ('CLEANUP bad=' + $Bad)
        $global:LASTEXITCODE = [int]($Bad -ne 0)
    }
} finally {
    if ($null -ne $Synthetic) {  # an error above: the background step still ends, writes its log and counts
        Wait-Background $Synthetic
    }
    Pop-Location
}
$Failed = @($Results.Keys | Where-Object { $Results[$_].code -ne 0 })
if ($Pins.runtime_features.goal_text) { $GoalText = (@($Pins.runtime_features.goal_text.files) -join ',') } else { $GoalText = 'NOT_AVAILABLE' }
$Summary = [ordered]@{stamp = $Stamp; jobs = $Jobs; commit = $Pins.commit; payload_sha256 = $Second[0]; entry_script_sha256 = $Second[1]; pins_sha256 = $Second[2]; manifest_sha256 = $Pins.tbench_half.manifest_sha256; goal_text = $GoalText; steps = $Results; failed = $Failed}
[IO.File]::WriteAllText((Join-Path $Logs 'summary.json'), ($Summary | ConvertTo-Json -Depth 5))
if ($Failed.Count) { Write-Output ('LOCAL_VALIDATION_FAIL failed=' + ($Failed -join ',') + ' commit=' + $Pins.commit + ' logs=' + $Logs); exit 1 }
Write-Output ('LOCAL_VALIDATION_PASS commit=' + $Pins.commit + ' goal_text=' + $GoalText + ' payload=' + $Second[0] + ' logs=' + $Logs)
