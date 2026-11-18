# perf-mab (pmab) local validation: no connection to DeltaAI, no job submission, nothing on C:.
# Two builds (determinism), the unit tests, the delivered payload as the stage imports it, the sealed runtime's own
# serve.py and kv-session protocol module against the harness (release parity), the runner with the real tokenizer
# (tokenizer venv; the Gemma 4 client on the pinned Gemma metadata, REMOTE_RA_GEMMA4_METADATA), the verifier and the
# tables with forgeries, the gates, the entry script's self-test / dry run of every plan (Qwen g1, g2, g4o, g4g, g4v;
# Gemma gm1, gm4o, gm4g, gm4v, gm4r and the interactive gm4oi, gm4gi, gm4ri; the release smoke plans smq, smg and, since
# the main ba4d1ba re-pin, smqt (kv-session thinking on) and smga (Gemma async); 11-15 smqc) and its refusals, the re-pin tests
# (test_repin, integrate-1112 4ebff97: the previous pin main ba4d1ba has no policy registry and the async resolve of
# job 3568311, so it refuses every plan and, with the registry supplied, every async plan, decision D-G), the fake-SSH transport, the fetch
# self-test, the synthetic stages (g1, g2, replay, forge, norelease, crash, squeeze, skip, main, real, realgoal,
# realvanilla; gm1, gmreplay, gmreal, gmrealgoal, gmrealvanilla, gmrealrfill, gmrealgoali, gmhandoff; the release smoke
# smoke = smq, gmsmoke = smg, smoket = smqt, gmsmokea = smga; 11-15 plan smqc, the async-maxtok case cell: smokecut
# PMAB_PASS_VERIFIED, smokecutmiss (no request in flight at any cut) PMAB_FAIL_VERIFIED by async-cut alone) verified
# and turned into tables, the fetch script on a synthetic archive, the tbench verification (11-14:
# postrun\verify_on_tbench.py's self-test, its tests among the postrun tests, and the fetch step's archives verified on a
# local stand-in root with --with-bulk and compared with the fetch result), and the clean-up checks (processes, bytecode,
# line endings, temporary files, no private data).
# Logs: validation\<stamp>\.
# Usage: powershell.exe -NoProfile -ExecutionPolicy Bypass -File '<perf-mab>\local_validation.ps1' [-Skip synthetic,tokenizer]
param([string[]]$Skip = @(), [string]$Python = 'D:\Anaconda\python.exe',
      [string]$TokenizerPython = 'D:\Remote-RA-temp\venvs\remote-ra-tok\Scripts\python.exe',
      [string]$GemmaMetadata = 'D:\Remote-RA-temp\scratch\gemma4-meta')
$ErrorActionPreference = 'Stop'
$Temp = 'D:\Remote-RA-temp\tmp\perf-mab'
foreach ($Name in 'TEMP', 'TMP', 'TMPDIR') { [Environment]::SetEnvironmentVariable($Name, $Temp) }
[Environment]::SetEnvironmentVariable('PYTHONDONTWRITEBYTECODE', '1')
[Environment]::SetEnvironmentVariable('REMOTE_RA_GEMMA4_METADATA', $GemmaMetadata)
[void](New-Item -ItemType Directory -Force -Path $Temp)
$Here = $PSScriptRoot
$Stamp = Get-Date -Format 'yyyyMMddTHHmmss'
$RunStart = Get-Date
$Logs = Join-Path $Here ('validation\' + $Stamp)
[void](New-Item -ItemType Directory -Force -Path $Logs)
$Scratch = 'D:\Remote-RA-temp\scratch\perf-mab\lv-' + $Stamp
[void](New-Item -ItemType Directory -Force -Path $Scratch)
$CTemp = Join-Path $env:LOCALAPPDATA 'Temp'
$CBefore = @(Get-ChildItem -LiteralPath $CTemp -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'pmab*' -or $_.Name -like 'sealed-git*' } | ForEach-Object { $_.FullName })
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
    Run-Step 'unit-tests' 'unit-tests.log' { & $Python -B -m unittest test_mab -v }
    Run-Step 'entry-points' 'entry-points.log' { & $Python -B -m unittest test_entry_points -v }
    Run-Step 'release-parity' 'release-parity.log' { & $Python -B -m unittest test_release_parity -v }
    Run-Step 'repin-tests' 'repin-tests.log' { & $Python -B -m unittest test_repin -v }
    if ($Skip -notcontains 'tokenizer') {
        Run-Step 'real-tokenizer' 'real-tokenizer.log' { & $TokenizerPython -B -m unittest test_real_tokenizer -v }
    }
    Run-Step 'postrun-tests' 'postrun-tests.log' { & $Python -B -m unittest discover -s postrun -p 'test_*.py' -v }
    Run-Step 'grade-tests' 'grade-tests.log' { & $Python -B -m unittest test_mab_grade -v }
    Run-Step 'gate-checks' 'gate-checks.log' { & $Python -B gate_checks.py }
    $Pins = [IO.File]::ReadAllText((Join-Path $Here 'pins.json')) | ConvertFrom-Json
    $Entry = Join-Path $Here 'run_one_connection.ps1'
    Run-Step 'entry' 'entry.log' {
        $Bad = 0
        foreach ($Plan in $Pins.config.plan_order) {
            $Phrase = $Pins.config.plans.$Plan.authorization
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry -Plan $Plan -SelfTest
            if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output ('UNEXPECTED self-test failure: ' + $Plan) }
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry -Plan $Plan -Authorize -Authorization $Phrase -DryRun
            if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output ('UNEXPECTED dry-run failure: ' + $Plan) }
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry -Plan $Plan
            if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output ('UNEXPECTED local-only failure: ' + $Plan) }
        }
        $G1 = $Pins.config.plans.g1.authorization
        $G2 = $Pins.config.plans.g2.authorization
        $Refusals = @(
            @('-SelfTest'),
            @('-Plan', 'g1', '-Authorize', '-Authorization', $G2, '-DryRun'),
            @('-Plan', 'g2', '-Authorize', '-Authorization', $G1, '-DryRun'),
            @('-Authorize', '-Authorization', $G1, '-DryRun'),
            @('-Plan', 'g1', '-Authorization', $G1),
            @('-Plan', 'g2', '-Authorize', '-Authorization', ($G2 -replace '_[0-9]+MIN_', '_120MIN_'), '-DryRun'),
            @('-Plan', 'g1', '-Authorize', '-Authorization', $G1.Replace('_BATCH_', '_INTERACTIVE_'), '-DryRun'),
            @('-Plan', 'g1', '-Authorize', '-Authorization', $G1.ToLowerInvariant(), '-DryRun'),
            @('-Plan', 'g1', '-Authorize', '-Authorization', 'AUTHORIZE_SPD2_DECODEB_FASTPATH2_VS_STOCK_ITL_BATCH_REUSED_V9_ENV_ONE_GH200_195MIN_NO_RETRY', '-DryRun'),
            @('-Plan', 'g3', '-SelfTest'),
            @('-Plan', 'g1', '-RunLabel', 'R1', '-SelfTest'),
            @('-Plan', 'g2', '-RunLabel', 'r100', '-SelfTest'),
            @('-Plan', 'gm1', '-Authorize', '-Authorization', $G1, '-DryRun'),
            @('-Plan', 'gm4o', '-Authorize', '-Authorization', $Pins.config.plans.g4o.authorization, '-DryRun'),
            @('-Plan', 'gm4v', '-Authorize', '-Authorization', ($Pins.config.plans.gm4v.authorization -replace '_[0-9]+MIN_', '_87MIN_'), '-DryRun'),
            @('-Plan', 'gm4g', '-Authorize', '-Authorization', $Pins.config.plans.gm4g.authorization.Replace('_BATCH_', '_INTERACTIVE_'), '-DryRun'),
            # integrate-1021: the interactive variants take their own phrase only (and never the batch plan's)
            @('-Plan', 'gm4oi', '-Authorize', '-Authorization', $Pins.config.plans.gm4o.authorization, '-DryRun'),
            @('-Plan', 'gm4o', '-Authorize', '-Authorization', $Pins.config.plans.gm4oi.authorization, '-DryRun'),
            @('-Plan', 'gm4gi', '-Authorize', '-Authorization', $Pins.config.plans.gm4gi.authorization.Replace('_INTERACTIVE_', '_BATCH_'), '-DryRun'),
            @('-Plan', 'gm4ri', '-Authorize', '-Authorization', ($Pins.config.plans.gm4ri.authorization -replace '_[0-9]+MIN_', '_120MIN_'), '-DryRun'),
            # integrate-1103: the release smoke plans take their own phrase only
            @('-Plan', 'smq', '-Authorize', '-Authorization', $Pins.config.plans.smg.authorization, '-DryRun'),
            @('-Plan', 'smg', '-Authorize', '-Authorization', ($Pins.config.plans.smg.authorization -replace '_[0-9]+MIN_', '_121MIN_'), '-DryRun'),
            @('-Plan', 'smq', '-Authorize', '-Authorization', $Pins.config.plans.smq.authorization.Replace('_BATCH_', '_INTERACTIVE_'), '-DryRun'),
            # main ba4d1ba re-pin: smqt / smga take their own phrase only (never smq's / smg's)
            @('-Plan', 'smqt', '-Authorize', '-Authorization', $Pins.config.plans.smq.authorization, '-DryRun'),
            @('-Plan', 'smq', '-Authorize', '-Authorization', $Pins.config.plans.smqt.authorization, '-DryRun'),
            @('-Plan', 'smga', '-Authorize', '-Authorization', $Pins.config.plans.smg.authorization, '-DryRun'),
            @('-Plan', 'smga', '-Authorize', '-Authorization', ($Pins.config.plans.smga.authorization -replace '_[0-9]+MIN_', '_121MIN_'), '-DryRun'),
            # 11-15: plan smqc (the async-maxtok case cell) takes its own phrase only (never smqt's, nor smqt its)
            @('-Plan', 'smqc', '-Authorize', '-Authorization', $Pins.config.plans.smqt.authorization, '-DryRun'),
            @('-Plan', 'smqt', '-Authorize', '-Authorization', $Pins.config.plans.smqc.authorization, '-DryRun'),
            @('-Plan', 'smqc', '-Authorize', '-Authorization', ($Pins.config.plans.smqc.authorization -replace '_[0-9]+MIN_', '_45MIN_'), '-DryRun'))
        foreach ($A in $Refusals) {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry @A
            if ($LASTEXITCODE -eq 0) { $Bad++; Write-Output ('UNEXPECTED acceptance: ' + ($A -join ' ')) } else { Write-Output ('REFUSED as expected: ' + ($A -join ' ')) }
        }
        Write-Output ('ENTRY_CHECKS bad=' + $Bad)
        $global:LASTEXITCODE = [int]($Bad -ne 0)
    }
    Run-Step 'transport' 'transport.log' { & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'test_transport.ps1') }
    Run-Step 'fetch-selftest' 'fetch-selftest.log' { & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'postrun\fetch_and_verify.ps1') -SelfTest }
    Run-Step 'tbench-selftest' 'tbench-selftest.log' { & $Python -I -B (Join-Path $Here 'postrun\verify_on_tbench.py') --self-test --cache-root (Join-Path $Scratch 'tbench-cache') }
    if ($Skip -notcontains 'synthetic') {
        foreach ($Case in 'g1', 'g2', 'replay', 'forge', 'norelease', 'crash', 'squeeze', 'skip', 'main', 'real', 'realgoal', 'realvanilla', 'gm1', 'gmreplay', 'gmreal', 'gmrealgoal', 'gmrealvanilla', 'gmrealrfill', 'gmrealgoali', 'gmhandoff', 'smoke', 'gmsmoke', 'smoket', 'gmsmokea', 'smokecut', 'smokecutmiss') {
            $Keep = @(); if ($Case -ceq 'g1') { $Keep = @('--keep', (Join-Path $Scratch 'keep')) }
            Run-Step ('synthetic-' + $Case) ('synthetic-' + $Case + '.log') { & $Python -B synthetic_stage.py --case $Case @Keep }
        }
        Run-Step 'fetch-local' 'fetch-local.log' {
            $Archive = Get-ChildItem -LiteralPath (Join-Path $Scratch 'keep') -Recurse -Filter 'pmab-core.tar.xz' | Select-Object -First 1
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'postrun\fetch_and_verify.ps1') -Stage ('/u/ycao11/pmab-' + ('b' * 32)) -JobId '3999999' -LocalArchive $Archive.FullName -SyntheticPins -SlurmState 'COMPLETED' -SlurmExitCode '0:0' -OutRoot (Join-Path $Scratch 'fetch')
        }
        # 11-14: the same archive and its bulk archives (as the fetch step takes them from next to it) through the fetch
        # script with -RemoteArchiveSha256 (as a real job's fetch: the archive-sha256 check) and through
        # postrun\verify_on_tbench.py --with-bulk on a local stand-in for the tbench root (the code tree, the uploads, the
        # detached run, the poll, the pull), then the two folders compared: the same verdict and exit code, the same
        # check rows, the same tables byte for byte, the same extracted files (check\files and check\bulk).
        Run-Step 'tbench-local' 'tbench-local.log' {
            $Archive = Get-ChildItem -LiteralPath (Join-Path $Scratch 'keep') -Recurse -Filter 'pmab-core.tar.xz' | Select-Object -First 1
            $Sha = Hash $Archive.FullName
            $Bulk = @(); foreach ($Item in Get-ChildItem -LiteralPath $Archive.DirectoryName -Filter 'pmab-bulk-*.tar.xz') { $Bulk += @('--local-bulk', $Item.FullName) }
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'postrun\fetch_and_verify.ps1') -Stage ('/u/ycao11/pmab-' + ('b' * 32)) -JobId '3999999' -LocalArchive $Archive.FullName -RemoteArchiveSha256 $Sha -SyntheticPins -SlurmState 'COMPLETED' -SlurmExitCode '0:0' -OutRoot (Join-Path $Scratch 'fetch-sha')
            $Fetch = $LASTEXITCODE
            & $Python -I -B (Join-Path $Here 'postrun\verify_on_tbench.py') --stage ('/u/ycao11/pmab-' + ('b' * 32)) --job-id '3999999' --local-root (Join-Path $Scratch 'tbench-root') --local-core $Archive.FullName --core-sha256 $Sha --with-bulk @Bulk --synthetic --slurm-state 'COMPLETED' --slurm-exit-code '0:0' --out-root (Join-Path $Scratch 'tbench-out') --cache-root (Join-Path $Scratch 'tbench-cache')
            $Verify = $LASTEXITCODE
            $Fetched = @(Get-ChildItem -LiteralPath (Join-Path $Scratch 'fetch-sha') -Directory)[0].FullName
            $OnTbench = @(Get-ChildItem -LiteralPath (Join-Path $Scratch 'tbench-out') -Directory)[0].FullName
            & $Python -I -B (Join-Path $Here 'postrun\verify_on_tbench.py') --compare $Fetched $OnTbench
            $Same = $LASTEXITCODE
            Write-Output ('TBENCH_LOCAL bulk_archives=' + ($Bulk.Count / 2) + ' verify_exit=' + $Verify + ' fetch_exit=' + $Fetch + ' compare_exit=' + $Same)
            $global:LASTEXITCODE = [int](($Bulk.Count -eq 0) -or ($Verify -ne $Fetch) -or ($Fetch -ne $script:Results['fetch-local'].code) -or ($Same -ne 0))
        }
    }
    Run-Step 'cleanup' 'cleanup.log' {
        $Bad = 0
        # This run's processes only: descendants of this script, or orphans created since it started (other
        # harnesses' validations may run in parallel with their own synthetic_stage.py).
        $All = @(Get-CimInstance Win32_Process)
        $Alive = @{}; foreach ($P in $All) { $Alive[[int]$P.ProcessId] = $true }
        $Mine = @{}; $Mine[[int]$PID] = $true
        do {
            $Grew = $false
            foreach ($P in $All) {
                if (-not $Mine.ContainsKey([int]$P.ProcessId) -and $Mine.ContainsKey([int]$P.ParentProcessId)) { $Mine[[int]$P.ProcessId] = $true; $Grew = $true }
            }
        } while ($Grew)
        $Left = @($All | Where-Object { $_.CommandLine -match 'fake_server\.py|synthetic_stage\.py|mab_runner\.py' -and ($Mine.ContainsKey([int]$_.ProcessId) -or (-not $Alive.ContainsKey([int]$_.ParentProcessId) -and $_.CreationDate -ge $RunStart)) })
        if ($Left.Count) { $Bad++; Write-Output ('LEFTOVER processes: ' + (($Left | ForEach-Object { $_.ProcessId }) -join ',')) }
        $Caches = @(Get-ChildItem -LiteralPath $Here -Recurse -Force -Directory -Filter '__pycache__')
        if ($Caches.Count) { $Bad++; Write-Output ('__pycache__ present: ' + ($Caches.FullName -join ', ')) }
        foreach ($File in Get-ChildItem -LiteralPath $Here -Recurse -File | Where-Object { $_.Extension -in '.py', '.json', '.ps1', '.sh', '.in', '.md' -and $_.FullName -notmatch '\\validation\\' }) {
            if ([IO.File]::ReadAllText($File.FullName).Contains("`r")) { $Bad++; Write-Output ('CR line ending: ' + $File.FullName) }
        }
        $Synthetic = @(Get-ChildItem -LiteralPath $Temp -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'pmabs-*' -or $_.Name -like 'pmab-fake-*' -or $_.Name -like 'pmab-realev-*' })
        if ($Synthetic.Count) { $Bad++; Write-Output ('synthetic leftovers in TEMP: ' + ($Synthetic.Name -join ', ')) }
        $CAfter = @(Get-ChildItem -LiteralPath $CTemp -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'pmab*' -or $_.Name -like 'sealed-git*' } | ForEach-Object { $_.FullName })
        $New = @($CAfter | Where-Object { $CBefore -notcontains $_ })
        if ($New.Count) { $Bad++; Write-Output ('new entries on C: ' + ($New -join ', ')) }
        # The real MemAgent32 data never enters this directory: no context file name, no chunk text.
        & $Python -B privacy_check.py $Here 'D:\Remote-RA-temp\ext\mab-contexts\x\shared'
        if ($LASTEXITCODE -ne 0) { $Bad++ }
        Write-Output ('CLEANUP bad=' + $Bad)
        $global:LASTEXITCODE = [int]($Bad -ne 0)
    }
} finally { Pop-Location }
$Failed = @($Results.Keys | Where-Object { $Results[$_].code -ne 0 })
$Summary = [ordered]@{stamp = $Stamp; commit = $Pins.commit; payload_sha256 = $Second[0]; entry_script_sha256 = $Second[1]; steps = $Results; failed = $Failed}
[IO.File]::WriteAllText((Join-Path $Logs 'summary.json'), ($Summary | ConvertTo-Json -Depth 5))
if ($Failed.Count) { Write-Output ('LOCAL_VALIDATION_FAIL failed=' + ($Failed -join ',') + ' logs=' + $Logs); exit 1 }
Write-Output ('LOCAL_VALIDATION_PASS commit=' + $Pins.commit + ' logs=' + $Logs)
