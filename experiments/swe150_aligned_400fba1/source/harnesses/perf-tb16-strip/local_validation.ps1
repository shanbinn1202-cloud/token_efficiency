# perf-tb16 local validation: no connection to DeltaAI or tbench, no job submission, nothing on C:.
# Two builds (determinism), the unit tests, the gates (every plan's memory / time / cost and the refusals), the release
# parity (the job's launch arguments through the sealed serve.py's own parser), the runtime's own chat-text evidence
# (runtime_evidence.py: the sealed commit's CPU end-to-end tests, every request through tb_checks' per-request checks,
# four tampered copies refused, the sampling probe on a stub vLLM), the real tokenizer's exact stream check
# (realtok_check.py: the runtime's ChatTextTranscript with the pinned tokenizer and template), the re-pin path in
# disposable copies (the pinned commit again: PINNED; integrate-1015 and an unaccepted module: PLACEHOLDER, a real
# submission refused there), the
# synthetic stages (e2e_tb16.py: the real job, tbench scripts with the fake Harbor and postrun on fakes, every scenario
# verified, joined, packaged, fetched), the eval pipeline on their outputs (eval_compat.py, --strict audit), the entry
# script's self-test and dry run of every plan plus refusals (never a real submission of a runnable plan: on the PINNED
# pin every accepted call is a -DryRun), the transport self-test, deploy_tbench.ps1 -SelfTest (manifest = pins),
# fetch_run.ps1 -SelfTest, and the clean-up checks. Logs: validation\<stamp>\.
# The Gemma plans (gfull, gobs, grfill, gstock, gdry) run through every step: the unit tests (incl. the Qwen plans'
# recorded digests: unchanged), the gates and their Gemma refusals, the Gemma launch arguments through the sealed
# serve.py's prepare_gemma4, the runtime's Gemma chat-text evidence, the real Gemma tokenizer, the re-pin rehearsal (a
# Qwen-only chat protocol: PLACEHOLDER), the Gemma synthetic stages, the eval pipeline on the Gemma set, SelfTest and
# DryRun of every plan; the Qwen plans' DryRun lines must equal the recorded ones (fixtures/qwen-dryrun-120bc3c.txt:
# stage, payload bytes and commit aside), and the 68d19b4 ones (fixtures/qwen-dryrun-68d19b4.txt) with the three new
# switches named: the re-pin of 2026-10-21 changed those plans' entry lines by the switches alone.
# The re-pin of 2026-10-21 (integrate-1019-tbgoal 120bc3c): the runtime evidence includes the goal_query chat-text
# servers and the optimized flags (test_integrate_flags, test_deferred_completion) with the goal checks and their
# tampered copies; the Terminus 2 template hash check (terminus2_check.py: the runtime's test_terminus2_goal with
# REMOTE_RA_HARBOR_TERMINUS2_DIR on the read-only copy of the agent host's harbor/agents/terminus_2, -Terminus2Dir);
# the synthetic stages goal (89 tasks) and goaldry (summaries); the eval pipeline on the Table 1 set (five plans).
# The re-pin of 2026-10-22 (integrate-1021 9ef0d6d): the runtime evidence adds the Gemma goal_query chat-text servers
# (runtime_evidence GEMMA_FLAG_GROUPS) with their tampered copies, the synthetic stages ggoaldry / gobsdry / grfilldry,
# the eval pipeline the Gemma dry set, the re-pin rehearsal the previous pin 120bc3c (PLACEHOLDER). The re-pin to
# fix-engine-stall-1022 53e3b7a (2026-10-22, job 3429670): the goal plans need the linear-time quoted-literal pass
# (unit tests, gates; the re-pin rehearsal's step 8: 9ef0d6d refused). The 262,144-token Qwen plans ctx25 / ctx24
# (2026-10-26; their own server settings): release parity (ctx24 through the sealed serve.py, ctx25 refused by its Qwen
# KV sizes), the synthetic stage ctx24, their DryRun lines (the settings named; ctx25 NOT_SUBMITTABLE blocked_plan,
# ctx24 none), every other Qwen plan's DryRun line as recorded. The same day the pin moved to main 4f5f4e7 (53e3b7a's
# runtime bytes; only .md files differ) with its release-check receipt, then to integrate-1026 cf2b28d (4f5f4e7 + the
# Qwen launcher's --kv-cache-gib 25 and two unused opt-ins): ctx25 runs (release parity as it is; no NOT_SUBMITTABLE;
# its synthetic stage), ctx24 the fallback.
# perf-tb16-strip (2026-10-30; a copy of perf-tb16, its own TEMP / scratch defaults): the pin thinking-strip 8ca8511 and
# the plan strip25 (--enable-thinking --thinking-history strip and the thinking-on stock control): the unit tests'
# ThinkingStrip / StripVsControl, release parity's test_the_strip_plan, the runtime's strip evidence through tb_checks'
# V1-V5 with ten tampered copies, realtok's strip streams (the stock render), the synthetic stage strip25, the strip25
# DryRun line with thinking=strip and its phrase refusals.
# The release smoke (2026-11-03; the pin integrate-1103 ea18256; plans smokeq / smokeqs / smokeg / smokegh): the unit
# tests' ReleaseSmoke (the plans, launch arguments, refusals, expectations, the two-engine log of job 3500565, the
# goal_forward frames, the coverage line), release parity's test_the_smoke_plans and the Gemma vLLM goal server, the
# runtime's strip + --goal-overlap chat-text evidence (runtime_evidence STRIP_GROUPS 'strip-ov-'), the synthetic stages
# smokeq / smokeqs / smokeg / smokegh, the eval pipeline's smokeset / gsmokeset tables, their DryRun lines (role, task
# set, async, coverage) and phrase refusals; the re-pin rehearsal's step 9 (8ca8511: the smoke plans blocked).
# The main ba4d1ba re-pin (2026-11-09; the smoke matrix's decisions D-A to D-I): plan smokeqt (smokeq + the history
# strip: unit tests Repin1109, gate refusals, release parity's test_the_thinking_smoke_plan, the synthetic stage
# smokeqt, its DryRun line and phrase refusals), the async-fix guard (D-G; the re-pin rehearsal's step 12: f4ccd4b),
# the client block of the Gemma 262,144-token plans (D-A: gctx25 NOT_SUBMITTABLE on ba4d1ba, CLIENT_BLOCKED below;
# the rehearsal's step 11: f5a5c36 lifts it), the Gemma async harness path (D-C: release parity's
# test_the_gemma_async_variant, the synthetic stage gasync).
# -Jobs (2026-11-10, default 6): runtime_evidence.py and e2e_tb16.py run their independent groups / scenarios in that
# many worker processes (--jobs; 1 = one after another, as before); each scenario's transcript is in
# <e2e work>\logs\<scenario>.log, each runner's in the runtime evidence's runner-*.log; the step logs keep their lines.
# Usage: powershell.exe -NoProfile -ExecutionPolicy Bypass -File '<perf-tb16>\local_validation.ps1'
#        [-Skip e2e,repin,evidence] [-Temp <dir on D:>] [-Scratch <dir on D:>] [-Jobs N]
param([string[]]$Skip = @(), [string]$Python = 'D:\Anaconda\python.exe', [int]$Jobs = 6,
      [string]$Terminus2Dir = 'D:\Remote-RA-temp\scratch\tb16-repin-1021\terminus2-agent-host',
      [string]$TokPython = 'D:\Remote-RA-temp\venvs\remote-ra-tok\Scripts\python.exe',
      [string]$Temp = 'D:\Remote-RA-temp\tmp\perf-tb16-strip', [string]$Scratch = 'D:\Remote-RA-temp\scratch\perf-tb16-strip')
$ErrorActionPreference = 'Stop'
foreach ($Name in 'TEMP', 'TMP', 'TMPDIR') { [Environment]::SetEnvironmentVariable($Name, $Temp) }
[Environment]::SetEnvironmentVariable('HF_HOME', (Join-Path $Temp 'hf'))  # Transformers' cache never on C:
[void](New-Item -ItemType Directory -Force -Path $Temp)
$Here = $PSScriptRoot
$Stamp = Get-Date -Format 'yyyyMMddTHHmmss'
$Logs = Join-Path $Here ('validation\' + $Stamp)
[void](New-Item -ItemType Directory -Force -Path $Logs)
[void](New-Item -ItemType Directory -Force -Path $Scratch)
$CTemp = Join-Path $env:LOCALAPPDATA 'Temp'
$CBefore = @(Get-ChildItem -LiteralPath $CTemp -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'tb16*' -or $_.Name -like 'sealed-git*' } | ForEach-Object { $_.FullName })
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
    Run-Step 'unit-tests' 'unit-tests.log' { & $Python -B -m unittest test_tb16 -v }
    Run-Step 'gate-checks' 'gate-checks.log' { & $Python -B gate_checks.py }
    Run-Step 'release-parity' 'release-parity.log' { & $Python -B -m unittest test_release_parity -v }
    if ($Skip -notcontains 'evidence') {
        Run-Step 'runtime-evidence' 'runtime-evidence.log' { & $Python -B runtime_evidence.py --python $Python --slow --jobs $Jobs --out (Join-Path $Scratch ('runtime-evidence-' + $Stamp)) }
        Run-Step 'real-tokenizer' 'real-tokenizer.log' { & $TokPython -B realtok_check.py --repo (Join-Path $Scratch ('runtime-evidence-' + $Stamp + '-repo\repo')) --work (Join-Path $Scratch ('realtok-' + $Stamp)) }
        Run-Step 'terminus2-goal' 'terminus2-goal.log' { & $Python -B terminus2_check.py --repo (Join-Path $Scratch ('runtime-evidence-' + $Stamp + '-repo\repo')) --harbor-dir $Terminus2Dir --out (Join-Path $Scratch ('terminus2-goal-' + $Stamp + '.json')) --python $Python }
    }
    if ($Skip -notcontains 'repin') {
        Run-Step 'repin' 'repin.log' { & $Python -B test_repin.py --work (Join-Path $Scratch ('repin-' + $Stamp)) }
    }
    $E2E = Join-Path $Scratch ('e2e-' + $Stamp)
    if ($Skip -notcontains 'e2e') {
        Run-Step 'synthetic-stages' 'synthetic-stages.log' { & $Python -B e2e_tb16.py --jobs $Jobs --work $E2E }
        Run-Step 'eval-pipeline' 'eval-pipeline.log' { & $Python -B eval_compat.py --e2e $E2E --out (Join-Path $Scratch ('eval-compat-' + $Stamp)) --python $Python }
    }
    $Pins = [IO.File]::ReadAllText((Join-Path $Here 'pins.json')) | ConvertFrom-Json
    $Entry = Join-Path $Here 'run_one_connection.ps1'
    Run-Step 'entry' 'entry.log' {
        $Bad = 0
        $QwenDry = @()
        foreach ($Plan in $Pins.config.plan_order) {
            $Phrase = $Pins.config.plans.$Plan.authorization
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry -Plan $Plan -SelfTest
            if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output ('UNEXPECTED self-test failure: ' + $Plan) }
            $Out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry -Plan $Plan -Authorize -Authorization $Phrase -DryRun
            $Out
            if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output ('UNEXPECTED dry-run failure: ' + $Plan) }
            if ($Pins.runtime_pin.status -cne 'PINNED' -and -not ($Out -match 'NOT_SUBMITTABLE runtime_pin=')) { $Bad++; Write-Output ('dry run without NOT_SUBMITTABLE on a placeholder pin: ' + $Plan) }
            # the light login gates: every DryRun names the pinned release-check receipt of the sealed commit
            if ($Pins.runtime_pin.status -ceq 'PINNED' -and -not ($Out -ccontains ('RELEASE_RECEIPT commit=' + $Pins.commit + ' status=GIT_ARCHIVE_TESTS_PASS tests=' + $Pins.release_receipt.tests_run + ' skipped=' + $Pins.release_receipt.skipped + ' result_line_sha256=' + $Pins.release_receipt.result_line_sha256 + ' log_sha256=' + $Pins.release_receipt.log_sha256 + ' login_test_runs=0'))) { $Bad++; Write-Output ('dry run without the pinned release-check receipt: ' + $Plan) }
            # a blocked plan (its sealed text, or ctx25: a KV size the sealed Qwen launcher lacks) is NOT_SUBMITTABLE
            $Own = $Pins.config.plans.$Plan.server
            $KvBlocked = $Own -and -not $Pins.config.plans.$Plan.model -and (@($Pins.runtime_features.qwen_kv_cache_gib) -notcontains [int]$Own.kv_cache_gib)
            # the main ba4d1ba re-pin (2026-11-09): a Gemma plan with its own context the sealed Terminal-Bench client does
            # not write (D-A, controller.client_block: gctx25 on ba4d1ba's 65536) and an async plan on a runtime without
            # the async fixes (D-G, async_fix_block) are NOT_SUBMITTABLE too
            $ClientBlocked = $Own -and [string]$Pins.config.plans.$Plan.model -ceq 'gemma4-31b-it' -and ([int]$Pins.runtime_features.tb_client.context.'gemma4-31b-it' -ne [int]$Own.max_model_len)
            $AsyncBlocked = ($Pins.config.plans.$Plan.async_scheduling -eq $true) -and -not $Pins.runtime_features.async_fix
            if ($ClientBlocked) { Write-Output ('CLIENT_BLOCKED plan=' + $Plan + ' (the sealed client writes the Gemma job at ' + $Pins.runtime_features.tb_client.context.'gemma4-31b-it' + ')') }
            if (($Pins.config.plans.$Plan.blocked -or $KvBlocked -or $ClientBlocked -or $AsyncBlocked) -and -not ($Out -match ('NOT_SUBMITTABLE blocked_plan=' + $Plan + ' '))) { $Bad++; Write-Output ('dry run of a blocked plan without NOT_SUBMITTABLE: ' + $Plan) }
            if ($Own) {
                # the plan's own settings on its line; a runnable one (ctx24) without any NOT_SUBMITTABLE on the PINNED pin
                $Want = ' kv_cache_gib=' + $Own.kv_cache_gib + ' max_model_len=' + $Own.max_model_len + ' dram_arena_gib=' + $Own.dram_arena_gib + ' max_input_tokens=' + $Own.max_model_len + ' '
                if (-not (@($Out | Where-Object { $_ -like 'DRY_RUN *' -and $_.Contains($Want) }).Count)) { $Bad++; Write-Output ('dry run of ' + $Plan + ' without its own settings: ' + $Want) }
                if ($Pins.runtime_pin.status -ceq 'PINNED' -and -not $KvBlocked -and -not $ClientBlocked -and -not $AsyncBlocked -and ($Out -match 'NOT_SUBMITTABLE')) { $Bad++; Write-Output ('a runnable plan with its own settings is NOT_SUBMITTABLE: ' + $Plan) }
            }
            # the release smoke (2026-11-03): a release-smoke plan names its role, task set, async switch and coverage
            if ([string]$Pins.config.plans.$Plan.role -ceq 'release-smoke') {
                $Async = 'off'
                if ($Pins.config.plans.$Plan.async_scheduling -eq $true) { $Async = 'on' }
                $Want = ' role=release-smoke task_set=smoke32 async_scheduling=' + $Async + ' coverage=evictions '
                if (-not (@($Out | Where-Object { $_ -like 'DRY_RUN *' -and $_.Contains($Want) }).Count)) { $Bad++; Write-Output ('dry run of ' + $Plan + ' without ' + $Want) }
                if (($Out -match 'NOT_SUBMITTABLE') -and -not $AsyncBlocked) { $Bad++; Write-Output ('a release smoke plan is NOT_SUBMITTABLE: ' + $Plan) }
            }
            # perf-tb16-strip: a thinking plan (strip25) names thinking=strip on its DryRun line
            if ($Pins.config.plans.$Plan.thinking -and -not (@($Out | Where-Object { $_ -like 'DRY_RUN *' -and $_.Contains(' thinking=strip') }).Count)) { $Bad++; Write-Output ('dry run of a thinking plan without thinking=strip: ' + $Plan) }
            # 2026-11-12 (config.tbench_verify): every DryRun line names the evidence push to tbench
            $Push = ' tbench_push=' + $Pins.config.tbench_verify.root + ':' + $Pins.config.tbench_verify.total_seconds + 's'
            if ($Pins.config.tbench_verify.push -and -not (@($Out | Where-Object { $_ -like 'DRY_RUN *' -and $_.Contains($Push) }).Count)) { $Bad++; Write-Output ('dry run without' + $Push + ': ' + $Plan) }
            $After = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry -Plan $Plan -AfterJob '1234567' -Authorize -Authorization $Phrase -DryRun
            $After
            if ($LASTEXITCODE -ne 0) { $Bad++; Write-Output ('UNEXPECTED dry-run failure with -AfterJob: ' + $Plan) }
            if (-not $Pins.config.plans.$Plan.model -and -not $Pins.config.plans.$Plan.server) { $QwenDry += @($Out) + @($After) | Where-Object { $_ -like 'DRY_RUN *' } }
        }
        # the Qwen plans' DryRun lines against the ones recorded at the re-pin of 2026-10-21 (stage, payload bytes,
        # commit aside), and the 68d19b4 ones with the three new switches named (the only change to those plans' lines)
        # (2026-11-12: the evidence push's field tbench_push=, new on every line, is dropped to compare)
        $Normal = @($QwenDry | ForEach-Object { ($_ -replace 'stage=\S+', 'stage=STAGE' -replace 'payload_bytes=[0-9]+', 'payload_bytes=N' -replace 'commit=[0-9a-f]+', 'commit=C' -replace ' tbench_push=\S+', '') })
        $Recorded = @([IO.File]::ReadAllLines((Join-Path $Here 'fixtures\qwen-dryrun-120bc3c.txt')))
        if (($Normal -join "`n") -ceq ($Recorded -join "`n")) { Write-Output ('QWEN_DRYRUN_IDENTICAL lines=' + $Normal.Count) } else { $Bad++; Write-Output ('QWEN_DRYRUN_DIFFERS now=' + $Normal.Count + ' recorded=' + $Recorded.Count); Compare-Object $Recorded $Normal | ForEach-Object { Write-Output ($_.SideIndicator + ' ' + $_.InputObject) } }
        $Old = @([IO.File]::ReadAllLines((Join-Path $Here 'fixtures\qwen-dryrun-68d19b4.txt')) | ForEach-Object { $_.Replace(' decode_fastpath2=off ', ' decode_fastpath2=on decode_fastpath3=on deferred_completion_io=on ') })
        $OldPlans = @($Normal | Where-Object { $_ -match ' plan=(full|obs|rfill|stock|dry|dryi|dryo) ' })
        if (($OldPlans -join "`n") -ceq ($Old -join "`n")) { Write-Output ('QWEN_DRYRUN_68D19B4_SWITCHES_ONLY lines=' + $OldPlans.Count) } else { $Bad++; Write-Output ('QWEN_DRYRUN_68D19B4_DIFFERS now=' + $OldPlans.Count + ' recorded=' + $Old.Count); Compare-Object $Old $OldPlans | ForEach-Object { Write-Output ($_.SideIndicator + ' ' + $_.InputObject) } }
        $Full = $Pins.config.plans.full.authorization
        $Refusals = @(
            @('-Plan', 'obs', '-Authorize', '-Authorization', $Full, '-DryRun'),
            @('-Plan', 'full', '-Authorize', '-Authorization', $Full.Replace('_BATCH_', '_INTERACTIVE_'), '-DryRun'),
            @('-Plan', 'full', '-Authorize', '-Authorization', $Full.Replace('_302MIN_', '_300MIN_'), '-DryRun'),
            @('-Plan', 'full', '-Authorize', '-Authorization', $Full.ToLowerInvariant(), '-DryRun'),
            @('-Plan', 'full', '-Authorization', $Full),
            @('-Plan', 'full', '-AfterJob', 'x7', '-Authorize', '-Authorization', $Full, '-DryRun'),
            @('-Plan', 'all', '-SelfTest'),
            @('-Plan', 'nope', '-SelfTest'),
            @('-Plan', 'full', '-RunLabel', 'x1', '-SelfTest'),
            @('-Plan', 'full', '-RunLabel', 'r100', '-SelfTest'),
            # goal_query (2026-10-21): the goal plan with another phrase or the obs plan's; the Gemma goal plans blocked
            @('-Plan', 'goal', '-Authorize', '-Authorization', 'TB16_GOAL_NOT_PLANNED', '-DryRun'),
            @('-Plan', 'goal', '-Authorize', '-Authorization', $Pins.config.plans.obs.authorization, '-DryRun'),
            @('-Plan', 'goaldryi', '-Authorize', '-Authorization', $Pins.config.plans.goaldry.authorization, '-DryRun'),
            # (the Gemma goal plans are blocked: their DryRun prints NOT_SUBMITTABLE blocked_plan, checked above; no real
            # submission is ever started on a PINNED pin, not even of a blocked plan)
            # the Gemma plans: another Gemma plan's phrase, the Qwen phrase of the same policy, the phrase without GEMMA4
            @('-Plan', 'gobs', '-Authorize', '-Authorization', $Pins.config.plans.gfull.authorization, '-DryRun'),
            @('-Plan', 'gfull', '-Authorize', '-Authorization', $Full, '-DryRun'),
            @('-Plan', 'gdry', '-Authorize', '-Authorization', $Pins.config.plans.gdry.authorization.Replace('_GEMMA4_', '_'), '-DryRun'),
            # the 262,144-token plans: each with the other's phrase
            @('-Plan', 'ctx24', '-Authorize', '-Authorization', $Pins.config.plans.ctx25.authorization, '-DryRun'),
            @('-Plan', 'ctx25', '-Authorize', '-Authorization', $Pins.config.plans.ctx24.authorization, '-DryRun'),
            # perf-tb16-strip: the strip plan with ctx25's phrase and ctx25 with the strip plan's
            @('-Plan', 'strip25', '-Authorize', '-Authorization', $Pins.config.plans.ctx25.authorization, '-DryRun'),
            @('-Plan', 'ctx25', '-Authorize', '-Authorization', $Pins.config.plans.strip25.authorization, '-DryRun'),
            # the release smoke (2026-11-03): each smoke plan with its sibling's phrase (the one plan switch of D1 / D2 is
            # the plan and its phrase), and a Qwen smoke plan with the Gemma one's
            @('-Plan', 'smokeq', '-Authorize', '-Authorization', $Pins.config.plans.smokeqs.authorization, '-DryRun'),
            @('-Plan', 'smokeqs', '-Authorize', '-Authorization', $Pins.config.plans.smokeq.authorization, '-DryRun'),
            @('-Plan', 'smokeg', '-Authorize', '-Authorization', $Pins.config.plans.smokegh.authorization, '-DryRun'),
            @('-Plan', 'smokegh', '-Authorize', '-Authorization', $Pins.config.plans.smokeg.authorization, '-DryRun'),
            @('-Plan', 'smokeq', '-Authorize', '-Authorization', $Pins.config.plans.smokeg.authorization, '-DryRun'),
            # the main ba4d1ba re-pin (2026-11-09): smokeqt (D-B) with smokeq's phrase and smokeq with smokeqt's
            @('-Plan', 'smokeqt', '-Authorize', '-Authorization', $Pins.config.plans.smokeq.authorization, '-DryRun'),
            @('-Plan', 'smokeq', '-Authorize', '-Authorization', $Pins.config.plans.smokeqt.authorization, '-DryRun'))
        # a real submission (no -DryRun) only where it is refused before any connection: the PLACEHOLDER pin
        if ($Pins.runtime_pin.status -cne 'PINNED') {
            foreach ($Plan in 'full', 'obs', 'rfill', 'stock', 'dry', 'gfull', 'gstock', 'gdry') { $Refusals += ,@('-Plan', $Plan, '-Authorize', '-Authorization', $Pins.config.plans.$Plan.authorization) }
        }
        foreach ($A in $Refusals) {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Entry @A
            if ($LASTEXITCODE -eq 0) { $Bad++; Write-Output ('UNEXPECTED acceptance: ' + ($A -join ' ')) } else { Write-Output ('REFUSED as expected: ' + ($A -join ' ')) }
        }
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
    Run-Step 'cleanup' 'cleanup.log' {
        $Bad = 0
        # this harness's fakes only (their command lines carry its absolute path): other harnesses' validations may run
        $E2EPosix = $E2E.Replace('\', '/')
        $Left = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and (($_.CommandLine.Contains($Here) -and $_.CommandLine -match 'fake_server|fake_tbench|e2e_tb16|relay\.py|run_harbor') -or $_.CommandLine.Contains($E2E) -or $_.CommandLine.Contains($E2EPosix)) })
        if ($Left.Count) { $Bad++; Write-Output ('LEFTOVER processes: ' + (($Left | ForEach-Object { [string]$_.ProcessId + ' ' + $_.CommandLine.Substring(0, [math]::Min(160, $_.CommandLine.Length)) }) -join ' | ')) }
        $Caches = @(Get-ChildItem -LiteralPath $Here -Recurse -Force -Directory -Filter '__pycache__')
        if ($Caches.Count) { $Bad++; Write-Output ('__pycache__ present: ' + ($Caches.FullName -join ', ')) }
        foreach ($File in Get-ChildItem -LiteralPath $Here -Recurse -File | Where-Object { $_.Extension -in '.py', '.json', '.ps1', '.sh', '.in', '.md', '.env', '.jinja' -and $_.FullName -notmatch '\\validation\\' -and $_.FullName -notmatch '\\dist\\' }) {
            if ([IO.File]::ReadAllText($File.FullName).Contains("`r")) { $Bad++; Write-Output ('CR line ending: ' + $File.FullName) }
        }
        $Synthetic = @(Get-ChildItem -LiteralPath $Temp -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'tb16-unit-*' -or $_.Name -like 'tb16-fake-*' })
        if ($Synthetic.Count) { $Bad++; Write-Output ('synthetic leftovers in TEMP: ' + ($Synthetic.Name -join ', ')) }
        $CAfter = @(Get-ChildItem -LiteralPath $CTemp -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'tb16*' -or $_.Name -like 'sealed-git*' } | ForEach-Object { $_.FullName })
        $New = @($CAfter | Where-Object { $CBefore -notcontains $_ })
        if ($New.Count) { $Bad++; Write-Output ('new entries on C: ' + ($New -join ', ')) }
        Write-Output ('CLEANUP bad=' + $Bad)
        $global:LASTEXITCODE = [int]($Bad -ne 0)
    }
} finally { Pop-Location }
$Failed = @($Results.Keys | Where-Object { $Results[$_].code -ne 0 })
$Summary = [ordered]@{stamp = $Stamp; commit = $Pins.commit; runtime_pin = $Pins.runtime_pin.status; payload_sha256 = $Second[0]; entry_script_sha256 = $Second[1]; pins_sha256 = $Second[2]; manifest_sha256 = $Pins.tbench_half.manifest_sha256; jobs = $Jobs; steps = $Results; failed = $Failed}
[IO.File]::WriteAllText((Join-Path $Logs 'summary.json'), ($Summary | ConvertTo-Json -Depth 5))
if ($Failed.Count) { Write-Output ('LOCAL_VALIDATION_FAIL failed=' + ($Failed -join ',') + ' commit=' + $Pins.commit + ' pin=' + $Pins.runtime_pin.status + ' logs=' + $Logs); exit 1 }
Write-Output ('LOCAL_VALIDATION_PASS commit=' + $Pins.commit + ' pin=' + $Pins.runtime_pin.status + ' payload=' + $Second[0] + ' logs=' + $Logs)
