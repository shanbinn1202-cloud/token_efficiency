[CmdletBinding()]
param(
    # Printed by the entry point's submission line (stage, job_id) and by the job's last JSON line.
    [string]$Stage = '',
    [string]$JobId = '',
    # sha256sum of fetch/gswe32-core.tar.xz and fetch/gswe32-bulk.tar.xz, printed by the README check block on DeltaAI.
    [string]$CoreSha256 = '',
    [string]$BulkSha256 = '',
    # State and ExitCode of the job row printed by sacct in the same check block.
    [string]$SlurmState = '',
    [string]$SlurmExitCode = '',
    # Fetch only the small core archive (receipts, summaries, measurements, samples, logs); the per-request checks,
    # the selection replay and the exact-token join need the bulk archive (fetch again without -CoreOnly).
    [switch]$CoreOnly,
    # Skip the DeltaAI download (the archives are already in the cache) or the tbench part.
    [switch]$SkipTbench,
    # Local files instead of connections (tests).
    [string]$LocalCore = '',
    [string]$LocalBulk = '',
    [string]$LocalPackage = '',
    # Verify again even if this job's archives were already verified (default: reuse that verification).
    [switch]$Reverify,
    # Verify again after a check of this harness was corrected (verify_evidence.py --corrected-checks): the verdict
    # follows the corrected checks; implies -Reverify and leaves the cache's verified.json / fetched.json as they are.
    [switch]$CorrectedChecks,
    [string]$Python = 'D:\Remote-RA-temp\venvs\remote-ra-tok\Scripts\python.exe',
    [string]$ParityPython = 'D:\Anaconda\python.exe',
    [switch]$NoParity,
    [string]$TbenchHost = 'tbench',
    [string]$TbenchRoot = '/home/ubuntu/gswe32',
    [string]$OutRoot = '',
    [string]$CacheRoot = 'D:\Remote-RA-temp\scratch\perf-gemma-swe32\fetch',
    # Local synthetic stages only (the CPU end-to-end run): the fake server's char tokenizer, no exact re-render, no
    # selection replay (the fake's selections are a stand-in).
    [switch]$Synthetic,
    # With -Synthetic only: the synthetic pins the fake stage was built with (the real pins have no such stage).
    [string]$SyntheticPins = '',
    [switch]$SelfTest
)
# perf-gemma-swe32: the one local command after a job's END mail. It fetches both halves of the job (DeltaAI: the two evidence
# archives in <stage>/fetch over one scp connection, Duo in this console; tbench: the run package over the `tbench`
# alias), checks every SHA-256, verifies the DeltaAI evidence (postrun\verify_evidence.py: token-exact checks with the
# tokenizer venv, the selection replay with Anaconda), joins the tbench trajectories and evaluation with it
# (postrun\join_gswe32.py, exact-token re-render) and writes the tables (postrun\gswe32_tables.py). Safe to re-run: the
# archives are cached by SHA-256 under -CacheRoot (never on C:, never in OneDrive), a verified job is not verified again,
# and while tbench is still evaluating it says so (exit 3) and fetches the evaluation on the next run.
$ErrorActionPreference = 'Stop'
foreach ($Name in 'TEMP', 'TMP', 'TMPDIR') {
    $Value = [Environment]::GetEnvironmentVariable($Name)
    if (-not $Value -or $Value -match '^[Cc]:' -or $Value.StartsWith('/')) {
        [void](New-Item -ItemType Directory -Force -Path 'D:\Remote-RA-temp\tmp\gswe32')
        [Environment]::SetEnvironmentVariable($Name, 'D:\Remote-RA-temp\tmp\gswe32')
    }
}
if ($SelfTest -and -not $Stage) { $Stage = '/u/ycao11/gswe32-' + ('0' * 32); $JobId = '1' }
if ($Stage -cnotmatch '^/u/ycao11/gswe32-[0-9a-f]{32}$') { throw 'invalid gswe32 stage path (expected /u/ycao11/gswe32-<32 hex>)' }
if ($JobId -cnotmatch '^[0-9]+$') { throw 'invalid Slurm job id' }
foreach ($Value in $CoreSha256, $BulkSha256) { if ($Value -and $Value -cnotmatch '^[0-9a-fA-F]{64}$') { throw 'invalid SHA256 (paste the 64-hex value)' } }
if ($SlurmState -and $SlurmState -cnotmatch '^[A-Z_+]+$') { throw 'invalid Slurm state' }
if ($SlurmExitCode -and $SlurmExitCode -cnotmatch '^[0-9]+:[0-9]+$') { throw 'invalid Slurm exit code' }
if ([bool]$SlurmState -ne [bool]$SlurmExitCode) { throw 'give both -SlurmState and -SlurmExitCode, or neither' }
if ($CacheRoot -match '^[Cc]:') { throw 'CacheRoot must not be on C:' }
$Here = $PSScriptRoot
$Verifier = Join-Path $Here 'verify_evidence.py'; $Joiner = Join-Path $Here 'join_gswe32.py'
$Tables = Join-Path $Here 'gswe32_tables.py'; $Unpack = Join-Path $Here 'unpack.py'; $Packer = Join-Path $Here 'pack_gswe32.py'
if (-not $OutRoot) { $OutRoot = Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $Here)))) 'verification' }
& $Python -I -B -c 'import sys;print(int(sys.version_info>=(3,11)))' | Out-Null
if ($LASTEXITCODE -ne 0) { throw ('Python 3.11+ required: '+$Python) }
if ($Synthetic) { $NoParity = $true }
if ($SyntheticPins -and -not $Synthetic) { throw '-SyntheticPins needs -Synthetic (local tests only)' }
if (-not $NoParity) { & $ParityPython -I -B -c 'import torch, numpy' 2>$null; if ($LASTEXITCODE -ne 0) { throw ('the parity Python lacks torch or numpy: '+$ParityPython) } }
$Cache = Join-Path $CacheRoot ('job'+$JobId)
if ($SelfTest) {
    $Tokens=$null; $Errors=$null
    [void][Management.Automation.Language.Parser]::ParseFile($PSCommandPath,[ref]$Tokens,[ref]$Errors)
    if ($Errors.Count) { throw 'PowerShell syntax invalid' }
    foreach ($Script in $Verifier, $Joiner, $Tables, $Unpack, $Packer) { if (-not (Test-Path -LiteralPath $Script)) { throw ('missing '+$Script) } }
    foreach ($Tool in 'scp', 'ssh') { if (-not (Get-Command $Tool -CommandType Application -ErrorAction SilentlyContinue)) { throw ($Tool+' not found') } }
    Write-Output ('FETCH_RUN_SELFTEST_PASS connections=0 stage='+$Stage+' job='+$JobId+' cache='+$Cache+' out_root='+$OutRoot+' python='+$Python+' parity_python='+$ParityPython)
    return
}
[void](New-Item -ItemType Directory -Force -Path $Cache)
function Get-Sha([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
$CorePath = Join-Path $Cache 'gswe32-core.tar.xz'; $BulkPath = Join-Path $Cache 'gswe32-bulk.tar.xz'
# ---- DeltaAI: the two archives (cached by SHA-256)
$NeedCore = -not ((Test-Path -LiteralPath $CorePath) -and (-not $CoreSha256 -or (Get-Sha $CorePath) -ceq $CoreSha256.ToLowerInvariant()))
$NeedBulk = (-not $CoreOnly) -and -not ((Test-Path -LiteralPath $BulkPath) -and (-not $BulkSha256 -or (Get-Sha $BulkPath) -ceq $BulkSha256.ToLowerInvariant()))
if ($LocalCore) { Copy-Item -LiteralPath $LocalCore -Destination $CorePath -Force; $NeedCore = $false }
if ($LocalBulk) { Copy-Item -LiteralPath $LocalBulk -Destination $BulkPath -Force; $NeedBulk = $false }
if ($NeedCore -or $NeedBulk) {
    $Temp = Join-Path $Cache ('dl-'+[Guid]::NewGuid().ToString('N'))
    [void](New-Item -ItemType Directory -Path $Temp)
    try {
        Write-Output 'READ-ONLY FETCH (DeltaAI): one scp connection; password/Duo prompts appear in this console. Safe to rerun.'
        if ($CoreOnly -or -not $NeedBulk) {
            & scp -o ConnectionAttempts=1 ('ycao11@gh-login03.delta.ncsa.illinois.edu:'+$Stage+'/fetch/gswe32-core.tar.xz') (Join-Path $Temp 'gswe32-core.tar.xz')
        } else {
            & scp -r -o ConnectionAttempts=1 ('ycao11@gh-login03.delta.ncsa.illinois.edu:'+$Stage+'/fetch') $Temp
        }
        if ($LASTEXITCODE -ne 0) { throw ('scp failed with exit '+$LASTEXITCODE+'; nothing on DeltaAI changed; rerun is safe') }
        foreach ($Name in 'gswe32-core.tar.xz', 'gswe32-bulk.tar.xz', 'archives.json', 'eval-launch.json') {
            $Found = @(Get-ChildItem -LiteralPath $Temp -Recurse -File -Filter $Name)
            if ($Found.Count -eq 1) { Move-Item -LiteralPath $Found[0].FullName -Destination (Join-Path $Cache $Name) -Force }
        }
    } finally { Remove-Item -LiteralPath $Temp -Recurse -Force -ErrorAction SilentlyContinue }
}
if (-not (Test-Path -LiteralPath $CorePath)) { throw 'no core archive in the cache' }
$CoreHash = Get-Sha $CorePath
Write-Output ('CORE_SHA256='+$CoreHash+' BYTES='+(Get-Item -LiteralPath $CorePath).Length)
if ($CoreSha256 -and $CoreHash -cne $CoreSha256.ToLowerInvariant()) { throw 'core archive does not match the remote sha256sum; keep both and report' }
$HaveBulk = Test-Path -LiteralPath $BulkPath
if ($HaveBulk) {
    $BulkHash = Get-Sha $BulkPath
    Write-Output ('BULK_SHA256='+$BulkHash+' BYTES='+(Get-Item -LiteralPath $BulkPath).Length)
    if ($BulkSha256 -and $BulkHash -cne $BulkSha256.ToLowerInvariant()) { throw 'bulk archive does not match the remote sha256sum; keep both and report' }
} else { $BulkHash = '' }
# ---- verification (reused when this job's archives were already verified)
$Marker = Join-Path $Cache 'verified.json'
$Check = $null
if ((Test-Path -LiteralPath $Marker) -and -not $Reverify -and -not $CorrectedChecks) {
    $Previous = [IO.File]::ReadAllText($Marker) | ConvertFrom-Json
    if ($Previous.core -ceq $CoreHash -and $Previous.bulk -ceq $BulkHash -and (Test-Path -LiteralPath (Join-Path $Previous.check 'local-verification.json'))) {
        $Check = $Previous.check; $Verdict = $Previous.verdict; $VerifyCode = [int]$Previous.code
        Write-Output ('VERIFICATION_REUSED check='+$Check+' verdict='+$Verdict)
    }
}
$Stamp = Get-Date -Format 'yyyyMMddTHHmmss'
if (-not $Check) {
    $Out = Join-Path $OutRoot ('gswe32-job'+$JobId+'-'+$Stamp)
    $Check = Join-Path $Out 'check'
    $Evidence = Join-Path $Cache ('evidence-'+$Stamp)
    $Arguments = @('-B', $Verifier, '--core', $CorePath, '--job-id', $JobId, '--stage', $Stage, '--out', $Check, '--evidence', $Evidence)
    if ($HaveBulk) { $Arguments += @('--bulk', $BulkPath) }
    if ($CoreSha256) { $Arguments += @('--core-sha256', $CoreSha256) }
    if ($BulkSha256 -and $HaveBulk) { $Arguments += @('--bulk-sha256', $BulkSha256) }
    if ($SlurmState) { $Arguments += @('--slurm-state', $SlurmState, '--slurm-exit-code', $SlurmExitCode) }
    if (-not $NoParity) { $Arguments += @('--parity-python', $ParityPython) }
    if ($Synthetic) { $Arguments += @('--fake-tokens') }
    if ($SyntheticPins) { $Arguments += @('--synthetic-pins', $SyntheticPins) }
    if ($CorrectedChecks) { $Arguments += @('--corrected-checks') }
    & $Python @Arguments
    $VerifyCode = $LASTEXITCODE
    $Verdict = ([IO.File]::ReadAllText((Join-Path $Check 'local-verification.json')) | ConvertFrom-Json).verdict
    if (-not $CorrectedChecks) {
        [IO.File]::WriteAllText($Marker, (@{core=$CoreHash; bulk=$BulkHash; check=$Check; verdict=$Verdict; code=$VerifyCode} | ConvertTo-Json -Compress))
    }
}
$Report = [IO.File]::ReadAllText((Join-Path $Check 'local-verification.json')) | ConvertFrom-Json
$RunId = $Report.run_id
Write-Output ('VERDICT='+$Verdict+' RUN_ID='+$RunId+' CHECK_DIR='+$Check)
Write-Output ('GOAL_TEXT='+$Report.goal_text_status)
if ($SkipTbench) { exit $VerifyCode }
# ---- tbench: the run package (trajectories, timelines, closes, host samples, evaluation)
$Package = $null; $EvalState = 'absent'
if ($LocalPackage) {
    $Package = $LocalPackage; $EvalState = 'local'
} else {
    if ($RunId -cnotmatch '^gswe32-[0-9]+-r[0-9]{1,2}$') { throw ('unexpected run id '+$RunId) }
    $Lines = & ssh -o BatchMode=yes -o ConnectionAttempts=1 $TbenchHost ($TbenchRoot+'/bin/package_run.sh '+$RunId)
    $Code = $LASTEXITCODE
    $Last = @($Lines | Where-Object { $_ -like '{*' }) | Select-Object -Last 1
    if (-not $Last) { throw ('package_run.sh printed no JSON line (exit '+$Code+'): '+($Lines -join ' ')) }
    $Info = $Last | ConvertFrom-Json
    if ($Info.status -ceq 'EVAL_RUNNING') {
        Write-Output ('TBENCH_EVALUATION_RUNNING run_id='+$RunId+' state='+$Info.eval_state+': the DeltaAI half is verified and cached; run this same command again later to fetch the evaluation.')
        exit 3
    }
    if ($Info.status -cne 'PACKAGED') { throw ('tbench package failed: '+$Last) }
    $EvalState = $Info.eval_state
    $Package = Join-Path $Cache ('tbench-'+$RunId+'.'+$EvalState+'.tar.xz')
    if (-not ((Test-Path -LiteralPath $Package) -and (Get-Sha $Package) -ceq $Info.sha256)) {
        Write-Output ('READ-ONLY FETCH (tbench): '+$Info.path+' ('+$Info.bytes+' bytes)')
        & scp -o BatchMode=yes -o ConnectionAttempts=1 ($TbenchHost+':'+$Info.path) $Package
        if ($LASTEXITCODE -ne 0) { throw ('tbench scp failed with exit '+$LASTEXITCODE+'; rerun is safe') }
    }
    if ((Get-Sha $Package) -cne $Info.sha256) { throw 'tbench package SHA-256 differs from the remote one' }
}
$Unpacked = Join-Path $Cache ('tbench-'+$EvalState+'-'+$Stamp)
& $Python -I -B $Unpack $Package $Unpacked
if ($LASTEXITCODE -ne 0) { throw 'tbench package could not be unpacked safely' }
$RunsDir = Join-Path $Unpacked $RunId
if (-not (Test-Path -LiteralPath $RunsDir)) { throw ('the package holds no '+$RunId) }
$JoinOut = Join-Path (Split-Path -Parent $Check) ('join-'+$EvalState+'-'+$Stamp)
if ($Synthetic) { & $Python -B $Joiner $Check $RunsDir --out $JoinOut } else { & $Python -B $Joiner $Check $RunsDir --tokenizer-python $Python --out $JoinOut }
$JoinCode = $LASTEXITCODE
$TablesOut = Join-Path (Split-Path -Parent $Check) ('tables-'+$EvalState+'-'+$Stamp)
& $Python -B $Tables --out $TablesOut $JoinOut
Write-Output ('JOIN_DIR='+$JoinOut+' TABLES_DIR='+$TablesOut+' EVALUATION='+$EvalState)
# gswe32: the fetch record postrun\pack_gswe32.py reads (the package for Xuanxue, after every job of the rerun)
$Fetched = [ordered]@{job_id=$JobId; run_id=$RunId; stage=$Stage; check=$Check; join=$JoinOut; tables=$TablesOut; runs_dir=$RunsDir; eval_state=$EvalState; verdict=$Verdict; goal_text_status=$Report.goal_text_status; stamp=$Stamp}
if ($CorrectedChecks) {
    Write-Output ('CORRECTED_CHECKS_FETCH='+($Fetched | ConvertTo-Json -Compress))
} else {
    [IO.File]::WriteAllText((Join-Path $Cache 'fetched.json'), ($Fetched | ConvertTo-Json -Compress))
    Write-Output ('FETCHED='+(Join-Path $Cache 'fetched.json')+' (after the last job: python -B postrun\pack_gswe32.py <job id> <job id> <job id>)')
}
if ($EvalState -ceq 'absent') { Write-Output 'TBENCH_EVALUATION_ABSENT: no evaluation of this run on tbench (eval-launch.json says why); the tables show accuracy as not evaluated.' }
if ($VerifyCode -ne 0 -and $VerifyCode -ne 2 -and $VerifyCode -ne 3 -and $VerifyCode -ne 4) { exit $VerifyCode }
if ($JoinCode -ne 0) { exit 1 }
if ($EvalState -ceq 'absent') { exit 3 }
exit $VerifyCode
