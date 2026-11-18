[CmdletBinding()]
param(
    # Printed by the entry point's submission line (stage, job_id) and by the job's last JSON line.
    [string]$Stage = '',
    [string]$JobId = '',
    # sha256sum of fetch/tb16-core.tar.xz and fetch/tb16-bulk.tar.xz, printed by the README check block on DeltaAI.
    [string]$CoreSha256 = '',
    [string]$BulkSha256 = '',
    # State and ExitCode of the job row printed by sacct in the same check block.
    [string]$SlurmState = '',
    [string]$SlurmExitCode = '',
    # Fetch only the small core archive (receipts, summaries, measurements, samples, logs); the per-request checks and
    # the trial <-> server join need the bulk archive (fetch again without -CoreOnly).
    [switch]$CoreOnly,
    # Skip the tbench part.
    [switch]$SkipTbench,
    # Local files instead of connections (tests).
    [string]$LocalCore = '',
    [string]$LocalBulk = '',
    [string]$LocalPackage = '',
    # Verify again even if this job's archives were already verified (default: reuse that verification).
    [switch]$Reverify,
    # Verify again after a check of this harness was corrected (verify_evidence.py --corrected-checks): the verdict
    # follows the corrected checks; implies -Reverify and leaves the cache's verified.json as it is.
    [switch]$CorrectedChecks,
    [string]$Python = 'D:\Remote-RA-temp\venvs\remote-ra-tok\Scripts\python.exe',
    [string]$TbenchHost = 'tbench',
    [string]$TbenchRoot = '/home/ubuntu/tb16',
    [string]$OutRoot = '',
    [string]$CacheRoot = 'D:\Remote-RA-temp\scratch\perf-tb16-strip\fetch',
    # Local synthetic stages only (the CPU end-to-end run): the fake server's char tokenizer.
    [switch]$Synthetic,
    # With -Synthetic only: the synthetic pins the fake stage was built with (the real pins have no such stage).
    [string]$SyntheticPins = '',
    [switch]$SelfTest
)
# perf-tb16: the one local command after a job's END mail. It fetches both halves of the job (DeltaAI: the two evidence
# archives in <stage>/fetch over one scp connection, Duo in this console; tbench: the run package over the `tbench`
# alias, read-only: bin/package_run.sh packs a finished run once), checks every SHA-256, verifies the DeltaAI evidence
# (postrun\verify_evidence.py), joins Harbor's trials (results, ATIF trajectories, verifier rewards) with it by
# key_sha256 (postrun\join_tb16.py) and writes the tables (postrun\tb16_tables.py). Safe to re-run: the archives are
# cached by SHA-256 under -CacheRoot (never on C:, never in OneDrive), a verified job is not verified again, and while
# the Harbor run is still going on tbench it says so (exit 3).
$ErrorActionPreference = 'Stop'
foreach ($Name in 'TEMP', 'TMP', 'TMPDIR') {
    $Value = [Environment]::GetEnvironmentVariable($Name)
    if (-not $Value -or $Value -match '^[Cc]:' -or $Value.StartsWith('/')) {
        [void](New-Item -ItemType Directory -Force -Path 'D:\Remote-RA-temp\tmp\perf-tb16-strip')
        [Environment]::SetEnvironmentVariable($Name, 'D:\Remote-RA-temp\tmp\perf-tb16-strip')
    }
}
if ($SelfTest -and -not $Stage) { $Stage = '/u/ycao11/tb16-' + ('0' * 32); $JobId = '1' }
if ($Stage -cnotmatch '^/u/ycao11/tb16-[0-9a-f]{32}$') { throw 'invalid tb16 stage path (expected /u/ycao11/tb16-<32 hex>)' }
if ($JobId -cnotmatch '^[0-9]+$') { throw 'invalid Slurm job id' }
foreach ($Value in $CoreSha256, $BulkSha256) { if ($Value -and $Value -cnotmatch '^[0-9a-fA-F]{64}$') { throw 'invalid SHA256 (paste the 64-hex value)' } }
if ($SlurmState -and $SlurmState -cnotmatch '^[A-Z_+]+$') { throw 'invalid Slurm state' }
if ($SlurmExitCode -and $SlurmExitCode -cnotmatch '^[0-9]+:[0-9]+$') { throw 'invalid Slurm exit code' }
if ([bool]$SlurmState -ne [bool]$SlurmExitCode) { throw 'give both -SlurmState and -SlurmExitCode, or neither' }
if ($CacheRoot -match '^[Cc]:') { throw 'CacheRoot must not be on C:' }
$Here = $PSScriptRoot
$Verifier = Join-Path $Here 'verify_evidence.py'; $Joiner = Join-Path $Here 'join_tb16.py'
$Tables = Join-Path $Here 'tb16_tables.py'; $Unpack = Join-Path $Here 'unpack.py'
if (-not $OutRoot) { $OutRoot = Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $Here)))) 'verification' }
& $Python -I -B -c 'import sys;print(int(sys.version_info>=(3,11)))' | Out-Null
if ($LASTEXITCODE -ne 0) { throw ('Python 3.11+ required: '+$Python) }
if ($SyntheticPins -and -not $Synthetic) { throw '-SyntheticPins needs -Synthetic (local tests only)' }
$Cache = Join-Path $CacheRoot ('job'+$JobId)
if ($SelfTest) {
    $Tokens=$null; $Errors=$null
    [void][Management.Automation.Language.Parser]::ParseFile($PSCommandPath,[ref]$Tokens,[ref]$Errors)
    if ($Errors.Count) { throw 'PowerShell syntax invalid' }
    foreach ($Script in $Verifier, $Joiner, $Tables, $Unpack) { if (-not (Test-Path -LiteralPath $Script)) { throw ('missing '+$Script) } }
    foreach ($Tool in 'scp', 'ssh') { if (-not (Get-Command $Tool -CommandType Application -ErrorAction SilentlyContinue)) { throw ($Tool+' not found') } }
    Write-Output ('FETCH_RUN_SELFTEST_PASS connections=0 stage='+$Stage+' job='+$JobId+' cache='+$Cache+' out_root='+$OutRoot+' python='+$Python)
    return
}
[void](New-Item -ItemType Directory -Force -Path $Cache)
function Get-Sha([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
$CorePath = Join-Path $Cache 'tb16-core.tar.xz'; $BulkPath = Join-Path $Cache 'tb16-bulk.tar.xz'
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
            & scp -o ConnectionAttempts=1 ('ycao11@gh-login03.delta.ncsa.illinois.edu:'+$Stage+'/fetch/tb16-core.tar.xz') (Join-Path $Temp 'tb16-core.tar.xz')
        } else {
            & scp -r -o ConnectionAttempts=1 ('ycao11@gh-login03.delta.ncsa.illinois.edu:'+$Stage+'/fetch') $Temp
        }
        if ($LASTEXITCODE -ne 0) { throw ('scp failed with exit '+$LASTEXITCODE+'; nothing on DeltaAI changed; rerun is safe') }
        foreach ($Name in 'tb16-core.tar.xz', 'tb16-bulk.tar.xz', 'archives.json') {
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
    $Out = Join-Path $OutRoot ('tb16-job'+$JobId+'-'+$Stamp)
    $Check = Join-Path $Out 'check'
    $Evidence = Join-Path $Cache ('evidence-'+$Stamp)
    $Arguments = @('-B', $Verifier, '--core', $CorePath, '--job-id', $JobId, '--stage', $Stage, '--out', $Check, '--evidence', $Evidence)
    if ($HaveBulk) { $Arguments += @('--bulk', $BulkPath) }
    if ($CoreSha256) { $Arguments += @('--core-sha256', $CoreSha256) }
    if ($BulkSha256 -and $HaveBulk) { $Arguments += @('--bulk-sha256', $BulkSha256) }
    if ($SlurmState) { $Arguments += @('--slurm-state', $SlurmState, '--slurm-exit-code', $SlurmExitCode) }
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
if ($SkipTbench) { exit $VerifyCode }
# ---- tbench: the run package (Harbor's job directories, the specs and job configs, the host samples)
$Package = $null; $State = 'done'
if ($LocalPackage) {
    $Package = $LocalPackage; $State = 'local'
} else {
    if ($RunId -cnotmatch '^tb16-[0-9]+-r[0-9]{1,2}$') { throw ('unexpected run id '+$RunId) }
    $Lines = & ssh -o BatchMode=yes -o ConnectionAttempts=1 $TbenchHost ($TbenchRoot+'/bin/package_run.sh '+$RunId)
    $Code = $LASTEXITCODE
    $Last = @($Lines | Where-Object { $_ -like '{*' }) | Select-Object -Last 1
    if (-not $Last) { throw ('package_run.sh printed no JSON line (exit '+$Code+'): '+($Lines -join ' ')) }
    $Info = $Last | ConvertFrom-Json
    if ($Info.status -ceq 'HARBOR_RUNNING') {
        Write-Output ('TBENCH_HARBOR_RUNNING run_id='+$RunId+': the DeltaAI half is verified and cached; run this same command again later.')
        exit 3
    }
    if ($Info.status -cne 'PACKAGED') { throw ('tbench package failed: '+$Last) }
    $Package = Join-Path $Cache ('tbench-'+$RunId+'.tar.xz')
    if (-not ((Test-Path -LiteralPath $Package) -and (Get-Sha $Package) -ceq $Info.sha256)) {
        Write-Output ('READ-ONLY FETCH (tbench): '+$Info.path+' ('+$Info.bytes+' bytes)')
        & scp -o BatchMode=yes -o ConnectionAttempts=1 ($TbenchHost+':'+$Info.path) $Package
        if ($LASTEXITCODE -ne 0) { throw ('tbench scp failed with exit '+$LASTEXITCODE+'; rerun is safe') }
    }
    if ((Get-Sha $Package) -cne $Info.sha256) { throw 'tbench package SHA-256 differs from the remote one' }
}
$Unpacked = Join-Path $Cache ('tbench-'+$State+'-'+$Stamp)
& $Python -I -B $Unpack $Package $Unpacked
if ($LASTEXITCODE -ne 0) { throw 'tbench package could not be unpacked safely' }
$RunsDir = Join-Path $Unpacked $RunId
if (-not (Test-Path -LiteralPath $RunsDir)) { throw ('the package holds no '+$RunId) }
$JoinOut = Join-Path (Split-Path -Parent $Check) ('join-'+$State+'-'+$Stamp)
if ($SyntheticPins) { & $Python -B $Joiner $Check $RunsDir --out $JoinOut --synthetic-pins $SyntheticPins } else { & $Python -B $Joiner $Check $RunsDir --out $JoinOut }
$JoinCode = $LASTEXITCODE
$TablesOut = Join-Path (Split-Path -Parent $Check) ('tables-'+$State+'-'+$Stamp)
& $Python -B $Tables --out $TablesOut $JoinOut
Write-Output ('JOIN_DIR='+$JoinOut+' TABLES_DIR='+$TablesOut)
$Fetched = [ordered]@{job_id=$JobId; run_id=$RunId; stage=$Stage; check=$Check; join=$JoinOut; tables=$TablesOut; runs_dir=$RunsDir; verdict=$Verdict; stamp=$Stamp}
if ($CorrectedChecks) {
    Write-Output ('CORRECTED_CHECKS_FETCH='+($Fetched | ConvertTo-Json -Compress))
} else {
    [IO.File]::WriteAllText((Join-Path $Cache 'fetched.json'), ($Fetched | ConvertTo-Json -Compress))
    Write-Output ('FETCHED='+(Join-Path $Cache 'fetched.json'))
}
if ($VerifyCode -ne 0 -and $VerifyCode -ne 2 -and $VerifyCode -ne 3 -and $VerifyCode -ne 4) { exit $VerifyCode }
if ($JoinCode -ne 0) { exit 1 }
exit $VerifyCode
