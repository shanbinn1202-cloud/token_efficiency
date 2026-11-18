[CmdletBinding()]
param(
    # Both printed by the perf-mab entry point's submission line (stage, job_id).
    [string]$Stage = '',
    [string]$JobId = '',
    # sha256sum of <stage>/fetch/pmab-core.tar.xz printed by the read-only check block on DeltaAI.
    [string]$RemoteArchiveSha256 = '',
    # State and ExitCode columns of the job row printed by sacct in the same check block.
    [string]$SlurmState = '',
    [string]$SlurmExitCode = '',
    # Also fetch the bulk archives (runtime.jsonl, native receipts, trajectories): every fetch/*.tar.xz in one scp.
    [switch]$WithBulk,
    # Verify an archive that is already local instead of fetching it (no connection); its bulk archives next to it.
    [string]$LocalArchive = '',
    # Local synthetic stages only (fake servers): accept the archive's synthetic pins after re-validating them.
    [switch]$SyntheticPins,
    # A Python with torch and numpy: the offline selection replay of the observation_query servers (needs the bulk).
    [string]$ParityPython = '',
    [string]$Python = '',
    [string]$OutRoot = '',
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
# Nothing on C:: when TEMP points at C:, use D:\Remote-RA-temp\tmp\perf-mab (localtemp.py does the same).
if (Test-Path -LiteralPath 'D:\Remote-RA-temp') {
    foreach ($Name in 'TEMP', 'TMP', 'TMPDIR') {
        $Value = [Environment]::GetEnvironmentVariable($Name)
        if (-not $Value -or $Value -match '^[Cc]:' -or $Value.StartsWith('/')) {
            [void](New-Item -ItemType Directory -Force -Path 'D:\Remote-RA-temp\tmp\perf-mab')
            [Environment]::SetEnvironmentVariable($Name, 'D:\Remote-RA-temp\tmp\perf-mab')
        }
    }
}
if ($SelfTest -and -not $Stage) { $Stage = '/u/ycao11/pmab-' + ('0' * 32); $JobId = '1' }
if ($Stage -cnotmatch '^/u/ycao11/pmab-[0-9a-f]{32}$') { throw 'invalid perf-mab stage path (expected /u/ycao11/pmab-<32 hex>)' }
if ($JobId -cnotmatch '^[0-9]+$') { throw 'invalid Slurm job id' }
if ($RemoteArchiveSha256 -and $RemoteArchiveSha256 -cnotmatch '^[0-9a-fA-F]{64}$') { throw 'invalid remote SHA256 (paste the 64-hex value, not a placeholder)' }
if ($SlurmState -and $SlurmState -cnotmatch '^[A-Z_+]+$') { throw 'invalid Slurm state' }
if ($SlurmExitCode -and $SlurmExitCode -cnotmatch '^[0-9]+:[0-9]+$') { throw 'invalid Slurm exit code' }
if ([bool]$SlurmState -ne [bool]$SlurmExitCode) { throw 'give both -SlurmState and -SlurmExitCode, or neither' }
if ($SyntheticPins -and -not $LocalArchive) { throw '-SyntheticPins is for local synthetic archives only (-LocalArchive)' }

$Verifier = Join-Path $PSScriptRoot 'verify_evidence.py'
$Tables = Join-Path $PSScriptRoot 'mab_tables.py'
if (-not $OutRoot) { $OutRoot = Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))) 'verification' }
if (-not $Python) {
    $Python = if (Test-Path -LiteralPath 'D:\Anaconda\python.exe') { 'D:\Anaconda\python.exe' }
              else { (Get-Command python -CommandType Application | Select-Object -First 1).Source }
}
$Version = & $Python -I -B -c 'import sys;print(int(sys.version_info>=(3,12)))'
if ($LASTEXITCODE -ne 0 -or "$Version".Trim() -ne '1') { throw 'Python 3.12+ required for the local verifier (tarfile data filter)' }

if ($SelfTest) {
    $Tokens=$null; $Errors=$null
    [void][Management.Automation.Language.Parser]::ParseFile($PSCommandPath,[ref]$Tokens,[ref]$Errors)
    if ($Errors.Count) { throw 'PowerShell syntax invalid' }
    if (-not (Test-Path -LiteralPath $Verifier) -or -not (Test-Path -LiteralPath $Tables)) { throw 'verifier or table script missing' }
    if (-not $LocalArchive -and -not (Get-Command scp -CommandType Application -ErrorAction SilentlyContinue)) { throw 'scp not found' }
    Write-Output ('FETCH_SELFTEST_PASS connections=0 stage='+$Stage+' job='+$JobId+' python='+$Python+' out_root='+$OutRoot)
    return
}

$Out = Join-Path $OutRoot ('pmab-job'+$JobId+'-'+(Get-Date -Format 'yyyyMMddTHHmmss'))
if (Test-Path -LiteralPath $Out) { throw 'output directory already exists' }
[void](New-Item -ItemType Directory -Path $Out)
$Target = Join-Path $Out 'pmab-core.tar.xz'
$BulkDir = Join-Path $Out 'bulk-archives'
[void](New-Item -ItemType Directory -Path $BulkDir)

if ($LocalArchive) {
    Copy-Item -LiteralPath $LocalArchive -Destination $Target
    foreach ($Item in Get-ChildItem -LiteralPath (Split-Path -Parent $LocalArchive) -Filter 'pmab-bulk-*.tar.xz') { Copy-Item -LiteralPath $Item.FullName -Destination $BulkDir }
    Write-Output ('LOCAL_ARCHIVE_COPIED connections=0 source='+$LocalArchive)
} else {
    # scp into an ASCII temp path first (off C: and off OneDrive), then copy into the OneDrive tree and re-hash.
    $TempRoot = if (Test-Path -LiteralPath 'D:\Remote-RA-temp') { 'D:\Remote-RA-temp\scratch\perf-mab\fetch' } else { [IO.Path]::GetTempPath() }
    $Temp = Join-Path $TempRoot ('pmab-evidence-'+[Guid]::NewGuid().ToString('N'))
    [void](New-Item -ItemType Directory -Force -Path $Temp)
    try {
        $Remote = if ($WithBulk) { $Stage+'/fetch/pmab-*.tar.xz' } else { $Stage+'/fetch/pmab-core.tar.xz' }
        Write-Output ('READ-ONLY FETCH: one scp connection ('+$Remote+'); password/Duo prompts appear in this console. Safe to rerun.')
        & scp -o ConnectionAttempts=1 ('ycao11@gh-login03.delta.ncsa.illinois.edu:'+$Remote) $Temp
        if ($LASTEXITCODE -ne 0) { throw ('scp failed with exit '+$LASTEXITCODE+'; nothing on DeltaAI was changed; rerun is safe') }
        Copy-Item -LiteralPath (Join-Path $Temp 'pmab-core.tar.xz') -Destination $Target
        if ((Get-FileHash -LiteralPath (Join-Path $Temp 'pmab-core.tar.xz') -Algorithm SHA256).Hash -cne (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash) { throw 'local copy hash changed' }
        foreach ($Item in Get-ChildItem -LiteralPath $Temp -Filter 'pmab-bulk-*.tar.xz') { Copy-Item -LiteralPath $Item.FullName -Destination $BulkDir }
    } finally {
        Remove-Item -LiteralPath $Temp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
$Hash = (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Output ('LOCAL_ARCHIVE_SHA256='+$Hash)
if ($RemoteArchiveSha256 -and $Hash -cne $RemoteArchiveSha256.ToLowerInvariant()) { throw 'fetched archive does not match the remote sha256sum; keep both and report' }

$Check = Join-Path $Out 'check'
$Arguments = @('-B', $Verifier, '--archive', $Target, '--job-id', $JobId, '--stage', $Stage, '--out', $Check)
if ($RemoteArchiveSha256) { $Arguments += @('--remote-archive-sha256', $RemoteArchiveSha256) }
if ($SlurmState) { $Arguments += @('--slurm-state', $SlurmState, '--slurm-exit-code', $SlurmExitCode) }
if ($SyntheticPins) { $Arguments += @('--synthetic-pins') }
if (@(Get-ChildItem -LiteralPath $BulkDir -Filter 'pmab-bulk-*.tar.xz').Count) { $Arguments += @('--bulk-dir', $BulkDir) }
if ($ParityPython) { $Arguments += @('--parity-python', $ParityPython) }
& $Python @Arguments
$Code = $LASTEXITCODE
# The tables are a local derivation of the verified files; they never change the verdict or the exit code.
if (Test-Path -LiteralPath (Join-Path $Check 'local-verification.json')) {
    & $Python -B $Tables $Check
    if ($LASTEXITCODE -ne 0) { Write-Output ('MAB_TABLES_FAILED exit='+$LASTEXITCODE) }
    Write-Output ('MAB_TABLES_DIR='+(Join-Path $Check 'mab-tables'))
}
Write-Output ('EVIDENCE_DIR='+$Out)
exit $Code
