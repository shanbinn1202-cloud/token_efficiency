# Deploy the gswe32 tbench half to tbench:/home/ubuntu/gswe32 (perf-gemma-swe32; a new directory next to the v12 pilot's
# /home/ubuntu/selective-prefix-pilot-v12, which is never touched; the v11 agent env and the cached evaluator env are
# only read). Creates bin/, upstream/, tasks/, runs/, sock/ (0700), lock/, packages/ and an empty mswea-config/, uploads
# the pinned files through a temporary directory and installs them only if every SHA256 matches this checkout; writes
# bin/MANIFEST.sha256 (the login-node preflight compares it with pins.json tbench_half.manifest_sha256). bin/ carries
# lockset.py and byte-identical copies of the other halves' lock modules (tb16, swe32, swe16, swe10: the shared tbench
# locks; their halves are never written here). Refuses while this half's own tbench lock is held by a live run (never
# redeploy under a running pool or evaluation: a gswe32 run always holds its own lock, the first of the set). Uses the
# `tbench` SSH alias of this machine. Run from a Windows PowerShell:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File '<perf-gemma-swe32>\tbench\deploy_tbench.ps1'
# -SelfTest: print the manifest only, no connection.
param([switch]$SelfTest, [string]$HostAlias = 'tbench')
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = '/home/ubuntu/gswe32'
$files = [ordered]@{}
foreach ($f in (Get-ChildItem -LiteralPath (Join-Path $here 'bin') -File | Sort-Object Name)) { $files['bin/' + $f.Name] = $f.FullName }
foreach ($f in (Get-ChildItem -LiteralPath (Join-Path $here 'upstream') -File | Sort-Object Name)) { $files['upstream/' + $f.Name] = $f.FullName }
$files['tasks/gswe32-tasks.json'] = Join-Path (Split-Path -Parent $here) 'tasks\gswe32-tasks.json'
$files['gswe32.env'] = Join-Path $here 'gswe32.env'
[string[]]$names = @($files.Keys)
[Array]::Sort($names, [StringComparer]::Ordinal)
$manifest = @()
foreach ($name in $names) {
    $path = $files[$name]
    $bytes = [IO.File]::ReadAllBytes($path)
    if ($bytes -contains 13) { throw "CR line ending in $path" }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower()
    $manifest += "$hash  $name"
}
$manifestText = ($manifest -join "`n") + "`n"
$manifestSha = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::ASCII.GetBytes($manifestText))).Replace('-', '').ToLower()
$manifest | ForEach-Object { Write-Output $_ }
Write-Output ('MANIFEST_SHA256=' + $manifestSha)
if ($SelfTest) { Write-Output 'DEPLOY_SELFTEST_PASS connections=0'; exit 0 }
$ssh = (Get-Command ssh).Source; $scp = (Get-Command scp).Source
$lock = (& $ssh -o BatchMode=yes $HostAlias "test ! -d $root/lock/tbench.lock && echo FREE || cat $root/lock/tbench.lock/owner.json").Trim()
if ($LASTEXITCODE -ne 0) { throw "remote lock check failed: $lock" }
if ($lock -ne 'FREE') {
    $alive = (& $ssh -o BatchMode=yes $HostAlias "/usr/bin/python3 -I -B $root/bin/gswe32_lock.py $root status").Trim()
    if ($alive -match '"alive": true') { throw "tbench lock held by a live run: $alive" }
}
$tmp = (& $ssh -o BatchMode=yes $HostAlias "mkdir -p $root/bin $root/upstream $root/tasks $root/runs $root/sock $root/lock $root/packages $root/mswea-config && chmod 700 $root/sock && mktemp -d /tmp/gswe32-deploy-XXXXXX").Trim()
if ($LASTEXITCODE -ne 0 -or $tmp -notmatch '^/tmp/gswe32-deploy-[A-Za-z0-9]+$') { throw "remote setup failed: $tmp" }
$i = 0
foreach ($name in $names) {
    & $scp -o BatchMode=yes -q $files[$name] ("${HostAlias}:$tmp/f$i")
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $name" }
    $i++
}
$lines = @(); $i = 0
foreach ($entry in $manifest) { $hash, $name = $entry -split '  ', 2; $lines += "$hash $name f$i"; $i++ }
$manifestB64 = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($manifestText))
$script = @"
set -euo pipefail
cd $tmp
while read -r hash name file; do
  [[ `$(sha256sum "`$file" | cut -d' ' -f1) == "`$hash" ]] || { echo "DEPLOY_HASH_MISMATCH `$name"; exit 1; }
done <<'LIST'
$($lines -join "`n")
LIST
while read -r hash name file; do
  mode=644; [[ `$name == *.sh ]] && mode=755
  install -m "`$mode" "`$file" "$root/`$name"
done <<'LIST'
$($lines -join "`n")
LIST
echo $manifestB64 | base64 -d > $root/bin/MANIFEST.sha256
cd / && rm -rf "$tmp"
(cd $root/upstream && sha256sum --check --quiet SHA256SUMS)
cd $root && sha256sum bin/MANIFEST.sha256
echo DEPLOY_OK
"@
$script = $script -replace "`r", ''
# Windows PowerShell can prefix piped stdin with a UTF-8 BOM: the script travels base64-encoded in the command.
$b64 = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($script))
$out = & $ssh -o BatchMode=yes $HostAlias "echo $b64 | base64 -d | bash"
$out | ForEach-Object { Write-Output $_ }
if ($LASTEXITCODE -ne 0 -or -not ($out -contains 'DEPLOY_OK')) { throw 'deploy failed' }
Write-Output ('DEPLOYED root=' + $root + ' manifest_sha256=' + $manifestSha)
