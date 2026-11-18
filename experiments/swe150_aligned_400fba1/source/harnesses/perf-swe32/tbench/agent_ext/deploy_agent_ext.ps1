# Install the keep-reasoning agents' model class (swe32_keep_reasoning.py, 2026-11-16) next to the pinned runner on
# tbench: /home/ubuntu/swe32/upstream/swe32_keep_reasoning.py (the directory swebench_runner.py puts first on sys.path).
# The file is NOT part of the deployed half's manifest (bin/MANIFEST.sha256 stays as pinned, so the queued jobs'
# preflight is unchanged); the keep plans' login gate compares its SHA-256 with config.stock.thinking.kept. Refuses
# unless the local file equals that pin; uploads through a temporary directory, installs only if the SHA-256 matches,
# never touches another file. Uses the `tbench` SSH alias of this machine. Run from a Windows PowerShell:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File '<perf-swe32>\tbench\agent_ext\deploy_agent_ext.ps1'
# -SelfTest: check the pin only, no connection.
# -Kind edge (2026-11-16): the scripted edge-case client instead (swe32_edge_script.py, pinned in
# config.agent.edge_script; plan keepedge), the same way. Default keep (the command above is unchanged).
param([switch]$SelfTest, [string]$HostAlias = 'tbench', [ValidateSet('keep', 'edge')][string]$Kind = 'keep')
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$harness = Split-Path -Parent (Split-Path -Parent $here)
$config = Get-Content -LiteralPath (Join-Path $harness 'swe32-config.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Kind -eq 'edge') {
  $file = Join-Path $here 'swe32_edge_script.py'
  $module = $config.agent.edge_script.agent_module
  $expected = '/home/ubuntu/swe32/upstream/swe32_edge_script.py'
} else {
  $file = Join-Path $here 'swe32_keep_reasoning.py'
  $module = $config.stock.thinking.kept.agent_module
  $expected = '/home/ubuntu/swe32/upstream/swe32_keep_reasoning.py'
}
$bytes = [IO.File]::ReadAllBytes($file)
if ($bytes -contains 13) { throw "CR line ending in $file" }
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLower()
if ($hash -cne $module.sha256) { throw "local $file is $hash, the config pins $($module.sha256)" }
$target = [string]$module.tbench_path
if ($target -cne $expected) { throw "unexpected target $target" }
Write-Output ('AGENT_MODULE sha256=' + $hash + ' target=' + $target)
if ($SelfTest) { Write-Output 'DEPLOY_AGENT_EXT_SELFTEST_PASS connections=0'; exit 0 }
$ssh = (Get-Command ssh).Source; $scp = (Get-Command scp).Source
$tmp = (& $ssh -o BatchMode=yes $HostAlias "test -d /home/ubuntu/swe32/upstream && mktemp -d /tmp/swe32-agent-ext-XXXXXX").Trim()
if ($LASTEXITCODE -ne 0 -or $tmp -notmatch '^/tmp/swe32-agent-ext-[A-Za-z0-9]+$') { throw "remote setup failed: $tmp" }
& $scp -o BatchMode=yes -q $file ("${HostAlias}:$tmp/f0")
if ($LASTEXITCODE -ne 0) { throw 'scp failed' }
$script = "set -euo pipefail; cd $tmp; [[ `$(sha256sum f0 | cut -d' ' -f1) == $hash ]] || { echo DEPLOY_HASH_MISMATCH; exit 1; }; install -m 644 f0 $target; cd / && rm -rf $tmp; sha256sum $target; echo DEPLOY_OK"
$b64 = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($script))
$out = & $ssh -o BatchMode=yes $HostAlias "echo $b64 | base64 -d | bash"
$out | ForEach-Object { Write-Output $_ }
if ($LASTEXITCODE -ne 0 -or -not ($out -contains 'DEPLOY_OK') -or -not (($out -join "`n").Contains("$hash  $target"))) { throw 'deploy failed' }
Write-Output ('DEPLOYED target=' + $target + ' sha256=' + $hash)
