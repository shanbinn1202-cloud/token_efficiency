$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'transport.ps1')
$Python=(Get-Command python -CommandType Application | Select-Object -First 1).Source
$Raw=[IO.File]::ReadAllBytes((Join-Path $PSScriptRoot 'dist\payload.json'))
$Remote='python -I -S -B -c ''import base64;exec(base64.b64decode("eA=="))'' 0123456789abcdef0123456789abcdef'
foreach ($Code in @(0,37)) {
    $Record=Join-Path ([IO.Path]::GetTempPath()) ('transport-fixture-'+[Guid]::NewGuid().ToString('N')+'.json')
    try {
        $Actual=Invoke-OneConnectionTransport -Executable $Python -PrefixArguments @((Join-Path $PSScriptRoot 'fake_ssh.py'),'--record',$Record,'--code',[string]$Code,'--probe','space "quoted" \ ending\') -RemoteCommand $Remote -PayloadBytes $Raw
        $Value=[IO.File]::ReadAllText($Record) | ConvertFrom-Json
        if ($Actual -ne $Code -or $Value.argv.Count -ne 5 -or $Value.argv[-1] -cne $Remote) { throw 'exit/argv changed across Windows transport' }
        if ($Value.probe -cne 'space "quoted" \ ending\' -or $Value.payload_base64 -cne [Convert]::ToBase64String($Raw)) { throw 'byte/quote roundtrip failed' }
    } finally {
        if (Test-Path -LiteralPath $Record) { Remove-Item -LiteralPath $Record }
    }
}
Write-Output 'FAKE_SSH_TRANSPORT_PASS byte_exact=pass quoted_argv=pass exit_0_37=pass connections=0'
