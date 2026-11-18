function ConvertTo-BashLiteral([string]$Value) {
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function New-OneConnectionStartInfo {
    param([string]$Executable,[string[]]$PrefixArguments=@(),[string]$RemoteCommand)
    $Info = New-Object System.Diagnostics.ProcessStartInfo
    $Info.FileName = $Executable
    $AllArguments = @($PrefixArguments) + @('-T','-o','ConnectionAttempts=1','ycao11@gh-login03.delta.ncsa.illinois.edu',$RemoteCommand)
    $Info.Arguments = ($AllArguments | ForEach-Object { ConvertTo-WindowsArgument $_ }) -join ' '
    $Info.UseShellExecute = $false
    # Windows OpenSSH password/Duo prompts need the caller's visible console.
    $Info.CreateNoWindow = $false
    $Info.RedirectStandardInput = $true
    $Info.RedirectStandardOutput = $false
    $Info.RedirectStandardError = $false
    return $Info
}

function Invoke-OneConnectionTransport {
    param([string]$Executable,[string[]]$PrefixArguments=@(),[string]$RemoteCommand,[byte[]]$PayloadBytes,
          [int]$DeliveryTimeoutSeconds=180,[int]$TotalTimeoutSeconds=14400)
    $Info = New-OneConnectionStartInfo -Executable $Executable -PrefixArguments $PrefixArguments -RemoteCommand $RemoteCommand
    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $Info
    $InputClosed = $false
    $Clock = [Diagnostics.Stopwatch]::StartNew()
    $NextNotice = 15
    # .NET Framework creates the stdin writer with Console.InputEncoding and
    # AutoFlush, which emits a BOM first when that encoding has one (UTF-8
    # consoles, code page 65001). Start under a preamble-free encoding instead.
    $PreviousInputEncoding = $null
    if ([Console]::InputEncoding.GetPreamble().Length -gt 0) {
        try { $PreviousInputEncoding = [Console]::InputEncoding; [Console]::InputEncoding = [Text.UTF8Encoding]::new($false) }
        catch { $PreviousInputEncoding = $null }
    }
    try {
        [Console]::Error.WriteLine('SSH_CONNECTING: password/Duo prompts use this visible console. No automatic retry.')
        try { $Started = $Process.Start() }
        finally { if ($PreviousInputEncoding) { [Console]::InputEncoding = $PreviousInputEncoding } }
        if (-not $Started) { throw 'SSH did not start; no retry' }
        $WriteFailure = $null
        # Never block the supervising thread on a full pre-authentication pipe.
        $WriteTask = $Process.StandardInput.BaseStream.WriteAsync($PayloadBytes,0,$PayloadBytes.Length)
        while (-not $Process.WaitForExit(100)) {
            if (-not $InputClosed -and $WriteTask.IsCompleted) {
                try { [void]$WriteTask.GetAwaiter().GetResult() }
                catch { $WriteFailure = $_ }
                try { $Process.StandardInput.BaseStream.Close() }
                catch { if (-not $WriteFailure) { $WriteFailure = $_ } }
                $InputClosed = $true
                [Console]::Error.WriteLine('SSH_PAYLOAD_DELIVERY_FINISHED: awaiting remote result; this is not submission confirmation.')
            }
            if ((-not $InputClosed -and $Clock.Elapsed.TotalSeconds -ge $DeliveryTimeoutSeconds) -or
                $Clock.Elapsed.TotalSeconds -ge $TotalTimeoutSeconds) {
                [Console]::Error.WriteLine('SSH_LOCAL_DEADLINE: stopping only the local SSH process. Remote outcome unknown; do not retry.')
                $Process.Kill()
                [void]$Process.WaitForExit(5000)
                throw 'SSH delivery/flow deadline expired; remote outcome unknown. Preserve stage/claim and do not retry.'
            }
            if ($Clock.Elapsed.TotalSeconds -ge $NextNotice) {
                [Console]::Error.WriteLine('SSH_WAITING: complete password/Duo in this console if requested; remote outcome not yet known.')
                $NextNotice += 15
            }
        }
        $Code = $Process.ExitCode
        if (-not $InputClosed) {
            if ($WriteTask.IsCompleted) {
                try { [void]$WriteTask.GetAwaiter().GetResult() }
                catch { $WriteFailure = $_ }
            }
            try { $Process.StandardInput.BaseStream.Close() }
            catch { if (-not $WriteFailure) { $WriteFailure = $_ } }
            $InputClosed = $true
        }
        if ($WriteFailure -and $Code -eq 0) { throw $WriteFailure }
        return $Code
    }
    finally {
        if (-not $InputClosed) { try { $Process.StandardInput.Close() } catch {} }
        $Process.Dispose()
    }
}

function ConvertTo-WindowsArgument([string]$Value) {
    # Microsoft CRT argv quoting. No cmd.exe/PowerShell command interpretation.
    # Backslashes are literal except before a quote or the final closing quote.
    $Builder = New-Object Text.StringBuilder
    [void]$Builder.Append('"')
    $Backslashes = 0
    foreach ($Character in $Value.ToCharArray()) {
        if ($Character -eq '\') { $Backslashes++; continue }
        if ($Character -eq '"') {
            [void]$Builder.Append(('\' * (2 * $Backslashes + 1)))
            [void]$Builder.Append('"')
        }
        else {
            [void]$Builder.Append(('\' * $Backslashes))
            [void]$Builder.Append($Character)
        }
        $Backslashes = 0
    }
    [void]$Builder.Append(('\' * (2 * $Backslashes)))
    [void]$Builder.Append('"')
    return $Builder.ToString()
}
