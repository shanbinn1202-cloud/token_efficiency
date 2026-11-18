[CmdletBinding()]
param(
    [switch]$Authorize,
    [string]$Authorization = '',
    # One sealed plan per submission (README "Plans"): Qwen3.6 g1, g2, g4o, g4g, g4v or Gemma 4 gm1, gm4o, gm4g, gm4v,
    # gm4r on the ghx4 batch partition; gm4oi, gm4gi, gm4ri their ghx4-interactive variants (urgent runs only, one
    # submitted job per user, billed 2x); smq, smg the release smoke plans (integrate-1103); smqt (kv-session thinking
    # on) and smga (Gemma async) their main ba4d1ba variants; smqc the async-maxtok-1112 case cell (11-15). Always name
    # it.
    [string]$Plan = '',
    [string]$RunLabel = 'r1',
    [switch]$SelfTest,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$PinnedPayload = 'bd816111e5a9efd05d0b19229df93da2e98d507539cccbe5185046ff0e417a7b'
$ExpectedCommit = '4ebff978661f4c5e2e06da0ee68a5217f92dfa82'
$ExpectedConfig = 'fd7de34fd95b2bedc60a311252a8794b0d984b5eb7f40fa6c4bb3b9afd136f20'
$ExpectedPlanSha = '036406829749117db71d839177c65019e3e528c5437b0cfcd25689d12ccdb353'
# Light login gates: the login node runs no runtime test suite; the local release-check receipt of the commit is pinned.
$ExpectedReceipt = '308a6e79fc938fd2e2b62a87ebd7b7760703df31befac715fdb966459b30ea8f'
$ExpectedReceiptLog = '7bb8bcbd486c873fcab85964d9f59f4213c503fe4c8a2a785bbc424b54b82a6c'
$ExpectedCleanEnv = '/u/ycao11/spq9-277fa5f1390947e08caefecc8e928a3d/clean-env'
$StagePrefix = '/u/ycao11/pmab-'
# The pinned plans (build.py writes them from pins.json config): phrase, partition, Slurm time, expected time, servers.
$Plans = [ordered]@{
    'g1' = @{authorization='AUTHORIZE_PMAB_G1_KVSESSION_FULL_OBS_STOCK_REPLAY_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY'; billing_factor='1'; expected_seconds='3212'; factor='1.308'; gpu_hours_expected='0.89'; gpu_hours_limit='1.17'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='55699'; optional='g1v'; partition='ghx4'; required='g1a,g1b,g1c'; required_minimum_seconds='2930'; required_spare_seconds='1270'; servers='g1a:required:897s,g1b:required:837s,g1c:required:653s,g1v:optional:435s'; startup_spike='ELIMINATED'; switches='-'; time='01:10:00'}
    'g2' = @{authorization='AUTHORIZE_PMAB_G2_ROUTEB_16SESSIONS_16_24GIB_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY'; billing_factor='1'; expected_seconds='3308'; factor='1.27'; gpu_hours_expected='0.92'; gpu_hours_limit='1.17'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='81197'; optional='g2c'; partition='ghx4'; required='g2a,g2b'; required_minimum_seconds='2370'; required_spare_seconds='1830'; servers='g2a:required:1054s,g2b:required:1054s,g2c:optional:870s'; startup_spike='ELIMINATED'; switches='-'; time='01:10:00'}
    'g4o' = @{authorization='AUTHORIZE_PMAB_G4O_REAL32_OBSQUERY_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_47MIN_NO_RETRY'; billing_factor='2'; expected_seconds='2164'; factor='1.303'; gpu_hours_expected='1.20'; gpu_hours_limit='1.57'; mem='227120M'; mem_max_mib='227120'; mem_need_mib='168237'; optional='-'; partition='ghx4'; required='g4o'; required_minimum_seconds='2150'; required_spare_seconds='670'; servers='g4o:required:1834s'; startup_spike='ELIMINATED'; switches='-'; time='00:47:00'}
    'g4g' = @{authorization='AUTHORIZE_PMAB_G4G_REAL32_GOALQUERY_REWRITER_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_61MIN_NO_RETRY'; billing_factor='2'; expected_seconds='2804'; factor='1.305'; gpu_hours_expected='1.56'; gpu_hours_limit='2.03'; mem='227120M'; mem_max_mib='227120'; mem_need_mib='168237'; optional='-'; partition='ghx4'; required='g4g'; required_minimum_seconds='2750'; required_spare_seconds='910'; servers='g4g:required:2474s'; startup_spike='ELIMINATED'; switches='-'; time='01:01:00'}
    'g4v' = @{authorization='AUTHORIZE_PMAB_G4V_REAL32_STOCK_VANILLA_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_87MIN_NO_RETRY'; billing_factor='1'; expected_seconds='4024'; factor='1.297'; gpu_hours_expected='1.12'; gpu_hours_limit='1.45'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='37888'; optional='-'; partition='ghx4'; required='g4v'; required_minimum_seconds='3390'; required_spare_seconds='1830'; servers='g4v:required:3634s'; startup_spike='ELIMINATED'; switches='-'; time='01:27:00'}
    'gm1' = @{authorization='AUTHORIZE_PMAB_GM1_GEMMA4_KVSESSION_FULL_OBS_STOCK_REPLAY_BATCH_REUSED_V9_ENV_ONE_GH200_75MIN_NO_RETRY'; billing_factor='1'; expected_seconds='3390'; factor='1.327'; gpu_hours_expected='0.94'; gpu_hours_limit='1.25'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='49367'; optional='gm1v'; partition='ghx4'; required='gm1a,gm1b,gm1c'; required_minimum_seconds='3070'; required_spare_seconds='1430'; servers='gm1a:required:963s,gm1b:required:903s,gm1c:required:666s,gm1v:optional:448s'; startup_spike='ELIMINATED'; switches='-'; time='01:15:00'}
    'gm4o' = @{authorization='AUTHORIZE_PMAB_GM4O_GEMMA4_E2E8_OBSQUERY_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY'; billing_factor='1'; expected_seconds='1968'; factor='1.494'; gpu_hours_expected='0.55'; gpu_hours_limit='0.82'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='100367'; optional='-'; partition='ghx4'; required='gm4o'; required_minimum_seconds='1930'; required_spare_seconds='1010'; servers='gm4o:required:1618s'; startup_spike='ELIMINATED'; switches='gm4o:kv25,fastpath1-3,deferred_io'; time='00:49:00'}
    'gm4g' = @{authorization='AUTHORIZE_PMAB_GM4G_GEMMA4_E2E8_GOALQUERY_REWRITER_B4_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_56MIN_NO_RETRY'; billing_factor='1'; expected_seconds='2297'; factor='1.463'; gpu_hours_expected='0.64'; gpu_hours_limit='0.93'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='100367'; optional='-'; partition='ghx4'; required='gm4g'; required_minimum_seconds='2230'; required_spare_seconds='1130'; servers='gm4g:required:1947s'; startup_spike='ELIMINATED'; switches='gm4g:kv25,fastpath1-3,deferred_io,batch4/T25600'; time='00:56:00'}
    'gm4v' = @{authorization='AUTHORIZE_PMAB_GM4V_GEMMA4_REAL32_STOCK_VANILLA_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_345MIN_NO_RETRY'; billing_factor='1'; expected_seconds='15812'; factor='1.309'; gpu_hours_expected='4.39'; gpu_hours_limit='5.75'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='37888'; optional='-'; partition='ghx4'; required='gm4v'; required_minimum_seconds='15510'; required_spare_seconds='5190'; servers='gm4v:required:15402s'; startup_spike='ELIMINATED'; switches='-'; time='05:45:00'}
    'gm4r' = @{authorization='AUTHORIZE_PMAB_GM4R_GEMMA4_E2E8_RANDOMFILL_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY'; billing_factor='1'; expected_seconds='1968'; factor='1.494'; gpu_hours_expected='0.55'; gpu_hours_limit='0.82'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='100367'; optional='-'; partition='ghx4'; required='gm4r'; required_minimum_seconds='1930'; required_spare_seconds='1010'; servers='gm4r:required:1618s'; startup_spike='ELIMINATED'; switches='gm4r:kv25,fastpath1-3,deferred_io'; time='00:49:00'}
    'gm4oi' = @{authorization='AUTHORIZE_PMAB_GM4OI_GEMMA4_E2E8_OBSQUERY_25GIB_FAST3_DCIO_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY'; billing_factor='2'; expected_seconds='1968'; factor='1.494'; gpu_hours_expected='1.09'; gpu_hours_limit='1.63'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='100367'; optional='-'; partition='ghx4-interactive'; required='gm4o'; required_minimum_seconds='1930'; required_spare_seconds='1010'; servers='gm4o:required:1618s'; startup_spike='ELIMINATED'; switches='gm4o:kv25,fastpath1-3,deferred_io'; time='00:49:00'}
    'gm4gi' = @{authorization='AUTHORIZE_PMAB_GM4GI_GEMMA4_E2E8_GOALQUERY_REWRITER_B4_25GIB_FAST3_DCIO_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_56MIN_NO_RETRY'; billing_factor='2'; expected_seconds='2297'; factor='1.463'; gpu_hours_expected='1.28'; gpu_hours_limit='1.87'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='100367'; optional='-'; partition='ghx4-interactive'; required='gm4g'; required_minimum_seconds='2230'; required_spare_seconds='1130'; servers='gm4g:required:1947s'; startup_spike='ELIMINATED'; switches='gm4g:kv25,fastpath1-3,deferred_io,batch4/T25600'; time='00:56:00'}
    'gm4ri' = @{authorization='AUTHORIZE_PMAB_GM4RI_GEMMA4_E2E8_RANDOMFILL_25GIB_FAST3_DCIO_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY'; billing_factor='2'; expected_seconds='1968'; factor='1.494'; gpu_hours_expected='1.09'; gpu_hours_limit='1.63'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='100367'; optional='-'; partition='ghx4-interactive'; required='gm4r'; required_minimum_seconds='1930'; required_spare_seconds='1010'; servers='gm4r:required:1618s'; startup_spike='ELIMINATED'; switches='gm4r:kv25,fastpath1-3,deferred_io'; time='00:49:00'}
    'smq' = @{authorization='AUTHORIZE_PMAB_SMQ_RELEASE_SMOKE_QWEN_OBS_RFILL_GOAL_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_110MIN_NO_RETRY'; billing_factor='2'; expected_seconds='5093'; factor='1.296'; gpu_hours_expected='2.83'; gpu_hours_limit='3.67'; mem='227120M'; mem_max_mib='227120'; mem_need_mib='168237'; optional='-'; partition='ghx4'; required='smqo,smqr,smqg'; required_minimum_seconds='5040'; required_spare_seconds='1560'; servers='smqo:required:1433s,smqr:required:1433s,smqg:required:1897s'; startup_spike='ELIMINATED'; switches='smqo:kv32,fastpath1-3,deferred_io,async;smqr:kv32,fastpath1-3,deferred_io,async;smqg:kv32,fastpath1-3,deferred_io,vllm_rewriter3g,goal_overlap,async'; time='01:50:00'}
    'smg' = @{authorization='AUTHORIZE_PMAB_SMG_RELEASE_SMOKE_GEMMA4_OBS_RFILL_GOAL_8SESSIONS_25GIB_BATCH_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY'; billing_factor='1'; expected_seconds='5533'; factor='1.301'; gpu_hours_expected='1.54'; gpu_hours_limit='2.00'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='100367'; optional='-'; partition='ghx4'; required='smgo,smgr,smgg'; required_minimum_seconds='5390'; required_spare_seconds='1810'; servers='smgo:required:1618s,smgr:required:1618s,smgg:required:1947s'; startup_spike='ELIMINATED'; switches='smgo:kv25,fastpath1-3,deferred_io;smgr:kv25,fastpath1-3,deferred_io;smgg:kv25,fastpath1-3,deferred_io,vllm_rewriter3g,goal_overlap'; time='02:00:00'}
    'smqt' = @{authorization='AUTHORIZE_PMAB_SMQT_RELEASE_SMOKE_QWEN_THINKING_OBS_RFILL_GOAL_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_110MIN_NO_RETRY'; billing_factor='2'; expected_seconds='5093'; factor='1.296'; gpu_hours_expected='2.83'; gpu_hours_limit='3.67'; mem='227120M'; mem_max_mib='227120'; mem_need_mib='168237'; optional='-'; partition='ghx4'; required='smqot,smqrt,smqgt'; required_minimum_seconds='5040'; required_spare_seconds='1560'; servers='smqot:required:1433s,smqrt:required:1433s,smqgt:required:1897s'; startup_spike='ELIMINATED'; switches='smqot:kv32,fastpath1-3,deferred_io,async,thinking_on;smqrt:kv32,fastpath1-3,deferred_io,async,thinking_on;smqgt:kv32,fastpath1-3,deferred_io,vllm_rewriter3g,goal_overlap,async,thinking_on'; time='01:50:00'}
    'smga' = @{authorization='AUTHORIZE_PMAB_SMGA_RELEASE_SMOKE_GEMMA4_ASYNC_OBS_RFILL_GOAL_8SESSIONS_25GIB_BATCH_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY'; billing_factor='1'; expected_seconds='5533'; factor='1.301'; gpu_hours_expected='1.54'; gpu_hours_limit='2.00'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='100367'; optional='-'; partition='ghx4'; required='smgoa,smgra,smgga'; required_minimum_seconds='5390'; required_spare_seconds='1810'; servers='smgoa:required:1618s,smgra:required:1618s,smgga:required:1947s'; startup_spike='ELIMINATED'; switches='smgoa:kv25,fastpath1-3,deferred_io,async;smgra:kv25,fastpath1-3,deferred_io,async;smgga:kv25,fastpath1-3,deferred_io,vllm_rewriter3g,goal_overlap,async'; time='02:00:00'}
    'smqc' = @{authorization='AUTHORIZE_PMAB_SMQC_ASYNC_MAXTOK_CASE_QWEN_THINKING_OBS_CUT_CELL_4SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_30MIN_NO_RETRY'; billing_factor='1'; expected_seconds='1169'; factor='1.54'; gpu_hours_expected='0.32'; gpu_hours_limit='0.50'; mem='113560M'; mem_max_mib='113560'; mem_need_mib='72268'; optional='-'; partition='ghx4'; required='smqco'; required_minimum_seconds='1200'; required_spare_seconds='600'; servers='smqco:required:839s'; startup_spike='ELIMINATED'; switches='smqco:kv32,fastpath1-3,deferred_io,async,thinking_on'; time='00:30:00'}
}
# The user's rule for every DeltaAI job: mail BEGIN, END and FAIL to this address; pinned, not a parameter.
$MailUser = 'cao58@illinois.edu'
$MailType = 'BEGIN,END,FAIL'
$PayloadPath = Join-Path $PSScriptRoot 'dist\payload.json'
if ((Get-FileHash -LiteralPath $PayloadPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $PinnedPayload) { throw 'sealed payload hash mismatch' }
$Bundle = [IO.File]::ReadAllText($PayloadPath) | ConvertFrom-Json
foreach ($Property in $Bundle.files.PSObject.Properties) {
    $Path = Join-Path $PSScriptRoot $Property.Name
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Property.Value.sha256) { throw ('helper hash mismatch: '+$Property.Name) }
}
. (Join-Path $PSScriptRoot 'transport.ps1')
$Pins = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Bundle.files.'pins.json'.base64)) | ConvertFrom-Json
if ($Pins.commit -cne $ExpectedCommit -or $Pins.package_sha256 -cne 'f151fd29a2d58cb6aa3de27cfdd876ab5e00cd023a284015c84b440d5ea90e4e' -or $Pins.config_sha256 -cne $ExpectedConfig -or $Pins.workload_plan.sha256 -cne $ExpectedPlanSha -or $Pins.clean_env.prefix -cne $ExpectedCleanEnv) { throw 'sealed pins differ from this entry point' }
$Receipt = $Pins.release_receipt
if (-not $Receipt -or $Receipt.commit -cne $ExpectedCommit -or $Receipt.status -cne 'GIT_ARCHIVE_TESTS_PASS' -or $Receipt.kind -cne 'check_git_release receipt' -or $Receipt.result_line_sha256 -cne $ExpectedReceipt -or $Receipt.log_sha256 -cne $ExpectedReceiptLog -or [int]$Receipt.tests_run -le 0) { throw 'sealed pins lack the passing release-check receipt of the sealed commit (light login gates: the login node runs no runtime test suite)' }
if ($Pins.slurm.mail_user -cne $MailUser -or $Pins.slurm.mail_type -cne $MailType -or $Pins.slurm.job_name_prefix -cne 'pmab-') { throw 'sealed Slurm pins differ from the mail rule' }
if ($Pins.slurm.cpus_per_task -ne 16 -or $null -ne $Pins.slurm.mem) { throw 'sealed Slurm resources differ from --cpus-per-task=16 with a per-plan --mem' }
$Features = $Pins.runtime_features
if ($Features.stage2 -ne $true -or $Features.decoupled_layout -ne $true -or @($Features.page_tokens_choices) -notcontains 16) { throw 'sealed runtime lacks stage 2 or the decoupled 16-token layout' }
if (-not $Features.kv_session -or $Features.kv_session.method -cne 'dram_backed_kv' -or -not $Features.release_idle_pages -or -not $Features.decode_fastpath) { throw 'sealed runtime lacks the kv-session protocol, --release-idle-pages or --decode-fastpath' }
if (@($Features.context_limits.max_model_len_choices) -notcontains 262144) { throw 'sealed runtime lacks the 262144-token context' }
$Server = $Pins.config.runtime_server
if ($Server.protocol -cne 'kv-session' -or $Server.page_tokens -ne 16 -or $Server.max_tokens -ne 4096 -or $Server.max_model_len -ne 262144 -or $null -ne $Server.kv_session_thinking) { throw 'sealed runtime server settings differ from the plan (kv-session, P 16, 4096, 262144; thinking from generation.enable_thinking)' }
$Gen = $Pins.config.generation
if (($Gen.enable_thinking -ne $true -and $Gen.enable_thinking -ne $false) -or $Gen.max_tokens -ne 4096 -or $Gen.max_turns -ne 120 -or $Gen.temperature -ne 0) { throw 'sealed generation settings differ from the harness config (thinking on, 4096, 120 turns, greedy)' }
if ($null -ne $Pins.config.runner.tokenizer_override) { throw 'a sealed runner tokenizer override (synthetic stages only)' }
if (-not $Plan) { throw ('name the plan: -Plan '+(@($Plans.Keys) -join ' or -Plan ')) }
if (-not $Plans.Contains($Plan)) { throw ('unknown plan '''+$Plan+''' (one of '+(@($Plans.Keys) -join ', ')+')') }
$Chosen = $Plans[$Plan]
$Sealed = $Pins.config.plans.$Plan
if (-not $Sealed -or $Sealed.authorization -cne $Chosen.authorization -or $Sealed.time -cne $Chosen.time -or $Sealed.partition -cne $Chosen.partition -or $Sealed.mem -cne $Chosen.mem) { throw 'sealed plan differs from this entry point' }
if ($Chosen.mem -cnotmatch '^[1-9][0-9]{0,6}[MG]$') { throw ('plan '+$Plan+' asks --mem='+$Chosen.mem) }
$MemMiB = [int64]$Chosen.mem.Substring(0, $Chosen.mem.Length - 1); if ($Chosen.mem.EndsWith('G')) { $MemMiB *= 1024 }
$MemMax = [int64]$Chosen.mem_max_mib
$TwoShare = @($Pins.config.budget.memory.two_share_plans) + @($Pins.config.gemma.memory.two_share_plans)
if ($MemMax -ne 113560 -and -not ($MemMax -eq 227120 -and $TwoShare -contains $Plan)) { throw ('plan '+$Plan+' claims a memory ceiling of '+$Chosen.mem_max_mib+'M (one GH200 share is 113560M; two only for the sealed two-share plans)') }
# A Gemma 4 plan (config.plans.<plan>.model): the sealed launcher's Gemma kv-session path and the pinned checkpoint.
$ModelNote = ''
if ($Sealed.model) {
    if ($Sealed.model -cne 'gemma4-31b-it' -or $Features.gemma4.profile -cne 'gemma4-31b-it' -or $Features.gemma4.grammar -cne 'gemma4' -or $Features.gemma4.servable -ne $true -or $Features.gemma4.thinking -cne 'off' -or $Pins.config.gemma.model.dir -cne '/projects/bcjw/ycao11/models/gemma-4-31b-it' -or $Pins.config.gemma.model.revision -cne '842da3794eaa0b77d5f08bae87a17459d91ff475') { throw ('plan '+$Plan+': the sealed Gemma 4 kv-session path or checkpoint pin differs') }
    $ModelNote = ' model='+$Sealed.model
}
if ($MemMiB -gt $MemMax -or $MemMiB -lt [int64]$Chosen.mem_need_mib) { throw ('plan '+$Plan+' asks --mem='+$Chosen.mem+', outside its pinned-host need '+$Chosen.mem_need_mib+'M .. '+$MemMax+'M') }
$Interactive = $false
if ($Chosen.partition -ceq 'ghx4-interactive') {
    # Only a sealed interactive variant of a batch plan (its servers, time and memory; the login gate checks the
    # partition's one-job limit by squeue before the claim): urgent runs only, billed 2x.
    $Base = $Pins.config.plans.($Sealed.interactive_of)
    if (-not $Sealed.interactive_of -or -not $Base -or $Base.partition -cne 'ghx4' -or $Base.time -cne $Sealed.time -or $Base.mem -cne $Sealed.mem -or ((@($Base.servers) -join ',') -cne (@($Sealed.servers) -join ','))) { throw ('plan '+$Plan+' is not a sealed ghx4-interactive variant of a batch plan') }
    if ($Pins.slurm.partition_limits.'ghx4-interactive'.max_submitted_jobs_per_user -ne 1) { throw 'Slurm pin of ghx4-interactive changed (one submitted job per user)' }
    $Interactive = $true
} elseif ($Chosen.partition -cne 'ghx4' -or $Sealed.interactive_of) { throw 'perf-mab runs on the ghx4 batch partition (ghx4-interactive only as a sealed variant: urgent runs)' }
$PartitionMax = $Pins.slurm.partition_limits.($Chosen.partition).max_time
if (-not $PartitionMax -or ([TimeSpan]::Parse($Chosen.time) -gt [TimeSpan]::Parse($PartitionMax))) { throw ('plan '+$Plan+' asks --time='+$Chosen.time+', above the '+$Chosen.partition+' maximum '+$PartitionMax) }
if ([int]$Chosen.required_minimum_seconds -gt [TimeSpan]::Parse($Chosen.time).TotalSeconds) { throw ('plan '+$Plan+' does not hold its required minimum within its Slurm time') }
if ($RunLabel -cnotmatch '^r[0-9]{1,2}$') { throw 'RunLabel must match r[0-9]{1,2} (for example r1, r2)' }
$Nonce = [Guid]::NewGuid().ToString('N')
$BootstrapCode = $Bundle.files.'bootstrap.py'.base64
$PythonCode = 'import base64;exec(compile(base64.b64decode("'+$BootstrapCode+'"),"sealed-bootstrap","exec"))'
$RemoteCommand = '/work/nvme/bcjw/ycao11/vllm-arm/bin/python -I -S -B -c ' + (ConvertTo-BashLiteral $PythonCode) + ' ' + $Nonce
$Options = [ordered]@{plan=$Plan; authorization=$Chosen.authorization; mail_user=$MailUser; run_label=$RunLabel}
$Bundle | Add-Member -MemberType NoteProperty -Name options -Value $Options
$Bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Bundle | ConvertTo-Json -Depth 10 -Compress))
$Summary = ' plan='+$Plan+' partition='+$Chosen.partition+' time='+$Chosen.time+' expected_seconds='+$Chosen.expected_seconds+' factor='+$Chosen.factor+' required_minimum_seconds='+$Chosen.required_minimum_seconds+' required_spare_seconds='+$Chosen.required_spare_seconds+' required='+$Chosen.required+' optional='+$Chosen.optional+' servers='+$Chosen.servers+' job_name='+$Pins.slurm.job_name_prefix+$Plan+' run_label='+$RunLabel+' sbatch=--nodes=1,--ntasks=1,--gpus=1,--cpus-per-task='+$Pins.slurm.cpus_per_task+',--mem='+$Chosen.mem+' mem_need_mib='+$Chosen.mem_need_mib+' mem_max_mib='+$Chosen.mem_max_mib+' startup_spike='+$Chosen.startup_spike+' mail_user='+$MailUser+' mail_type='+$MailType+' release_receipt='+$Receipt.result_line_sha256+' login_gates=light billing_factor='+$Chosen.billing_factor+' gpu_hours_limit='+$Chosen.gpu_hours_limit+' gpu_hours_expected='+$Chosen.gpu_hours_expected+' switches='+$Chosen.switches+$ModelNote
if ($SelfTest) {
    $Tokens=$null; $Errors=$null
    [void][Management.Automation.Language.Parser]::ParseFile($PSCommandPath,[ref]$Tokens,[ref]$Errors)
    if ($Errors.Count) { throw 'PowerShell syntax invalid' }
    $Info = New-OneConnectionStartInfo -Executable 'ssh' -RemoteCommand $RemoteCommand
    if ($Info.CreateNoWindow -or $Info.RedirectStandardOutput -or $Info.RedirectStandardError -or -not $Info.RedirectStandardInput) { throw 'visible SSH console contract broken' }
    if (-not $Info.Arguments.Contains('ConnectionAttempts=1')) { throw 'single connection contract broken' }
    $RoundTrip = [Text.Encoding]::UTF8.GetString($Bytes) | ConvertFrom-Json
    if ($RoundTrip.files.'bootstrap.py'.base64 -cne $BootstrapCode -or $RoundTrip.options.plan -cne $Plan -or $RoundTrip.options.run_label -cne $RunLabel -or $RoundTrip.options.mail_user -cne $MailUser -or $RoundTrip.options.authorization -cne $Chosen.authorization) { throw 'sealed JSON binary transport failed' }
    foreach ($Name in 'job.py','job.sh','controller.py','mab_runner.py','mab_workload.py','mab_checks.py','mab_metrics.py','stock.py','measure.py','resources.py','public-release.bundle') { if (-not $RoundTrip.files.$Name.sha256) { throw ('sealed file missing: '+$Name) } }
    if ($Pins.slurm.account -cne 'bcjw-dtai-gh' -or $Pins.slurm.partition_limits.'ghx4'.max_time -cne '48:00:00' -or $Pins.slurm.partition_limits.'ghx4'.phrase_tag -cne 'BATCH' -or $null -ne $Pins.slurm.partition_limits.'ghx4'.max_submitted_jobs_per_user) { throw 'Slurm pin changed' }
    if ($Pins.slurm.partition_limits.'ghx4-interactive'.max_time -cne '02:00:00' -or $Pins.slurm.partition_limits.'ghx4-interactive'.phrase_tag -cne 'INTERACTIVE') { throw 'Slurm pin of ghx4-interactive changed' }
    if ($Pins.clean_env.snapshot_sha256 -cne 'cbfab91650bfbeffb149a2df8ead3787a07b91a2cf332edd7efd0b5b0a07e5f0' -or $Pins.clean_env.release_manifest_sha256 -cne 'b23b5f20e7c85f8f8e3ab93fe51d8519dfd465aa50fecf0cda27c95d18e9af93') { throw 'reused v9 environment pin changed' }
    if ($Pins.wheel_sha256 -cne '700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5' -or -not $Pins.wheel_url.EndsWith('/vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl')) { throw 'official vLLM wheel pin changed' }
    if (([Text.Encoding]::UTF8.GetByteCount($StagePrefix+$Nonce+'/tmp')+1+36) -gt 99) { throw 'stage path exceeds IPC budget' }
    Write-Output ('LOCAL_SELFTEST_PASS connections=0 install=0 submit=0 payload_sha256='+$PinnedPayload+' commit='+$ExpectedCommit+$Summary)
    return
}
if (-not $Authorize) {
    if ($Authorization) { throw 'explicit -Authorize switch also required' }
    Write-Output ('LOCAL_ONLY_PASS connections=0 install=0 submit=0 plan='+$Plan)
    return
}
if ($Authorization -cne $Chosen.authorization) { throw ('exact authorization phrase of plan '+$Plan+' required') }
if ($DryRun) {
    Write-Output ('DRY_RUN connections=0 stage='+$StagePrefix+$Nonce+' payload_bytes='+$Bytes.Length+' commit='+$ExpectedCommit+$Summary)
    return
}
if ($Chosen.startup_spike -cne 'ELIMINATED') { throw ('plan '+$Plan+': '+$Chosen.startup_spike+' (a server would start without --language-model-only). Nothing was connected or submitted.') }
Write-Output ('PLANNED_STAGE='+$StagePrefix+$Nonce+' PLAN='+$Plan)
$Kind = 'ghx4 batch'; if ($Interactive) { $Kind = 'ghx4-interactive (urgent, billed 2x; the login gate refuses it while you have a job there)' }
Write-Output ('One visible SSH/Duo connection: reuse the v9 clean env read-only, private release and light pre-claim gates (about 3-5 min: bundle verification and git fsck, the clean env S0 / S1 comparison and the official wheel RECORD, the private release install, IPC / sampler / compiler probes, a serve.py dry run per runtime server, the stock, runner and real-data gates; no runtime test suite on the login node: the local release-check receipt '+$Receipt.result_line_sha256.Substring(0, 12)+' of commit '+$ExpectedCommit.Substring(0, 7)+' is pinned), then exactly one '+$Kind+' GH200 sbatch of plan '+$Plan+' (--time='+$Chosen.time+', mail on BEGIN, END, FAIL to '+$MailUser+'). Any unknown outcome must not be retried.')
$Code = Invoke-OneConnectionTransport -Executable (Get-Command ssh -CommandType Application | Select-Object -First 1).Source -RemoteCommand $RemoteCommand -PayloadBytes $Bytes
if ($Code -eq 86) { throw 'Pre-claim setup failed: remote confirmed NO_SUBMISSION_CLAIM. No sbatch was reached. Read the printed QUALIFY_ERROR/ERROR line, fix it, then start a separately authorized run with a new -RunLabel; no automatic retry.' }
if ($Code -ne 0) { throw ('Setup failed or submission outcome unknown; consult the printed FAILURE_STATUS/CLAIM_STATE. Without confirmed NO_SUBMISSION_CLAIM, keep the stage and claim and reconcile read-only before any replacement. No automatic retry. SSH exit='+$Code) }
Write-Output 'SUBMITTED: the last JSON line above holds stage and job_id. After the END mail: the README check block, then postrun\fetch_and_verify.ps1 (verify and tables).'
