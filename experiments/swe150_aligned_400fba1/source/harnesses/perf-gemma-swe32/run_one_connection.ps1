[CmdletBinding()]
param(
    [switch]$Authorize,
    [string]$Authorization = '',
    # One sealed plan per submission (README "Plans"): full, goal, obs, rfill, stock - one ghx4 batch job each
    # (32 tasks, 16 running, Gemma 4 31B IT, --mem=113560M; no goalb4: the Gemma profile refuses the rewriter batch);
    # fulli, obsi, rfilli, stocki: their ghx4-interactive variants (urgent runs only: billed 2x, one job per user).
    # goal is NOT READY (rebuilt on the wave-2 head): its SelfTest passes, its DryRun and submission are refused.
    # smoke (2026-11-03, the release smoke matrix J2, role release-smoke): obs4ks -> rfill4ks -> goalvllm4ks in one ghx4
    # batch job (32 tasks, 16 running, 10 steps, 4,096-token forwards, the vLLM Goal rewriter with --goal-overlap);
    # smokehf: its fallback with the docs' Hugging Face batch-4 goal server (only if cg 3508345 fails on the child).
    # -AfterJob <J> queues after job J (afterany): tbench serves one job at a time, so every tbench job of yours
    # (gswe32-, swe32-, swe10-, swe16-, tb16-, gem0-) runs in one chain; the queue gate allows exactly the jobs J waits
    # for (J and its afterany chain). Name the last job of the queued chain. Interactive plans take no -AfterJob: they
    # head a chain, while no other tbench job of yours is queued.
    [string]$Plan = '',
    [string]$RewriterModel = '',
    [string]$RunLabel = 'r1',
    [string]$AfterJob = '',
    [switch]$SelfTest,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$PinnedPayload = 'aaa272a23ebfec9209f1a6d95aebf1c64be268980f3f6da05a9fc218b2d6f31c'
$ExpectedCommit = '4ebff978661f4c5e2e06da0ee68a5217f92dfa82'
$ExpectedConfig = 'b8ab63573b51f26b879d16bd29bce1172849c5508bffb042a8ef62fe73156342'
$StatementsSealed = 'yes'
# Light login gates: the local check_git_release.py receipt of the sealed commit (pins.release_receipt); the login node
# runs no runtime test suite and compares the receipt with the cloned HEAD.
$ExpectedReceipt = '308a6e79fc938fd2e2b62a87ebd7b7760703df31befac715fdb966459b30ea8f'
$ExpectedReceiptLog = '7bb8bcbd486c873fcab85964d9f59f4213c503fe4c8a2a785bbc424b54b82a6c'
$ExpectedCleanEnv = '/u/ycao11/spq9-277fa5f1390947e08caefecc8e928a3d/clean-env'
$StagePrefix = '/u/ycao11/gswe32-'
# The pinned plans (build.py writes them from pins.json config): phrase, partition, Slurm time, time fit, servers.
$Plans = [ordered]@{
    'full' = @{all_caps_fit='true'; authorization='AUTHORIZE_GSWE32_FULL_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY'; billing_factor='1'; cap_after_wait_fits='true'; eval_wait_seconds='600'; expected_seconds='3120'; gpu_free_mib='540'; gpu_hours_expected='0.87'; gpu_hours_limit='1.17'; gpu_peak_mib='97331'; gpu_status='BELOW_MIN_FREE'; guaranteed_seconds='4200'; kv_cache_gib='32'; margin_seconds='300'; memory_need_gib='105.22'; memory_spare_gib='5.68'; not_ready=''; partition='ghx4'; servers='full:full:32tasks:120steps:2280s'; spare_seconds='0'; startup_spike='ELIMINATED'; time='01:10:00'; with_eval_wait_seconds='3720'; worst_case_seconds='4200'}
    'goal' = @{all_caps_fit='true'; authorization='AUTHORIZE_GSWE32_GOAL_GEMMA4_32TASKS_16SESSIONS_REWRITER_BATCH_REUSED_V9_ENV_ONE_GH200_121MIN_NO_RETRY'; billing_factor='1'; cap_after_wait_fits='true'; eval_wait_seconds='600'; expected_seconds='5580'; gpu_free_mib='2200'; gpu_hours_expected='1.55'; gpu_hours_limit='2.02'; gpu_peak_mib='95671'; gpu_status='LEAVES_MIN_FREE'; guaranteed_seconds='7260'; kv_cache_gib='25'; margin_seconds='300'; memory_need_gib='105.22'; memory_spare_gib='5.68'; not_ready=''; partition='ghx4'; servers='goal:goal_query:32tasks:120steps:5340s:batch4:T15360'; spare_seconds='0'; startup_spike='ELIMINATED'; time='02:01:00'; with_eval_wait_seconds='5620'; worst_case_seconds='7260'}
    'obs' = @{all_caps_fit='true'; authorization='AUTHORIZE_GSWE32_OBS_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'; billing_factor='1'; cap_after_wait_fits='true'; eval_wait_seconds='600'; expected_seconds='3600'; gpu_free_mib='7708'; gpu_hours_expected='1.00'; gpu_hours_limit='1.30'; gpu_peak_mib='90163'; gpu_status='LEAVES_MIN_FREE'; guaranteed_seconds='4680'; kv_cache_gib='25'; margin_seconds='300'; memory_need_gib='105.22'; memory_spare_gib='5.68'; not_ready=''; partition='ghx4'; servers='obs:observation_query:32tasks:120steps:2760s'; spare_seconds='0'; startup_spike='ELIMINATED'; time='01:18:00'; with_eval_wait_seconds='4070'; worst_case_seconds='4680'}
    'obs4k' = @{all_caps_fit='true'; authorization='AUTHORIZE_GSWE32_OBS4K_GEMMA4_32TASKS_16SESSIONS_FORWARD4096_BATCH_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'; billing_factor='1'; cap_after_wait_fits='true'; eval_wait_seconds='600'; expected_seconds='3600'; gpu_free_mib='7708'; gpu_hours_expected='1.00'; gpu_hours_limit='1.30'; gpu_peak_mib='90163'; gpu_status='LEAVES_MIN_FREE'; guaranteed_seconds='4680'; kv_cache_gib='25'; margin_seconds='300'; memory_need_gib='105.22'; memory_spare_gib='5.68'; not_ready=''; partition='ghx4'; servers='obs4k:observation_query:32tasks:120steps:2760s:forward4096'; spare_seconds='0'; startup_spike='ELIMINATED'; time='01:18:00'; with_eval_wait_seconds='4070'; worst_case_seconds='4680'}
    'rfill' = @{all_caps_fit='true'; authorization='AUTHORIZE_GSWE32_RFILL_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY'; billing_factor='1'; cap_after_wait_fits='true'; eval_wait_seconds='600'; expected_seconds='3720'; gpu_free_mib='7708'; gpu_hours_expected='1.03'; gpu_hours_limit='1.35'; gpu_peak_mib='90163'; gpu_status='LEAVES_MIN_FREE'; guaranteed_seconds='4860'; kv_cache_gib='25'; margin_seconds='300'; memory_need_gib='105.22'; memory_spare_gib='5.68'; not_ready=''; partition='ghx4'; servers='rfill:random_fill:32tasks:120steps:2940s'; spare_seconds='0'; startup_spike='ELIMINATED'; time='01:21:00'; with_eval_wait_seconds='4170'; worst_case_seconds='4860'}
    'stock' = @{all_caps_fit='true'; authorization='AUTHORIZE_GSWE32_STOCK_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_65MIN_NO_RETRY'; billing_factor='1'; cap_after_wait_fits='true'; eval_wait_seconds='600'; expected_seconds='2760'; gpu_free_mib='6998'; gpu_hours_expected='0.77'; gpu_hours_limit='1.08'; gpu_peak_mib='90873'; gpu_status='STOCK_MEASURED'; guaranteed_seconds='3900'; kv_cache_gib='vllm-default'; margin_seconds='300'; memory_need_gib='32.00'; memory_spare_gib='78.90'; not_ready=''; partition='ghx4'; servers='stock:stock:32tasks:120steps:2460s'; spare_seconds='0'; startup_spike='ELIMINATED'; time='01:05:00'; with_eval_wait_seconds='3340'; worst_case_seconds='3900'}
    'fulli' = @{all_caps_fit='true'; authorization='AUTHORIZE_GSWE32_FULLI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY'; billing_factor='2'; cap_after_wait_fits='true'; eval_wait_seconds='600'; expected_seconds='3120'; gpu_free_mib='540'; gpu_hours_expected='0.87'; gpu_hours_limit='1.17'; gpu_peak_mib='97331'; gpu_status='BELOW_MIN_FREE'; guaranteed_seconds='4200'; kv_cache_gib='32'; margin_seconds='300'; memory_need_gib='105.22'; memory_spare_gib='5.68'; not_ready=''; partition='ghx4-interactive'; servers='full:full:32tasks:120steps:2280s'; spare_seconds='0'; startup_spike='ELIMINATED'; time='01:10:00'; with_eval_wait_seconds='3720'; worst_case_seconds='4200'}
    'obsi' = @{all_caps_fit='true'; authorization='AUTHORIZE_GSWE32_OBSI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'; billing_factor='2'; cap_after_wait_fits='true'; eval_wait_seconds='600'; expected_seconds='3600'; gpu_free_mib='7708'; gpu_hours_expected='1.00'; gpu_hours_limit='1.30'; gpu_peak_mib='90163'; gpu_status='LEAVES_MIN_FREE'; guaranteed_seconds='4680'; kv_cache_gib='25'; margin_seconds='300'; memory_need_gib='105.22'; memory_spare_gib='5.68'; not_ready=''; partition='ghx4-interactive'; servers='obs:observation_query:32tasks:120steps:2760s'; spare_seconds='0'; startup_spike='ELIMINATED'; time='01:18:00'; with_eval_wait_seconds='4070'; worst_case_seconds='4680'}
    'rfilli' = @{all_caps_fit='true'; authorization='AUTHORIZE_GSWE32_RFILLI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY'; billing_factor='2'; cap_after_wait_fits='true'; eval_wait_seconds='600'; expected_seconds='3720'; gpu_free_mib='7708'; gpu_hours_expected='1.03'; gpu_hours_limit='1.35'; gpu_peak_mib='90163'; gpu_status='LEAVES_MIN_FREE'; guaranteed_seconds='4860'; kv_cache_gib='25'; margin_seconds='300'; memory_need_gib='105.22'; memory_spare_gib='5.68'; not_ready=''; partition='ghx4-interactive'; servers='rfill:random_fill:32tasks:120steps:2940s'; spare_seconds='0'; startup_spike='ELIMINATED'; time='01:21:00'; with_eval_wait_seconds='4170'; worst_case_seconds='4860'}
    'stocki' = @{all_caps_fit='true'; authorization='AUTHORIZE_GSWE32_STOCKI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_65MIN_NO_RETRY'; billing_factor='2'; cap_after_wait_fits='true'; eval_wait_seconds='600'; expected_seconds='2760'; gpu_free_mib='6998'; gpu_hours_expected='0.77'; gpu_hours_limit='1.08'; gpu_peak_mib='90873'; gpu_status='STOCK_MEASURED'; guaranteed_seconds='3900'; kv_cache_gib='vllm-default'; margin_seconds='300'; memory_need_gib='32.00'; memory_spare_gib='78.90'; not_ready=''; partition='ghx4-interactive'; servers='stock:stock:32tasks:120steps:2460s'; spare_seconds='0'; startup_spike='ELIMINATED'; time='01:05:00'; with_eval_wait_seconds='3340'; worst_case_seconds='3900'}
    'smoke' = @{all_caps_fit='true'; authorization='AUTHORIZE_GSWE32_SMOKE_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_VLLM_REWRITER_GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_153MIN_NO_RETRY'; billing_factor='1'; cap_after_wait_fits='true'; condition='goal_query on the vLLM Goal rewriter child with --goal-overlap on Gemma: submit only after cg 3508345 (perf-compose-g1, path (c)) passed; if it failed on the child, submit plan smokehf instead'; eval_wait_seconds='600'; expected_seconds='5200'; goal_overlap='goalvllm4ks'; goal_rewriter_engine='goalvllm4ks:vllm:lm-format-enforcer:3:batch-invariant'; gpu_free_mib='2276'; gpu_hours_expected='1.44'; gpu_hours_limit='2.55'; gpu_peak_mib='95595'; gpu_servers='obs4ks:25:90163:7708:LEAVES_MIN_FREE,rfill4ks:25:90163:7708:LEAVES_MIN_FREE,goalvllm4ks:25:95595:2276:LEAVES_MIN_FREE'; gpu_status='LEAVES_MIN_FREE'; guaranteed_seconds='8880'; kv_cache_gib='25'; limit_over_expected='1.76'; margin_seconds='300'; max_limit_over_expected='2.25'; memory_need_gib='105.22'; memory_spare_gib='5.68'; not_ready=''; partition='ghx4'; role='release-smoke'; servers='obs4ks:observation_query:32tasks:10steps:1500s:forward4096,rfill4ks:random_fill:32tasks:10steps:1600s:forward4096,goalvllm4ks:goal_query:32tasks:10steps:1700s:forward4096:vllm3gib:overlap'; spare_seconds='300'; startup_spike='ELIMINATED'; time='02:33:00'; with_eval_wait_seconds='7130'; worst_case_seconds='8880'}
    'smokehf' = @{all_caps_fit='true'; authorization='AUTHORIZE_GSWE32_SMOKEHF_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_HF_REWRITER_BATCH4_BATCH_REUSED_V9_ENV_ONE_GH200_148MIN_NO_RETRY'; billing_factor='1'; cap_after_wait_fits='true'; condition='the fallback of plan smoke: submit only if cg 3508345 failed on the vLLM Goal rewriter child'; eval_wait_seconds='600'; expected_seconds='5000'; gpu_free_mib='2200'; gpu_hours_expected='1.39'; gpu_hours_limit='2.47'; gpu_peak_mib='95671'; gpu_servers='obs4ks:25:90163:7708:LEAVES_MIN_FREE,rfill4ks:25:90163:7708:LEAVES_MIN_FREE,goalhf4ks:25:95671:2200:LEAVES_MIN_FREE'; gpu_status='LEAVES_MIN_FREE'; guaranteed_seconds='8880'; kv_cache_gib='25'; limit_over_expected='1.78'; margin_seconds='300'; max_limit_over_expected='2.25'; memory_need_gib='105.22'; memory_spare_gib='5.68'; not_ready=''; partition='ghx4'; role='release-smoke'; servers='obs4ks:observation_query:32tasks:10steps:1500s:forward4096,rfill4ks:random_fill:32tasks:10steps:1600s:forward4096,goalhf4ks:goal_query:32tasks:10steps:1700s:batch4:T15360:forward4096'; spare_seconds='0'; startup_spike='ELIMINATED'; time='02:28:00'; with_eval_wait_seconds='7130'; worst_case_seconds='8880'}
    'smokea' = @{all_caps_fit='true'; async_scheduling='obs4ks:on,rfill4ks:on,goalvllm4ks:on'; authorization='AUTHORIZE_GSWE32_SMOKEA_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_153MIN_NO_RETRY'; billing_factor='1'; cap_after_wait_fits='true'; condition='the async-on variant of plan smoke (--async-scheduling on the Gemma 4 profile, CPU-tested only): submit only after g1g r3 3560784 (Gemma async) is verified; if it does not pass, submit plan smoke instead'; eval_wait_seconds='600'; expected_seconds='5200'; goal_overlap='goalvllm4ks'; goal_rewriter_engine='goalvllm4ks:vllm:lm-format-enforcer:3:batch-invariant'; gpu_free_mib='2276'; gpu_hours_expected='1.44'; gpu_hours_limit='2.55'; gpu_peak_mib='95595'; gpu_servers='obs4ks:25:90163:7708:LEAVES_MIN_FREE,rfill4ks:25:90163:7708:LEAVES_MIN_FREE,goalvllm4ks:25:95595:2276:LEAVES_MIN_FREE'; gpu_status='LEAVES_MIN_FREE'; guaranteed_seconds='8880'; kv_cache_gib='25'; limit_over_expected='1.76'; margin_seconds='300'; max_limit_over_expected='2.25'; memory_need_gib='105.22'; memory_spare_gib='5.68'; not_ready=''; partition='ghx4'; role='release-smoke'; servers='obs4ks:observation_query:32tasks:10steps:1500s:forward4096,rfill4ks:random_fill:32tasks:10steps:1600s:forward4096,goalvllm4ks:goal_query:32tasks:10steps:1700s:forward4096:vllm3gib:overlap'; spare_seconds='300'; startup_spike='ELIMINATED'; time='02:33:00'; with_eval_wait_seconds='7130'; worst_case_seconds='8880'}
}
# The user's rule for every DeltaAI job: mail BEGIN, END and FAIL to this address; pinned, not a parameter.
$MailUser = 'cao58@illinois.edu'
$MailType = 'BEGIN,END,FAIL'
$PayloadPath = Join-Path $PSScriptRoot 'dist\payload.json'
if ((Get-FileHash -LiteralPath $PayloadPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $PinnedPayload) { throw 'sealed payload hash mismatch' }
$Bundle = [IO.File]::ReadAllText($PayloadPath) | ConvertFrom-Json
foreach ($Property in $Bundle.files.PSObject.Properties) {
    $Source = $Property.Name
    if ($Source -ceq 'tasks-pool.json') { $Source = 'tasks\tasks-pool.json' }
    $Path = Join-Path $PSScriptRoot $Source
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Property.Value.sha256) { throw ('helper hash mismatch: '+$Property.Name) }
}
. (Join-Path $PSScriptRoot 'transport.ps1')
$Pins = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Bundle.files.'pins.json'.base64)) | ConvertFrom-Json
if ($Pins.commit -cne $ExpectedCommit -or $Pins.package_sha256 -cne 'f151fd29a2d58cb6aa3de27cfdd876ab5e00cd023a284015c84b440d5ea90e4e' -or $Pins.config_sha256 -cne $ExpectedConfig -or $Pins.clean_env.prefix -cne $ExpectedCleanEnv) { throw 'sealed pins differ from this entry point' }
if ($Pins.slurm.mail_user -cne $MailUser -or $Pins.slurm.mail_type -cne $MailType -or $Pins.slurm.job_name_prefix -cne 'gswe32-') { throw 'sealed Slurm pins differ from the mail rule' }
$Receipt = $Pins.release_receipt
if (-not $Receipt -or $Receipt.commit -cne $ExpectedCommit -or $Receipt.status -cne 'GIT_ARCHIVE_TESTS_PASS' -or $Receipt.kind -cne 'check_git_release receipt' -or $Receipt.result_line_sha256 -cne $ExpectedReceipt -or $Receipt.log_sha256 -cne $ExpectedReceiptLog -or [int]$Receipt.tests_run -le 0 -or [int]$Receipt.skipped -lt 0 -or [int]$Receipt.skipped -ge [int]$Receipt.tests_run) { throw 'sealed pins lack the passing release-check receipt of the sealed commit (light login gates: run check_git_release.py --ref <commit> locally, then build.py)' }
$Features = $Pins.runtime_features
if ($Features.stage2 -ne $true -or $Features.goal_async -ne $true -or $Features.decoupled_layout -ne $true -or @($Features.page_tokens_choices) -notcontains 16) { throw 'sealed runtime lacks stage 2, the asynchronous Goal or the decoupled 16-token layout' }
if (-not $Features.session_pool -or [int]$Features.session_pool.max_tasks -lt 32) { throw 'sealed runtime has no session pool of 32 tasks' }
if (-not $Features.context_limits -or @($Features.context_limits.max_model_len_choices) -notcontains 131072 -or @($Features.context_limits.max_tokens_choices) -notcontains 4096 -or $Features.context_limits.dram_arena_flag -ne $true) { throw 'sealed runtime has no 131072 / 4096 context limits with --dram-arena-gib' }
if (-not $Features.admission_flags -or -not $Features.admission_flags.overlap) { throw 'sealed runtime has no admission flags with (f) --admission-overlap' }
if (-not $Features.cuda_graphs -or ((@($Features.cuda_graphs.modes) -join ',') -cne 'off,piecewise,full,full-decode')) { throw 'sealed runtime has no --cuda-graphs' }
if (-not $Features.decode_fastpath -or $Features.decode_fastpath.serve_arg -cne '--decode-fastpath') { throw 'sealed runtime has no --decode-fastpath (43fd9c3 or later)' }
if (-not $Features.language_model_only -or $Features.language_model_only.serve_arg -cne '--language-model-only') { throw 'sealed runtime has no serve.py --language-model-only (4c5b4af or later)' }
if (-not $Features.connector_pool -or $Features.connector_pool.serve_arg -cne '--connector-pool-gib') { throw 'sealed runtime has no --connector-pool-gib (integrate-1014 63d7225 or later)' }
if (-not $Features.gemma4 -or $Features.gemma4.profile -cne 'gemma4-31b-it' -or $Features.gemma4.protocol -cne 'mswea_xml_gemma4_v1' -or $Features.gemma4.attention_backend -cne 'TRITON_ATTN') { throw 'sealed runtime has no servable Gemma 4 profile (integrate-1016 0d37346 or later)' }
if (-not $Features.release_idle_pages -or $Features.release_idle_pages.serve_arg -cne '--release-idle-pages') { throw 'sealed runtime has no --release-idle-pages' }
if ($Pins.config.model.dir -cne '/projects/bcjw/ycao11/models/gemma-4-31b-it' -or $Pins.model -cne $Pins.config.model.dir -or $Pins.config.model.revision -cne $Features.gemma4.revision) { throw 'sealed model differs from the confirmed Gemma 4 download' }
if ($Pins.config.server.decode_fastpath2 -eq $true -and (-not $Features.decode_fastpath2 -or $Features.decode_fastpath2.serve_arg -cne '--decode-fastpath2')) { throw 'the decode_fastpath2 switch is on but the sealed runtime has no --decode-fastpath2' }
if ($Pins.config.server.decode_fastpath3 -ne $true -or -not $Features.decode_fastpath3 -or $Features.decode_fastpath3.serve_arg -cne '--decode-fastpath3' -or $Features.gemma4.decode_fastpath3 -ne $true -or $Pins.config.server.deferred_completion_io -ne $true -or -not $Features.deferred_completion_io -or $Features.deferred_completion_io.serve_arg -cne '--deferred-completion-io' -or $Features.gemma4.deferred_completion_io -ne $true) { throw 'sealed runtime or config lacks --decode-fastpath3 / --deferred-completion-io for the Gemma 4 profile (wave 2: integrate-1021 9ef0d6d or later, both switches on)' }
if (@($Pins.config.plans.goal.servers)[0].goal_rewriter_batch -ne 4 -or $Features.gemma4.goal_rewriter_batch -ne $true -or -not $Features.goal_rewriter_batch -or $Features.goal_rewriter_batch.tokens_arg -cne '--goal-rewriter-batch-tokens' -or [int]$Features.gemma4.goal_rewriter_batch_tokens.'25' -ne 25600 -or [int]@($Pins.config.plans.goal.servers)[0].goal_rewriter_batch_tokens -ne 15360) { throw 'sealed goal plan or runtime differs from the wave-2 decision (--goal-rewriter-batch 4 with the explicit cap T 15360, 2026-10-22: the 2-GiB rule at the 3410948-calibrated estimate; the runtime''s Gemma default 25600 at 25 GiB)' }
if ($Pins.config.policies.full.kv_cache_gib -ne 32 -or $Pins.config.policies.observation_query.kv_cache_gib -ne 25 -or $Pins.config.policies.random_fill.kv_cache_gib -ne 25 -or $Pins.config.policies.goal_query.kv_cache_gib -ne 25 -or $Pins.config.server.max_model_len -ne 131072 -or $Pins.config.server.max_tokens -ne 4096 -or $Pins.config.server.dram_arena_gib -ne 72 -or $Pins.config.server.max_running_sessions -ne 16 -or $Pins.config.server.decode_fastpath -ne $true -or $Pins.config.server.language_model_only -ne $true -or $Pins.config.server.connector_pool_gib -ne 1 -or $Pins.config.server.release_idle_pages -ne $true -or $Pins.config.server.cuda_graphs -cne 'full' -or $Pins.config.stock.language_model_only -ne $true -or $Pins.config.stock.max_num_seqs -ne 16 -or $Pins.config.stock.attention_backend -cne 'TRITON_ATTN') { throw 'sealed server flags differ from the gswe32 decision (KV 25 GiB for observation_query / random_fill / goal_query and 32 GiB for full (2026-10-21), 131072, 4096, arena 72 GiB, K 16, --decode-fastpath, --language-model-only, --connector-pool-gib 1, --release-idle-pages, graphs full; stock 16 seqs on TRITON_ATTN with --language-model-only)' }
if (@($Pins.tasks.instances).Count -ne 32) { throw 'the sealed task list must hold the 32 instances' }
if ($Pins.slurm.mem -cne '113560M') { throw ('sealed sbatch memory '+$Pins.slurm.mem+' differs from --mem=113560M (one GH200''s share, ghx4 DefMemPerGPU)') }
if ($Pins.config.agent.step_limit -ne 120 -or $Pins.config.agent.max_tokens -ne 4096) { throw 'sealed agent settings differ from STEP_LIMIT 120 / MAX_TOKENS 4096' }
if (-not $Plan) { throw ('-Plan required: one of '+(@($Plans.Keys) -join ', ')+' (one ghx4 batch job per configuration: full, goal_query, observation_query, random_fill, stock; Gemma 4 31B IT)') }
if (-not $Plans.Contains($Plan)) { throw ('unknown plan '''+$Plan+''' (one of '+(@($Plans.Keys) -join ', ')+')') }
$Chosen = $Plans[$Plan]
if ($Chosen.startup_spike -ceq 'RISK') { Write-Warning ('plan '+$Plan+': a server starts without --language-model-only (the sealed runtime lacks it); vLLM''s multi-modal warmup needed up to 152.4 GiB in swe16-era starts, above --mem='+$Pins.slurm.mem+' (README Memory)') }
if ($Chosen.gpu_status -ceq 'ESTIMATED_OOM') { Write-Warning ('plan '+$Plan+': the GPU-memory estimate at --kv-cache-gib '+$Chosen.kv_cache_gib+' exceeds the GH200 by '+(-[int]$Chosen.gpu_free_mib)+' MiB (README GPU memory): submit only after full''s measured peak (summary-full.json resources.gpu_memory_used_mib_max) shows room, or after the proposed runtime change') }
elseif ($Chosen.gpu_status -ceq 'BELOW_MIN_FREE') { Write-Warning ('plan '+$Plan+': the GPU-memory estimate at --kv-cache-gib '+$Chosen.kv_cache_gib+' leaves '+$Chosen.gpu_free_mib+' MiB, under the 2-GiB rule (README GPU memory: full is pinned at 32 GiB by the runtime, job 3410948 measured it TIGHT at 97,121 MiB)') }
$Sealed = $Pins.config.plans.$Plan
if (-not $Sealed -or $Sealed.authorization -cne $Chosen.authorization -or $Sealed.time -cne $Chosen.time -or $Sealed.partition -cne $Chosen.partition) { throw 'sealed plan differs from this entry point' }
$PartitionMax = $Pins.slurm.partition_limits.($Chosen.partition).max_time
if (-not $PartitionMax -or ([TimeSpan]::Parse($Chosen.time) -gt [TimeSpan]::Parse($PartitionMax))) { throw ('plan '+$Plan+' asks --time='+$Chosen.time+', above the '+$Chosen.partition+' maximum '+$PartitionMax) }
if ([int]$Chosen.worst_case_seconds -gt [TimeSpan]::Parse($Chosen.time).TotalSeconds -or [int]$Chosen.with_eval_wait_seconds -gt [TimeSpan]::Parse($Chosen.time).TotalSeconds) { throw ('plan '+$Plan+' does not fit its Slurm time') }
if (-not $RewriterModel) { $RewriterModel = $Pins.rewriter_default }
$RewriterModel = $RewriterModel.TrimEnd('/')
if ($RewriterModel -cnotmatch '^/[A-Za-z0-9._/-]+$') { throw 'RewriterModel must be an absolute DeltaAI path of [A-Za-z0-9._/-]' }
if ($RunLabel -cnotmatch '^r[0-9]{1,2}$') { throw 'RunLabel must match r[0-9]{1,2} (for example r1, r2)' }
if ($AfterJob -and $AfterJob -cnotmatch '^[0-9]{1,12}$') { throw 'AfterJob must be a Slurm job id' }
if ($AfterJob -and $Chosen.partition -ceq 'ghx4-interactive') { throw '-AfterJob is for ghx4 batch plans: a ghx4-interactive plan (one submitted job per user, 2x billing) goes first in a tbench chain, while no other tbench job of yours is queued' }
$AfterValue = $null
if ($AfterJob) { $AfterValue = $AfterJob }
$Nonce = [Guid]::NewGuid().ToString('N')
$BootstrapCode = $Bundle.files.'bootstrap.py'.base64
$PythonCode = 'import base64;exec(compile(base64.b64decode("'+$BootstrapCode+'"),"sealed-bootstrap","exec"))'
$RemoteCommand = '/work/nvme/bcjw/ycao11/vllm-arm/bin/python -I -S -B -c ' + (ConvertTo-BashLiteral $PythonCode) + ' ' + $Nonce
$Options = [ordered]@{plan=$Plan; authorization=$Chosen.authorization; mail_user=$MailUser; rewriter_model=$RewriterModel; run_label=$RunLabel; after_job=$AfterValue}
$Bundle | Add-Member -MemberType NoteProperty -Name options -Value $Options
$Bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Bundle | ConvertTo-Json -Depth 10 -Compress))
$TaskCount = @($Pins.tasks.instances).Count
if ($Pins.tasks.provisional) { $TaskState = 'tasks=PROVISIONAL('+$TaskCount+')' } else { $TaskState = 'tasks=CONFIRMED('+$TaskCount+' = the first 32 SWE-bench Verified instances, as swe16 / swe32)' }
if ($Pins.config.server.decode_fastpath2 -eq $true) { $Fast2 = 'on' } else { $Fast2 = 'off' }
if ($Features.goal_text) { $GoalText = 'goal_text=' + (@($Features.goal_text.files) -join ',') } else { $GoalText = 'goal_text=NOT_AVAILABLE' }
$Provisional = $TaskState+' model=gemma4-31b-it@'+$Pins.config.model.revision.Substring(0,7)+' decode_fastpath=on decode_fastpath2='+$Fast2+' decode_fastpath3=on deferred_completion_io=on language_model_only=on release_idle_pages=on connector_pool_gib='+$Pins.config.server.connector_pool_gib+' '+$GoalText+' still_provisional='+((@($Pins.config.provisional.PSObject.Properties | ForEach-Object { $_.Name }) | Sort-Object) -join ',')+' statements_sealed='+$StatementsSealed
$Ready = 'ready=yes'
if ($Chosen.not_ready) { $Ready = 'ready=NOT_READY'; Write-Warning ('plan '+$Plan+' is NOT_READY: '+$Chosen.not_ready) }
$Summary = ' plan='+$Plan+' '+$Ready+' billing_factor='+$Chosen.billing_factor+' servers='+$Chosen.servers+' partition='+$Chosen.partition+' time='+$Chosen.time+' expected_seconds='+$Chosen.expected_seconds+' gpu_hours_limit='+$Chosen.gpu_hours_limit+' gpu_hours_expected='+$Chosen.gpu_hours_expected+' worst_case_seconds='+$Chosen.worst_case_seconds+' with_eval_wait_seconds='+$Chosen.with_eval_wait_seconds+' cap_after_wait_fits='+$Chosen.cap_after_wait_fits+' eval_wait_seconds='+$Chosen.eval_wait_seconds+' margin_seconds='+$Chosen.margin_seconds+' spare_seconds='+$Chosen.spare_seconds+' kv_cache_gib='+$Chosen.kv_cache_gib+' gpu_peak_mib='+$Chosen.gpu_peak_mib+' gpu_free_mib='+$Chosen.gpu_free_mib+' gpu_status='+$Chosen.gpu_status+' mem='+$Pins.slurm.mem+' dram_arena_gib='+$Pins.config.server.dram_arena_gib+' memory_need_gib='+$Chosen.memory_need_gib+' memory_spare_gib='+$Chosen.memory_spare_gib+' startup_spike='+$Chosen.startup_spike+' job_name='+$Pins.slurm.job_name_prefix+$Plan+' run_label='+$RunLabel+' after_job='+$AfterJob+' mail_user='+$MailUser+' mail_type='+$MailType+' '+$Provisional
# The release smoke (2026-11-03): the plan's role, the vLLM Goal rewriter, --goal-overlap and every server's GPU estimate
# (only plans that name them: every other plan's line is unchanged); a plan's condition is a warning, it blocks nothing.
if ($Chosen.role) { $Summary += ' role='+$Chosen.role+' limit_over_expected='+$Chosen.limit_over_expected+' max_limit_over_expected='+$Chosen.max_limit_over_expected }
if ($Chosen.goal_rewriter_engine) { $Summary += ' goal_rewriter_engine='+$Chosen.goal_rewriter_engine }
if ($Chosen.goal_overlap) { $Summary += ' goal_overlap='+$Chosen.goal_overlap }
if ($Chosen.gpu_servers) { $Summary += ' gpu_servers='+$Chosen.gpu_servers }
# The 2026-11-09 re-pin (main ba4d1ba, decision D-C): a plan that names --async-scheduling (true or false, for all its
# runtime servers; plan smokea) needs, when on, the sealed switch of the Gemma 4 profile and (decision D-G) the sealed
# async fixes (runtime_features.async_fix: async-fix a0798a8, async-fix2 4ed1013 and, since 2026-11-14,
# async-maxtok-1112 13ee618's resolve), else it is NOT_SUBMITTABLE; its sealed value and the plan table agree and its
# line names it; every other plan's checks and line are unchanged.
$AsyncValue = $Pins.config.plans.$Plan.async_scheduling
if ($null -ne $AsyncValue) {
    if ($AsyncValue -eq $true -and (-not $Features.async_scheduling -or $Features.async_scheduling.serve_arg -cne '--async-scheduling' -or $Features.gemma4_goal.async_scheduling -ne $true)) { throw ('plan '+$Plan+' asks for --async-scheduling; the sealed runtime''s Gemma 4 profile has none') }
    if ($AsyncValue -eq $true -and (-not $Features.async_fix -or -not $Features.async_fix.resolve)) { throw ('NOT_SUBMITTABLE: plan '+$Plan+' asks for --async-scheduling but the sealed runtime lacks the async fixes (a0798a8, 4ed1013, 13ee618)') }
    if ($AsyncValue -eq $true) { $AsyncState = 'on' } else { $AsyncState = 'off' }
    $AsyncSeen = (@(@($Sealed.servers) | Where-Object { $_.policy -cne 'stock' } | ForEach-Object { $_.label+':'+$AsyncState }) -join ',')
    if ($AsyncSeen -cne [string]$Chosen.async_scheduling) { throw ('sealed --async-scheduling of plan '+$Plan+' ('+$AsyncSeen+') differs from this entry point ('+$Chosen.async_scheduling+')') }
    $Summary += ' async_scheduling='+$Chosen.async_scheduling
} elseif ($Chosen.async_scheduling) { throw ('plan '+$Plan+': the entry point names --async-scheduling the sealed plan lacks') }
# 2026-11-12 (config.tbench_verify; perf-swe32's push of 2026-11-11): every plan's job pushes its two archives to tbench
# and starts the verification there at its end (best effort, bounded); the line names it (the only new field: recorded
# lines compare with it masked).
$TbenchPush = 'off'
if ($Pins.config.tbench_verify -and $Pins.config.tbench_verify.push) {
    if ($Pins.config.tbench_verify.root -cne '/home/ubuntu/gswe32-verify' -or [int]$Pins.config.tbench_verify.total_seconds -gt 300) { throw 'sealed tbench_verify settings differ from the evidence-push decision' }
    $TbenchPush = $Pins.config.tbench_verify.root+':'+$Pins.config.tbench_verify.total_seconds+'s'
}
$Summary += ' tbench_push='+$TbenchPush
if ($Chosen.condition) { Write-Warning ('plan '+$Plan+': '+$Chosen.condition) }
if ($SelfTest) {
    $Tokens=$null; $Errors=$null
    [void][Management.Automation.Language.Parser]::ParseFile($PSCommandPath,[ref]$Tokens,[ref]$Errors)
    if ($Errors.Count) { throw 'PowerShell syntax invalid' }
    $Info = New-OneConnectionStartInfo -Executable 'ssh' -RemoteCommand $RemoteCommand
    if ($Info.CreateNoWindow -or $Info.RedirectStandardOutput -or $Info.RedirectStandardError -or -not $Info.RedirectStandardInput) { throw 'visible SSH console contract broken' }
    if (-not $Info.Arguments.Contains('ConnectionAttempts=1')) { throw 'single connection contract broken' }
    $RoundTrip = [Text.Encoding]::UTF8.GetString($Bytes) | ConvertFrom-Json
    if ($RoundTrip.files.'bootstrap.py'.base64 -cne $BootstrapCode -or $RoundTrip.options.plan -cne $Plan -or $RoundTrip.options.run_label -cne $RunLabel -or $RoundTrip.options.rewriter_model -cne $RewriterModel -or $RoundTrip.options.mail_user -cne $MailUser -or $RoundTrip.options.authorization -cne $Chosen.authorization) { throw 'sealed JSON binary transport failed' }
    foreach ($Name in 'job.py','controller.py','swe_checks.py','smoke_checks.py','measure.py','resources.py','tbench_link.py','stock.py','goal_text.py','tbench_known_hosts') { if (-not $RoundTrip.files.$Name.sha256) { throw ('sealed file missing: '+$Name) } }
    if ($Pins.slurm.account -cne 'bcjw-dtai-gh' -or $Pins.slurm.partition_limits.'ghx4-interactive'.max_submitted_jobs_per_user -ne 1) { throw 'Slurm pin changed' }
    if ($Pins.clean_env.snapshot_sha256 -cne 'cbfab91650bfbeffb149a2df8ead3787a07b91a2cf332edd7efd0b5b0a07e5f0' -or $Pins.clean_env.release_manifest_sha256 -cne 'b23b5f20e7c85f8f8e3ab93fe51d8519dfd465aa50fecf0cda27c95d18e9af93') { throw 'reused v9 environment pin changed' }
    if ($Pins.wheel_sha256 -cne '700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5' -or -not $Pins.wheel_url.EndsWith('/vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl')) { throw 'official vLLM wheel pin changed' }
    if ($TaskCount -lt [int]$Pins.config.server.max_running_sessions -or $TaskCount -gt 64) { throw ('the task list holds '+$TaskCount+' instances: at least K and at most 64') }
    if (([Text.Encoding]::UTF8.GetByteCount($StagePrefix+$Nonce+'/tmp')+1+36) -gt 99) { throw 'stage path exceeds IPC budget' }
    Write-Output ('LOCAL_SELFTEST_PASS connections=0 install=0 submit=0 payload_sha256='+$PinnedPayload+' commit='+$ExpectedCommit+$Summary)
    return
}
if (-not $Authorize) {
    if ($Authorization) { throw 'explicit -Authorize switch also required' }
    Write-Output ('LOCAL_ONLY_PASS connections=0 install=0 submit=0 plan='+$Plan+' '+$Provisional)
    return
}
if ($Authorization -cne $Chosen.authorization) { throw ('exact authorization phrase of plan '+$Plan+' required') }
# A plan marked not_ready (goal until its wave-2 rebuild) is neither dry-run nor submitted; nothing is sent.
if ($Chosen.not_ready) { throw ('plan '+$Plan+' is NOT_READY: '+$Chosen.not_ready+'; nothing was sent') }
if ($DryRun) {
    Write-Output ('DRY_RUN connections=0 stage='+$StagePrefix+$Nonce+' payload_bytes='+$Bytes.Length+' commit='+$ExpectedCommit+$Summary)
    return
}
if ($StatementsSealed -cne 'yes') { throw 'the task statements are not sealed yet (tasks\tasks-pool.json): run tbench bin/build_tasks.sh, then build.py --tasks <file> (README "Task statements"); nothing was submitted' }
Write-Output ('PLANNED_STAGE='+$StagePrefix+$Nonce+' PLAN='+$Plan)
Write-Output ('One visible SSH/Duo connection: reuse the v9 clean env read-only, private release and pre-claim gates (light login gates, a few minutes: no runtime test suite on the login node (the local release-check receipt of '+$ExpectedCommit.Substring(0,7)+' is pinned and compared with the cloned HEAD), the cheap probes, a serve.py dry run per server, the tbench preflight), then exactly one '+$Chosen.partition+' GH200 sbatch of plan '+$Plan+' (--time='+$Chosen.time+', mail on BEGIN, END, FAIL to '+$MailUser+'). Any unknown outcome must not be retried.')
if ($Chosen.partition -ceq 'ghx4-interactive') { Write-Output 'ghx4-interactive allows 1 submitted job per user and heads a tbench chain: the run stops before its claim (exit 86, nothing burned) while squeue lists one of yours there or any tbench job of yours (gswe32 / swe32 / swe10 / swe16 / tb16 / gem0) is queued or running.' }
if ($Chosen.partition -ceq 'ghx4') { Write-Output ('ghx4 batch (1 GPU at --mem=113560M: one GPU-hour per hour, at most '+$Chosen.gpu_hours_limit+' for this plan): the job waits in the queue by fairshare priority (possibly hours; the BEGIN mail says when it starts). The queue gate stops before the claim (exit 86) while a gswe32 / swe32 / swe10 / swe16 / tb16 / gem0 job of yours is queued or running that this job would not wait for: name the last job of the queued tbench chain with -AfterJob (afterany).') }
$Code = Invoke-OneConnectionTransport -Executable (Get-Command ssh -CommandType Application | Select-Object -First 1).Source -RemoteCommand $RemoteCommand -PayloadBytes $Bytes
if ($Code -eq 86) { throw 'Pre-claim setup failed: remote confirmed NO_SUBMISSION_CLAIM. No sbatch was reached. Read the printed QUALIFY_ERROR/ERROR line, fix it, then start a separately authorized run with a new -RunLabel; no automatic retry.' }
if ($Code -ne 0) { throw ('Setup failed or submission outcome unknown; consult the printed FAILURE_STATUS/CLAIM_STATE. Without confirmed NO_SUBMISSION_CLAIM, keep the stage and claim and reconcile read-only before any replacement. No automatic retry. SSH exit='+$Code) }
Write-Output 'SUBMITTED: the last JSON line above holds stage, job_id and run_id. After the END mail: the README check block, then postrun\fetch_run.ps1 (both sides, verify, join, tables).'
