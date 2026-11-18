[CmdletBinding()]
param(
    [switch]$Authorize,
    [string]$Authorization = '',
    # One sealed plan per submission (README "Plans"): full, obs, rfill, goal, stock - one ghx4 batch job each (all 89
    # Terminal-Bench 2.0 tasks, 16 concurrent trials, --mem=113560M); dry / dryi / dryo and goaldry / goaldryi on the
    # 16-task dry subset; the Gemma plans (g*). -AfterJob <J> queues after job J (--dependency=afterany:<J>); the queue
    # gate lets J and the jobs J itself waits for through (the submission chain: name the job submitted last).
    [string]$Plan = '',
    [string]$RewriterModel = '',
    [string]$RunLabel = 'r1',
    [string]$AfterJob = '',
    [switch]$SelfTest,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$PinnedPayload = '4203f013c52613edb64d365b6e014be6f70a0605a6c07fd2dc5262807ecc5bb5'
$ExpectedCommit = '4ebff978661f4c5e2e06da0ee68a5217f92dfa82'
$ExpectedConfig = '06927e939499a8ac0a8964771da2ff2565100715b27f8ef51432a8f5242d9499'
# The runtime pin (build.py, controller.pin_status): PINNED only for a commit whose chat protocol the harness was
# adapted to; a PLACEHOLDER never submits (SelfTest and DryRun still run).
$PinStatus = 'PINNED'
$PinReason = ''
# The light login gates (2026-10-18): the login node runs no test suite; the local check_git_release receipt of the
# sealed commit is pinned (pins.json release_receipt: commit, result line and log SHA-256). A PINNED pin needs it.
$ReceiptCommit = '4ebff978661f4c5e2e06da0ee68a5217f92dfa82'
$ReceiptResult = '308a6e79fc938fd2e2b62a87ebd7b7760703df31befac715fdb966459b30ea8f'
$ReceiptLog = '7bb8bcbd486c873fcab85964d9f59f4213c503fe4c8a2a785bbc424b54b82a6c'
$ReceiptTests = 2059
$ExpectedTasks = 89
$ExpectedCleanEnv = '/u/ycao11/spq9-277fa5f1390947e08caefecc8e928a3d/clean-env'
$StagePrefix = '/u/ycao11/tb16-'
# The pinned plans (build.py writes them from pins.json config): phrase, partition, Slurm time, time fit, servers.
$Plans = [ordered]@{
    'full' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_FULL_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='13900'; gpu_hours_billed_expected='3.86'; gpu_hours_billed_limit='5.03'; gpu_hours_expected='3.86'; gpu_hours_limit='5.03'; guaranteed_seconds='18300'; margin_seconds='300'; memory_need_gib='99.87'; memory_spare_gib='11.03'; partition='ghx4'; servers='full:full:89tasks:16380s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='89'; time='05:02:00'; urgent=''; with_eval_wait_seconds='14520'; worst_case_seconds='18120'}
    'obs' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_OBS_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='13900'; gpu_hours_billed_expected='3.86'; gpu_hours_billed_limit='5.03'; gpu_hours_expected='3.86'; gpu_hours_limit='5.03'; guaranteed_seconds='18300'; margin_seconds='300'; memory_need_gib='99.87'; memory_spare_gib='11.03'; partition='ghx4'; servers='obs:observation_query:89tasks:16380s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='89'; time='05:02:00'; urgent=''; with_eval_wait_seconds='14520'; worst_case_seconds='18120'}
    'rfill' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_RFILL_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='13900'; gpu_hours_billed_expected='3.86'; gpu_hours_billed_limit='5.03'; gpu_hours_expected='3.86'; gpu_hours_limit='5.03'; guaranteed_seconds='18300'; margin_seconds='300'; memory_need_gib='99.87'; memory_spare_gib='11.03'; partition='ghx4'; servers='rfill:random_fill:89tasks:16380s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='89'; time='05:02:00'; urgent=''; with_eval_wait_seconds='14520'; worst_case_seconds='18120'}
    'goal' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GOAL_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='13960'; gpu_hours_billed_expected='3.88'; gpu_hours_billed_limit='5.03'; gpu_hours_expected='3.88'; gpu_hours_limit='5.03'; guaranteed_seconds='18300'; margin_seconds='300'; memory_need_gib='99.87'; memory_spare_gib='11.03'; partition='ghx4'; servers='goal:goal_query:89tasks:16380s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='89'; time='05:02:00'; urgent=''; with_eval_wait_seconds='14520'; worst_case_seconds='18120'}
    'stock' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_STOCK_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_293MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='13500'; gpu_hours_billed_expected='3.75'; gpu_hours_billed_limit='4.88'; gpu_hours_expected='3.75'; gpu_hours_limit='4.88'; guaranteed_seconds='17760'; margin_seconds='300'; memory_need_gib='32.00'; memory_spare_gib='78.90'; partition='ghx4'; servers='stock:stock:89tasks:16320s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='89'; time='04:53:00'; urgent=''; with_eval_wait_seconds='14040'; worst_case_seconds='17580'}
    'dry' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_DRY_16TASKS_2SERVERS_BATCH_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='3960'; gpu_hours_billed_expected='1.10'; gpu_hours_billed_limit='1.43'; gpu_hours_expected='1.10'; gpu_hours_limit='1.43'; guaranteed_seconds='5340'; margin_seconds='300'; memory_need_gib='99.87'; memory_spare_gib='11.03'; partition='ghx4'; servers='dryfull:full:16tasks:1260s,dryobs:observation_query:16tasks:1260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='16'; time='01:26:00'; urgent=''; with_eval_wait_seconds='4820'; worst_case_seconds='5160'}
    'dryi' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_DRYI_16TASKS_2SERVERS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY'; billing_factor='2'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='3960'; gpu_hours_billed_expected='2.20'; gpu_hours_billed_limit='2.87'; gpu_hours_expected='1.10'; gpu_hours_limit='1.43'; guaranteed_seconds='5340'; margin_seconds='300'; memory_need_gib='99.87'; memory_spare_gib='11.03'; partition='ghx4-interactive'; servers='dryfull:full:16tasks:1260s,dryobs:observation_query:16tasks:1260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='16'; time='01:26:00'; urgent='ghx4 batch waits are about a day (coordinator, 2026-10-17)'; with_eval_wait_seconds='4820'; worst_case_seconds='5160'}
    'dryo' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_DRYO_16TASKS_1SERVER_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY'; billing_factor='2'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='2400'; gpu_hours_billed_expected='1.33'; gpu_hours_billed_limit='1.73'; gpu_hours_expected='0.67'; gpu_hours_limit='0.87'; guaranteed_seconds='3180'; margin_seconds='300'; memory_need_gib='99.87'; memory_spare_gib='11.03'; partition='ghx4-interactive'; servers='dryobs:observation_query:16tasks:1260s'; spare_seconds='120'; startup_spike='ELIMINATED'; tasks='16'; time='00:52:00'; urgent='ghx4 batch waits are about a day (coordinator, 2026-10-18)'; with_eval_wait_seconds='2920'; worst_case_seconds='3000'}
    'goaldry' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GOALDRY_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='2460'; gpu_hours_billed_expected='0.68'; gpu_hours_billed_limit='0.87'; gpu_hours_expected='0.68'; gpu_hours_limit='0.87'; guaranteed_seconds='3180'; margin_seconds='300'; memory_need_gib='99.87'; memory_spare_gib='11.03'; partition='ghx4'; servers='drygoal:goal_query:16tasks:1260s'; spare_seconds='120'; startup_spike='ELIMINATED'; tasks='16'; time='00:52:00'; urgent=''; with_eval_wait_seconds='2920'; worst_case_seconds='3000'}
    'goaldryi' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GOALDRYI_16TASKS_1SERVER_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY'; billing_factor='2'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='2460'; gpu_hours_billed_expected='1.37'; gpu_hours_billed_limit='1.73'; gpu_hours_expected='0.68'; gpu_hours_limit='0.87'; guaranteed_seconds='3180'; margin_seconds='300'; memory_need_gib='99.87'; memory_spare_gib='11.03'; partition='ghx4-interactive'; servers='drygoal:goal_query:16tasks:1260s'; spare_seconds='120'; startup_spike='ELIMINATED'; tasks='16'; time='00:52:00'; urgent='paper deadline 2026-10-30: the goal_query GPU validation before the 89-task goal run of Table 1 (the user, 2026-10-21)'; with_eval_wait_seconds='2920'; worst_case_seconds='3000'}
    'gfull' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GFULL_GEMMA4_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='14020'; gpu_hours_billed_expected='3.89'; gpu_hours_billed_limit='5.03'; gpu_hours_expected='3.89'; gpu_hours_limit='5.03'; guaranteed_seconds='18180'; margin_seconds='300'; memory_need_gib='97.00'; memory_spare_gib='13.90'; model='gemma4-31b-it'; partition='ghx4'; servers='gfull:full:89tasks:16260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='89'; time='05:02:00'; urgent=''; with_eval_wait_seconds='14520'; worst_case_seconds='18120'}
    'gobs' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GOBS_GEMMA4_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='14020'; gpu_hours_billed_expected='3.89'; gpu_hours_billed_limit='5.03'; gpu_hours_expected='3.89'; gpu_hours_limit='5.03'; guaranteed_seconds='18180'; margin_seconds='300'; memory_need_gib='97.00'; memory_spare_gib='13.90'; model='gemma4-31b-it'; partition='ghx4'; servers='gobs:observation_query:89tasks:16260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='89'; time='05:02:00'; urgent=''; with_eval_wait_seconds='14520'; worst_case_seconds='18120'}
    'grfill' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GRFILL_GEMMA4_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='14020'; gpu_hours_billed_expected='3.89'; gpu_hours_billed_limit='5.03'; gpu_hours_expected='3.89'; gpu_hours_limit='5.03'; guaranteed_seconds='18180'; margin_seconds='300'; memory_need_gib='97.00'; memory_spare_gib='13.90'; model='gemma4-31b-it'; partition='ghx4'; servers='grfill:random_fill:89tasks:16260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='89'; time='05:02:00'; urgent=''; with_eval_wait_seconds='14520'; worst_case_seconds='18120'}
    'gstock' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GSTOCK_GEMMA4_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_293MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='13620'; gpu_hours_billed_expected='3.78'; gpu_hours_billed_limit='4.88'; gpu_hours_expected='3.78'; gpu_hours_limit='4.88'; guaranteed_seconds='17640'; margin_seconds='300'; memory_need_gib='32.00'; memory_spare_gib='78.90'; model='gemma4-31b-it'; partition='ghx4'; servers='gstock:stock:89tasks:16200s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='89'; time='04:53:00'; urgent=''; with_eval_wait_seconds='14040'; worst_case_seconds='17580'}
    'gdry' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GDRY_GEMMA4_16TASKS_2SERVERS_BATCH_REUSED_V9_ENV_ONE_GH200_90MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='4200'; gpu_hours_billed_expected='1.17'; gpu_hours_billed_limit='1.50'; gpu_hours_expected='1.17'; gpu_hours_limit='1.50'; guaranteed_seconds='5460'; margin_seconds='300'; memory_need_gib='97.00'; memory_spare_gib='13.90'; model='gemma4-31b-it'; partition='ghx4'; servers='gdryfull:full:16tasks:1260s,gdryobs:observation_query:16tasks:1260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='16'; time='01:30:00'; urgent=''; with_eval_wait_seconds='4940'; worst_case_seconds='5400'}
    'ggoal' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GGOAL_GEMMA4_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='14080'; gpu_hours_billed_expected='3.91'; gpu_hours_billed_limit='5.03'; gpu_hours_expected='3.91'; gpu_hours_limit='5.03'; guaranteed_seconds='18180'; margin_seconds='300'; memory_need_gib='97.00'; memory_spare_gib='13.90'; model='gemma4-31b-it'; partition='ghx4'; servers='ggoal:goal_query:89tasks:16260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='89'; time='05:02:00'; urgent=''; with_eval_wait_seconds='14520'; worst_case_seconds='18120'}
    'ggoaldry' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GGOALDRY_GEMMA4_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='2580'; gpu_hours_billed_expected='0.72'; gpu_hours_billed_limit='0.87'; gpu_hours_expected='0.72'; gpu_hours_limit='0.87'; guaranteed_seconds='3180'; margin_seconds='300'; memory_need_gib='97.00'; memory_spare_gib='13.90'; model='gemma4-31b-it'; partition='ghx4'; servers='gdrygoal:goal_query:16tasks:1260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='16'; time='00:52:00'; urgent=''; with_eval_wait_seconds='2920'; worst_case_seconds='3120'}
    'gobsdry' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GOBSDRY_GEMMA4_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='2520'; gpu_hours_billed_expected='0.70'; gpu_hours_billed_limit='0.87'; gpu_hours_expected='0.70'; gpu_hours_limit='0.87'; guaranteed_seconds='3180'; margin_seconds='300'; memory_need_gib='97.00'; memory_spare_gib='13.90'; model='gemma4-31b-it'; partition='ghx4'; servers='gdryobs:observation_query:16tasks:1260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='16'; time='00:52:00'; urgent=''; with_eval_wait_seconds='2920'; worst_case_seconds='3120'}
    'grfilldry' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GRFILLDRY_GEMMA4_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='2520'; gpu_hours_billed_expected='0.70'; gpu_hours_billed_limit='0.87'; gpu_hours_expected='0.70'; gpu_hours_limit='0.87'; guaranteed_seconds='3180'; margin_seconds='300'; memory_need_gib='97.00'; memory_spare_gib='13.90'; model='gemma4-31b-it'; partition='ghx4'; servers='gdryrfill:random_fill:16tasks:1260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='16'; time='00:52:00'; urgent=''; with_eval_wait_seconds='2920'; worst_case_seconds='3120'}
    'ctx25' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_CTX25_16TASKS_2SERVERS_262K_KV25_BATCH_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; dram_arena_gib='70'; eval_wait_seconds='600'; expected_seconds='4020'; gpu_hours_billed_expected='1.12'; gpu_hours_billed_limit='1.43'; gpu_hours_expected='1.12'; gpu_hours_limit='1.43'; guaranteed_seconds='5340'; kv_cache_gib='25'; margin_seconds='300'; max_input_tokens='262144'; max_model_len='262144'; memory_need_gib='105.87'; memory_spare_gib='5.03'; partition='ghx4'; servers='drygoal262:goal_query:16tasks:1260s,dryobs262:observation_query:16tasks:1260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='16'; time='01:26:00'; urgent=''; with_eval_wait_seconds='4820'; worst_case_seconds='5160'}
    'ctx24' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_CTX24_16TASKS_2SERVERS_262K_KV24_BATCH_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; dram_arena_gib='70'; eval_wait_seconds='600'; expected_seconds='4020'; gpu_hours_billed_expected='1.12'; gpu_hours_billed_limit='1.43'; gpu_hours_expected='1.12'; gpu_hours_limit='1.43'; guaranteed_seconds='5340'; kv_cache_gib='24'; margin_seconds='300'; max_input_tokens='262144'; max_model_len='262144'; memory_need_gib='105.87'; memory_spare_gib='5.03'; partition='ghx4'; servers='drygoal262:goal_query:16tasks:1260s,dryobs262:observation_query:16tasks:1260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='16'; time='01:26:00'; urgent=''; with_eval_wait_seconds='4820'; worst_case_seconds='5160'}
    'strip25' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_STRIP25_16TASKS_3SERVERS_262K_KV25_THINKING_STRIP_BATCH_REUSED_V9_ENV_ONE_GH200_119MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; dram_arena_gib='70'; eval_wait_seconds='600'; expected_seconds='5400'; gpu_hours_billed_expected='1.50'; gpu_hours_billed_limit='1.98'; gpu_hours_expected='1.50'; gpu_hours_limit='1.98'; guaranteed_seconds='7320'; kv_cache_gib='25'; margin_seconds='300'; max_input_tokens='262144'; max_model_len='262144'; memory_need_gib='105.87'; memory_spare_gib='5.03'; partition='ghx4'; servers='stripobs262:observation_query:16tasks:1260s,stripgoal262:goal_query:16tasks:1260s,stockthink262:stock:16tasks:1260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='16'; time='01:59:00'; urgent=''; with_eval_wait_seconds='6540'; worst_case_seconds='7140'}
    'smokeq' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_SMOKEQ_32TASKS_3SERVERS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_173MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; dram_arena_gib='70'; eval_wait_seconds='600'; expected_seconds='6900'; gpu_hours_billed_expected='1.92'; gpu_hours_billed_limit='2.88'; gpu_hours_expected='1.92'; gpu_hours_limit='2.88'; guaranteed_seconds='10560'; kv_cache_gib='25'; margin_seconds='300'; max_input_tokens='262144'; max_model_len='262144'; memory_need_gib='105.87'; memory_spare_gib='5.03'; partition='ghx4'; servers='sobs:observation_query:32tasks:2280s,srfill:random_fill:32tasks:2280s,sgoal:goal_query:32tasks:2280s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='32'; time='02:53:00'; urgent=''; with_eval_wait_seconds='9840'; worst_case_seconds='10380'}
    'smokeqs' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_SMOKEQS_32TASKS_3SERVERS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_NO_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_173MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; dram_arena_gib='70'; eval_wait_seconds='600'; expected_seconds='6900'; gpu_hours_billed_expected='1.92'; gpu_hours_billed_limit='2.88'; gpu_hours_expected='1.92'; gpu_hours_limit='2.88'; guaranteed_seconds='10560'; kv_cache_gib='25'; margin_seconds='300'; max_input_tokens='262144'; max_model_len='262144'; memory_need_gib='105.87'; memory_spare_gib='5.03'; partition='ghx4'; servers='sobs:observation_query:32tasks:2280s,srfill:random_fill:32tasks:2280s,sgoal:goal_query:32tasks:2280s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='32'; time='02:53:00'; urgent=''; with_eval_wait_seconds='9840'; worst_case_seconds='10380'}
    'smokeg' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_SMOKEG_GEMMA4_32TASKS_3SERVERS_KV25_VLLM_REWRITER_GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_179MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='7000'; gpu_hours_billed_expected='1.94'; gpu_hours_billed_limit='2.98'; gpu_hours_expected='1.94'; gpu_hours_limit='2.98'; guaranteed_seconds='10800'; margin_seconds='300'; memory_need_gib='97.00'; memory_spare_gib='13.90'; model='gemma4-31b-it'; partition='ghx4'; servers='gsobs:observation_query:32tasks:2280s,gsrfill:random_fill:32tasks:2280s,gsgoal:goal_query:32tasks:2280s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='32'; time='02:59:00'; urgent=''; with_eval_wait_seconds='10080'; worst_case_seconds='10740'}
    'smokegh' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_SMOKEGH_GEMMA4_32TASKS_3SERVERS_KV25_HF_REWRITER_BATCH4_BATCH_REUSED_V9_ENV_ONE_GH200_179MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; eval_wait_seconds='600'; expected_seconds='7000'; gpu_hours_billed_expected='1.94'; gpu_hours_billed_limit='2.98'; gpu_hours_expected='1.94'; gpu_hours_limit='2.98'; guaranteed_seconds='10800'; margin_seconds='300'; memory_need_gib='97.00'; memory_spare_gib='13.90'; model='gemma4-31b-it'; partition='ghx4'; servers='gsobs:observation_query:32tasks:2280s,gsrfill:random_fill:32tasks:2280s,gsgoalhf:goal_query:32tasks:2280s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='32'; time='02:59:00'; urgent=''; with_eval_wait_seconds='10080'; worst_case_seconds='10740'}
    'gctx25' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_GCTX25_GEMMA4_16TASKS_4SERVERS_262K_KV25_BATCH_REUSED_V9_ENV_ONE_GH200_163MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; dram_arena_gib='72'; eval_wait_seconds='600'; expected_seconds='7440'; gpu_hours_billed_expected='2.07'; gpu_hours_billed_limit='2.72'; gpu_hours_expected='2.07'; gpu_hours_limit='2.72'; guaranteed_seconds='9840'; kv_cache_gib='25'; margin_seconds='300'; max_input_tokens='262144'; max_model_len='262144'; max_num_batched_tokens='4096'; max_running_sessions='20'; memory_need_gib='105.22'; memory_spare_gib='5.68'; model='gemma4-31b-it'; partition='ghx4'; servers='gobs262:observation_query:16tasks:1260s,ggoal262:goal_query:16tasks:1260s,grfill262:random_fill:16tasks:1260s,gstock262:stock:16tasks:1260s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='16'; time='02:43:00'; urgent=''; with_eval_wait_seconds='8800'; worst_case_seconds='9780'}
    'smokeqt' = @{all_caps_fit='true'; authorization='AUTHORIZE_TB16_SMOKEQT_32TASKS_3SERVERS_262K_KV25_THINKING_STRIP_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_173MIN_NO_RETRY'; billing_factor='1'; blocked=''; cap_after_wait_fits='false'; dram_arena_gib='70'; eval_wait_seconds='600'; expected_seconds='7000'; gpu_hours_billed_expected='1.94'; gpu_hours_billed_limit='2.88'; gpu_hours_expected='1.94'; gpu_hours_limit='2.88'; guaranteed_seconds='10560'; kv_cache_gib='25'; margin_seconds='300'; max_input_tokens='262144'; max_model_len='262144'; memory_need_gib='105.87'; memory_spare_gib='5.03'; partition='ghx4'; servers='sobst:observation_query:32tasks:2280s,srfillt:random_fill:32tasks:2280s,sgoalt:goal_query:32tasks:2280s'; spare_seconds='0'; startup_spike='ELIMINATED'; tasks='32'; time='02:53:00'; urgent=''; with_eval_wait_seconds='9840'; worst_case_seconds='10380'}
}
# The user's rule for every DeltaAI job: mail BEGIN, END and FAIL to this address; pinned, not a parameter.
$MailUser = 'cao58@illinois.edu'
$MailType = 'BEGIN,END,FAIL'
$PayloadPath = Join-Path $PSScriptRoot 'dist\payload.json'
if ((Get-FileHash -LiteralPath $PayloadPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $PinnedPayload) { throw 'sealed payload hash mismatch' }
$Bundle = [IO.File]::ReadAllText($PayloadPath) | ConvertFrom-Json
foreach ($Property in $Bundle.files.PSObject.Properties) {
    $Source = $Property.Name
    if ($Source -ceq 'tb16-tasks.json' -or $Source -ceq 'tasks-pool.json') { $Source = 'tasks\' + $Source }
    $Path = Join-Path $PSScriptRoot $Source
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Property.Value.sha256) { throw ('helper hash mismatch: '+$Property.Name) }
}
. (Join-Path $PSScriptRoot 'transport.ps1')
$Pins = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Bundle.files.'pins.json'.base64)) | ConvertFrom-Json
if ($Pins.commit -cne $ExpectedCommit -or $Pins.package_sha256 -cne 'f151fd29a2d58cb6aa3de27cfdd876ab5e00cd023a284015c84b440d5ea90e4e' -or $Pins.config_sha256 -cne $ExpectedConfig -or $Pins.clean_env.prefix -cne $ExpectedCleanEnv) { throw 'sealed pins differ from this entry point' }
if ($Pins.runtime_pin.status -cne $PinStatus) { throw 'sealed runtime pin status differs from this entry point' }
$Receipt = $Pins.release_receipt
if ($Receipt -and ($Receipt.commit -cne $ReceiptCommit -or $Receipt.result_line_sha256 -cne $ReceiptResult -or $Receipt.log_sha256 -cne $ReceiptLog -or [int]$Receipt.tests_run -ne $ReceiptTests)) { throw 'sealed release-check receipt differs from this entry point' }
if ($PinStatus -ceq 'PINNED' -and (-not $Receipt -or $Receipt.commit -cne $ExpectedCommit -or $Receipt.status -cne 'GIT_ARCHIVE_TESTS_PASS' -or $Receipt.result_line_sha256 -cnotmatch '^[0-9a-f]{64}$' -or $Receipt.log_sha256 -cnotmatch '^[0-9a-f]{64}$' -or $ReceiptTests -lt 1)) { throw ('the PINNED commit '+$ExpectedCommit+' has no matching release-check receipt (pinned: '+$ReceiptCommit+'); the login node runs no test suite') }
$ReceiptLine = 'RELEASE_RECEIPT commit='+$ReceiptCommit+' status='+[string]$Receipt.status+' tests='+$ReceiptTests+' skipped='+[string]$Receipt.skipped+' result_line_sha256='+$ReceiptResult+' log_sha256='+$ReceiptLog+' login_test_runs=0'
if ($Pins.slurm.mail_user -cne $MailUser -or $Pins.slurm.mail_type -cne $MailType -or $Pins.slurm.job_name_prefix -cne 'tb16-') { throw 'sealed Slurm pins differ from the mail rule' }
$Features = $Pins.runtime_features
if ($Features.stage2 -ne $true -or $Features.goal_async -ne $true -or $Features.decoupled_layout -ne $true -or @($Features.page_tokens_choices) -notcontains 16) { throw 'sealed runtime lacks stage 2, the asynchronous Goal or the decoupled 16-token layout' }
if (-not $Features.context_limits -or @($Features.context_limits.max_model_len_choices) -notcontains 65536 -or @($Features.context_limits.max_tokens_choices) -notcontains 4096) { throw 'sealed runtime has no 65536 / 4096 context limits' }
if (-not $Features.decode_fastpath -or $Features.decode_fastpath.serve_arg -cne '--decode-fastpath') { throw 'sealed runtime has no --decode-fastpath' }
if (-not $Features.language_model_only -or $Features.language_model_only.serve_arg -cne '--language-model-only') { throw 'sealed runtime has no serve.py --language-model-only' }
if (-not $Features.connector_pool -or $Features.connector_pool.serve_arg -cne '--connector-pool-gib') { throw 'sealed runtime has no --connector-pool-gib' }
if ($Pins.config.server.decode_fastpath2 -eq $true -and (-not $Features.decode_fastpath2 -or $Features.decode_fastpath2.serve_arg -cne '--decode-fastpath2')) { throw 'the decode_fastpath2 switch is on but the sealed runtime has no --decode-fastpath2' }
# The optimized runtime (2026-10-21): the Qwen servers run --decode-fastpath2 --decode-fastpath3 --deferred-completion-io
if ($PinStatus -ceq 'PINNED' -and ($Pins.config.server.decode_fastpath2 -ne $true -or $Pins.config.server.decode_fastpath3 -ne $true -or $Pins.config.server.deferred_completion_io -ne $true)) { throw 'sealed server switches differ from the optimized runtime (--decode-fastpath2 --decode-fastpath3 --deferred-completion-io on every Qwen runtime server)' }
if ($PinStatus -ceq 'PINNED' -and (-not $Features.decode_fastpath3 -or $Features.decode_fastpath3.serve_arg -cne '--decode-fastpath3' -or -not $Features.deferred_completion_io -or $Features.deferred_completion_io.serve_arg -cne '--deferred-completion-io')) { throw 'the sealed runtime has no --decode-fastpath3 / --deferred-completion-io' }
if ($Pins.config.server.kv_cache_gib -ne 32 -or $Pins.config.server.max_model_len -ne 65536 -or $Pins.config.server.max_tokens -ne 4096 -or $Pins.config.server.max_running_sessions -lt $Pins.config.agent.n_concurrent -or $Pins.config.server.decode_fastpath -ne $true -or $Pins.config.server.language_model_only -ne $true -or $Pins.config.server.connector_pool_gib -ne 1 -or $Pins.config.server.cuda_graphs -cne 'full' -or $Pins.config.stock.language_model_only -ne $true -or $Pins.config.stock.max_num_seqs -ne 16 -or $Pins.config.agent.n_concurrent -ne 16) { throw 'sealed server flags differ from the tb16 decision (32 GiB, 65536, 4096, K at least Harbor -n, --decode-fastpath, --language-model-only, --connector-pool-gib 1, graphs full; stock 16 seqs with --language-model-only; harbor -n 16)' }
if ($Pins.config.agent.thinking -cne 'off' -or $Pins.config.agent.agent_name -cne 'terminus-2' -or $Pins.config.agent.harbor_version -cne '0.21.0') { throw 'sealed agent settings differ from Harbor 0.21.0 Terminus 2 with thinking off' }
if (@($Pins.tasks.list).Count -ne $ExpectedTasks -or $ExpectedTasks -ne 89) { throw 'the sealed task list must hold the 89 Terminal-Bench 2.0 tasks' }
if ($Pins.slurm.mem -cne '113560M') { throw ('sealed sbatch memory '+$Pins.slurm.mem+' differs from --mem=113560M (one GH200''s share)') }
if (-not $Plan) { throw ('-Plan required: one of '+(@($Plans.Keys) -join ', ')+' (one ghx4 batch job per configuration: full, obs, rfill, goal, stock; dry / dryo / goaldry on the 16-task subset; dryi / goaldryi = the same on ghx4-interactive for urgent work (2x billing); g*: Gemma 4 31B IT; ctx24 / ctx25: goal_query then observation_query on the subset at 262,144 tokens, 24 / 25 GiB; gctx25: Gemma 4 31B IT on the subset at 262,144 tokens, three runtime servers and the stock control)') }
if (-not $Plans.Contains($Plan)) { throw ('unknown plan '''+$Plan+''' (one of '+(@($Plans.Keys) -join ', ')+')') }
$Chosen = $Plans[$Plan]
if ($Chosen.startup_spike -ceq 'RISK') { throw ('plan '+$Plan+': a server starts without --language-model-only') }
# A Gemma plan (model gemma4-31b-it): the sealed Gemma decisions (see below) and a runtime that serves Gemma's chat-text
# turns (and, for a Gemma goal plan, goal_query on Gemma's chat-text).
if ([string]$Chosen.model -cne [string]$Pins.config.plans.$Plan.model) { throw 'sealed plan model differs from this entry point' }
if ($Chosen.model -ceq 'gemma4-31b-it') {
    $G = $Pins.config.gemma
    # the user's decision of 2026-10-21: 25 GiB with K = 17 for the selective policies, full at 32 GiB with K = 20, the
    # stock control at 25 GiB; since the integrate-1021 9ef0d6d re-pin (2026-10-22) config.server's three fast paths and
    # the deferred files on the Gemma servers too (config.gemma.server sets none of them)
    if (-not $G -or $G.model_profile -cne 'gemma4-31b-it' -or $G.server.kv_cache_gib -ne 25 -or $G.server.max_running_sessions -ne 17 -or $G.server.release_idle_pages -ne $true -or $null -ne $G.server.decode_fastpath2 -or $null -ne $G.server.decode_fastpath3 -or $null -ne $G.server.deferred_completion_io -or $G.policy_server.full.kv_cache_gib -ne 32 -or $G.policy_server.full.max_running_sessions -ne 20 -or $G.stock.kv_cache_gib -ne 25 -or $G.stock.max_num_seqs -ne 16 -or $G.stock.attention_backend -cne 'TRITON_ATTN' -or $G.stock.language_model_only -ne $true) { throw 'sealed Gemma decisions differ from this entry point (25 GiB at K = 17, full 32 GiB at K = 20, --release-idle-pages, the fast paths and the deferred files of config.server, stock TRITON_ATTN 25 GiB 16 seqs)' }
    if ($PinStatus -ceq 'PINNED' -and (-not $Features.gemma -or $Features.gemma.chat_text -ne $true -or $Features.gemma.servable -ne $true -or $Features.gemma.profile -cne 'gemma4-31b-it')) { throw 'sealed runtime serves no Gemma 4 chat-text server' }
    if ($PinStatus -ceq 'PINNED' -and (($Pins.config.server.decode_fastpath3 -eq $true -and $Features.gemma.decode_fastpath3 -ne $true) -or ($Pins.config.server.deferred_completion_io -eq $true -and $Features.gemma.deferred_completion_io -ne $true))) { throw 'the sealed Gemma launcher refuses --decode-fastpath3 or --deferred-completion-io (before integrate-1021)' }
    if (-not $Pins.gemma_model -or $Pins.gemma_model.confirmed -ne $true -or $Pins.gemma_model.dir -cnotmatch '^/[A-Za-z0-9._/-]+$') { throw 'sealed Gemma checkpoint pin missing' }
    if ($Chosen.authorization -cnotmatch '_GEMMA4_') { throw 'a Gemma plan phrase names GEMMA4' }
}
# goal_query (the goal plans): the runtime's chat-text Goal inputs (provisional) and the rewriter batch with its token cap
$GoalPlan = @($Pins.config.plans.$Plan.servers | Where-Object { $_.policy -ceq 'goal_query' }).Count -gt 0
if ($GoalPlan -and $Chosen.model -ceq 'gemma4-31b-it' -and $PinStatus -ceq 'PINNED' -and $Features.gemma.goal_query -ne $true) { throw 'the sealed Gemma launcher serves no chat-text goal_query (gemma-tb-goal, integrate-1021)' }
if ($GoalPlan -and $PinStatus -ceq 'PINNED' -and (-not $Features.chat_text_goal -or $Features.chat_text_goal.DEFINITION -cne 'terminus2-goal-inputs-provisional-v2' -or -not $Features.goal_rewriter_batch -or $Features.goal_rewriter_batch.tokens_key -cne 'goal_rewriter_batch_tokens')) { throw 'the sealed runtime serves no chat-text goal_query with the provisional Terminus 2 Goal inputs and the rewriter batch token cap' }
# A plan with its own server settings (ctx25 / ctx24, 2026-10-26): 262,144 tokens, 25 or 24 GiB, the 70-GiB arena and
# Terminus 2's model_info at the same context; a KV size the sealed Qwen launcher lacks (serve.py KV_CACHE_GIB, not its
# argparse choices) must have blocked the plan (controller.kv_block). Every other plan's checks and line are unchanged.
$Override = $Pins.config.plans.$Plan.server
$Context = ''
if ($Override -and $Chosen.model -ceq 'gemma4-31b-it') {
    # gctx25 (2026-11-09): Gemma 4 31B IT at 262,144 tokens (the runtime's TERMINALBENCH A.14): 25 GiB, K = 20, the
    # 4,096-token forward, the 72-GiB arena and Terminus 2 at 262,144; a sealed Gemma launcher without them must have
    # blocked the plan (controller.gemma_plan_block)
    if ([int]$Override.max_model_len -ne 262144 -or [int]$Override.kv_cache_gib -ne 25 -or [int]$Override.dram_arena_gib -ne 72 -or [int]$Override.max_running_sessions -ne 20 -or [int]$Override.max_num_batched_tokens -ne 4096 -or [int]$Chosen.max_model_len -ne 262144 -or [int]$Chosen.max_input_tokens -ne 262144 -or [int]$Chosen.kv_cache_gib -ne 25 -or [int]$Chosen.dram_arena_gib -ne 72 -or [int]$Chosen.max_running_sessions -ne 20 -or [int]$Chosen.max_num_batched_tokens -ne 4096) { throw 'sealed plan server settings differ from this entry point (Gemma 4 31B IT, 262144 tokens, 25 GiB, K = 20, 4096-token forwards, the 72-GiB arena, Terminus 2 max_input_tokens 262144)' }
    if ($PinStatus -ceq 'PINNED' -and -not [string]$Chosen.blocked -and (@($Features.gemma.max_model_lens) -notcontains 262144 -or @($Features.gemma.kv_cache_gib) -notcontains 25 -or -not $Features.gemma.max_num_batched_tokens -or @($Features.gemma.max_num_batched_tokens.choices) -notcontains 4096)) { throw ('the sealed Gemma launcher serves no 262144 tokens / 25 GiB / --max-num-batched-tokens 4096 but plan '+$Plan+' is not blocked') }
    # decision D-A (2026-11-09): the sealed Terminal-Bench client must write the Gemma job at the plan's context, else the
    # build has blocked the plan (controller.client_block; main ba4d1ba's make_job.py: Gemma 65536)
    if ($PinStatus -ceq 'PINNED' -and -not [string]$Chosen.blocked -and [int]$Features.tb_client.context.'gemma4-31b-it' -ne [int]$Override.max_model_len) { throw ('the sealed Terminal-Bench client writes the Gemma job at '+$Features.tb_client.context.'gemma4-31b-it'+' tokens but plan '+$Plan+' is not blocked') }
    $Context = ' kv_cache_gib='+$Chosen.kv_cache_gib+' max_model_len='+$Chosen.max_model_len+' dram_arena_gib='+$Chosen.dram_arena_gib+' max_input_tokens='+$Chosen.max_input_tokens+' max_running_sessions='+$Chosen.max_running_sessions+' max_num_batched_tokens='+$Chosen.max_num_batched_tokens
} elseif ($Override) {
    if ($Chosen.model -or [int]$Override.max_model_len -ne 262144 -or @(24, 25) -notcontains [int]$Override.kv_cache_gib -or [int]$Override.dram_arena_gib -ne 70 -or @($Features.context_limits.max_model_len_choices) -notcontains 262144 -or [int]$Chosen.max_model_len -ne 262144 -or [int]$Chosen.max_input_tokens -ne 262144 -or [int]$Chosen.kv_cache_gib -ne [int]$Override.kv_cache_gib -or [int]$Chosen.dram_arena_gib -ne 70) { throw 'sealed plan server settings differ from this entry point (Qwen, 262144 tokens, 25 or 24 GiB, the 70-GiB arena, Terminus 2 max_input_tokens 262144)' }
    if (@($Features.qwen_kv_cache_gib) -notcontains [int]$Override.kv_cache_gib -and -not [string]$Chosen.blocked) { throw ('the sealed Qwen launcher serves no --kv-cache-gib '+$Override.kv_cache_gib+' (KV_CACHE_GIB '+(@($Features.qwen_kv_cache_gib) -join ',')+') but plan '+$Plan+' is not blocked') }
    $Context = ' kv_cache_gib='+$Chosen.kv_cache_gib+' max_model_len='+$Chosen.max_model_len+' dram_arena_gib='+$Chosen.dram_arena_gib+' max_input_tokens='+$Chosen.max_input_tokens
}
# perf-tb16-strip: a thinking plan (config.plans.<name>.thinking, strip25) runs --enable-thinking --thinking-history strip;
# the sealed launcher must serve it (runtime_features.thinking), else the build has blocked the plan (thinking_block)
$Thinking = $Pins.config.plans.$Plan.thinking
if ($Thinking) {
    if ([string]$Thinking.history -cne 'strip' -or $Chosen.model) { throw 'sealed plan thinking differs from this entry point (Qwen, history strip)' }
    if (-not $Features.thinking -and -not [string]$Chosen.blocked) { throw ('the sealed launcher has no --thinking-history strip but plan '+$Plan+' is not blocked') }
    $Context += ' thinking=strip'
}
$Blocked = [string]$Chosen.blocked
$Sealed = $Pins.config.plans.$Plan
if (-not $Sealed -or $Sealed.authorization -cne $Chosen.authorization -or $Sealed.time -cne $Chosen.time -or $Sealed.partition -cne $Chosen.partition) { throw 'sealed plan differs from this entry point' }
# the release smoke plans (2026-11-03, role release-smoke): their task set, D2's async switch, D1's Goal rewriter and
# --goal-overlap and the coverage line on the plan's line (every other plan's line unchanged)
$SmokeGoal = $null
if ([string]$Sealed.role -ceq 'release-smoke') {
    $SmokeAsync = 'off'
    if ($Sealed.async_scheduling -eq $true) {
        # (decision D-C, 2026-11-09: a Gemma plan's --async-scheduling needs async-gemma's prepare_gemma4, else the build
        # has blocked the plan; decision D-G: a runtime without the async fixes has blocked any async plan)
        if (((-not $Features.async_scheduling) -or ($Chosen.model -and $Features.gemma.async_scheduling -ne $true) -or (-not $Features.async_fix)) -and -not [string]$Chosen.blocked) { throw ('plan '+$Plan+': --async-scheduling on a launcher without it (a Gemma plan: without async-gemma) or without the async fixes, but the plan is not blocked') }
        $SmokeAsync = 'on'
    }
    $SmokeGoal = @($Sealed.servers | Where-Object { $_.policy -ceq 'goal_query' })[0]
    $Context += ' role=release-smoke task_set='+((@($Sealed.servers | ForEach-Object { $_.task_set }) | Sort-Object -Unique) -join '/')+' async_scheduling='+$SmokeAsync+' coverage=evictions'
}
if ($Chosen.partition -cne 'ghx4' -and -not ($Chosen.partition -ceq 'ghx4-interactive' -and $Chosen.urgent)) { throw ('plan '+$Plan+': tb16 plans run on ghx4 batch; ghx4-interactive only for a plan that names its urgent reason') }
if ([int]$Pins.slurm.partition_limits.($Chosen.partition).billing_factor -ne [int]$Chosen.billing_factor -or [int]$Pins.slurm.partition_limits.'ghx4-interactive'.billing_factor -ne 2) { throw 'sealed billing factor differs from this entry point (ghx4-interactive bills 2x)' }
$PartitionMax = $Pins.slurm.partition_limits.($Chosen.partition).max_time
if (-not $PartitionMax -or ([TimeSpan]::Parse($Chosen.time) -gt [TimeSpan]::Parse($PartitionMax))) { throw ('plan '+$Plan+' asks --time='+$Chosen.time+', above the '+$Chosen.partition+' maximum '+$PartitionMax) }
if ([int]$Chosen.worst_case_seconds -gt [TimeSpan]::Parse($Chosen.time).TotalSeconds -or [int]$Chosen.with_eval_wait_seconds -gt [TimeSpan]::Parse($Chosen.time).TotalSeconds) { throw ('plan '+$Plan+' does not fit its Slurm time') }
if (-not $RewriterModel) { $RewriterModel = $Pins.rewriter_default }
$RewriterModel = $RewriterModel.TrimEnd('/')
if ($RewriterModel -cnotmatch '^/[A-Za-z0-9._/-]+$') { throw 'RewriterModel must be an absolute DeltaAI path of [A-Za-z0-9._/-]' }
if ($RunLabel -cnotmatch '^r[0-9]{1,2}$') { throw 'RunLabel must match r[0-9]{1,2} (for example r1, r2)' }
if ($AfterJob -and $AfterJob -cnotmatch '^[0-9]{1,12}$') { throw 'AfterJob must be a Slurm job id' }
$AfterValue = $null
if ($AfterJob) { $AfterValue = $AfterJob }
$Nonce = [Guid]::NewGuid().ToString('N')
$BootstrapCode = $Bundle.files.'bootstrap.py'.base64
$PythonCode = 'import base64;exec(compile(base64.b64decode("'+$BootstrapCode+'"),"sealed-bootstrap","exec"))'
$RemoteCommand = '/work/nvme/bcjw/ycao11/vllm-arm/bin/python -I -S -B -c ' + (ConvertTo-BashLiteral $PythonCode) + ' ' + $Nonce
$Options = [ordered]@{plan=$Plan; authorization=$Chosen.authorization; mail_user=$MailUser; rewriter_model=$RewriterModel; run_label=$RunLabel; after_job=$AfterValue}
$Bundle | Add-Member -MemberType NoteProperty -Name options -Value $Options
$Bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Bundle | ConvertTo-Json -Depth 10 -Compress))
# the plan's switches (a Gemma plan: config.gemma.server over config.server; since 2026-10-22 it sets none of them)
$Switches = $Pins.config.server
if ($Chosen.model -ceq 'gemma4-31b-it') {
    $Switches = [ordered]@{}
    foreach ($Key in 'decode_fastpath2','decode_fastpath3','deferred_completion_io') { $Switches[$Key] = $Pins.config.server.$Key; if ($null -ne $Pins.config.gemma.server.$Key) { $Switches[$Key] = $Pins.config.gemma.server.$Key } }
    $Switches = [pscustomobject]$Switches
}
$OnOff = { param($Value) if ($Value -eq $true) { 'on' } else { 'off' } }
$Fast = 'decode_fastpath2='+(& $OnOff $Switches.decode_fastpath2)+' decode_fastpath3='+(& $OnOff $Switches.decode_fastpath3)+' deferred_completion_io='+(& $OnOff $Switches.deferred_completion_io)
$Goal = ''
if ($GoalPlan -and $SmokeGoal -and $SmokeGoal.goal_rewriter_engine) {
    # the release smoke (D1): the vLLM Goal rewriter with its settings and --goal-overlap, no --goal-rewriter-batch
    if ([string]$SmokeGoal.goal_rewriter_engine -cne 'vllm' -or ((-not $Features.goal_rewriter_engine -or ($SmokeGoal.goal_overlap -and -not $Features.goal_overlap)) -and -not [string]$Chosen.blocked)) { throw ('plan '+$Plan+': the vLLM Goal rewriter / --goal-overlap on a launcher without them') }
    $Overlap = ''
    if ($SmokeGoal.goal_overlap -eq $true) { $Overlap = ',goal_overlap' }
    $Invariant = ''
    if ($SmokeGoal.goal_rewriter_batch_invariant -eq $true) { $Invariant = ',batch_invariant' }
    $Goal = ' goal_query=admission_overlap16,goal_rewriter_engine=vllm,guided_backend='+$SmokeGoal.goal_rewriter_guided_backend+',kv_gib='+$SmokeGoal.goal_rewriter_vllm_kv_gib+$Invariant+$Overlap+',rewriter_model='+$RewriterModel+',goal_inputs='+[string]$Features.chat_text_goal.DEFINITION
} elseif ($GoalPlan) {
    $Goal = ' goal_query=admission_overlap16,goal_rewriter_batch4,rewriter_model='+$RewriterModel+',goal_inputs='+[string]$Features.chat_text_goal.DEFINITION
    # a Gemma goal server's explicit T (controller.gemma_goal_gpu); the Qwen servers keep serve.py's default
    $Tokens = @($Pins.config.plans.$Plan.servers | Where-Object { $null -ne $_.goal_rewriter_batch_tokens } | ForEach-Object { $_.goal_rewriter_batch_tokens })
    if ($Tokens.Count) { $Goal = $Goal+',goal_rewriter_batch_tokens='+($Tokens -join '/') }
}
$Provisional = 'runtime_pin='+$PinStatus+' tasks='+$ExpectedTasks+' kv_budget='+$Pins.config.selection.active_token_budget+' decode_fastpath=on '+$Fast+' language_model_only=on connector_pool_gib='+$Pins.config.server.connector_pool_gib+$Goal+' still_provisional='+((@($Pins.config.provisional.PSObject.Properties | ForEach-Object { $_.Name }) | Sort-Object) -join ',')
$Model = ''
if ($Chosen.model) { $Model = ' model='+$Chosen.model }
$Summary = ' plan='+$Plan+$Model+' servers='+$Chosen.servers+$Context+' partition='+$Chosen.partition+' time='+$Chosen.time+' expected_seconds='+$Chosen.expected_seconds+' gpu_hours_limit='+$Chosen.gpu_hours_limit+' gpu_hours_expected='+$Chosen.gpu_hours_expected+' billing=x'+$Chosen.billing_factor+' billed_gpu_hours_limit='+$Chosen.gpu_hours_billed_limit+' billed_gpu_hours_expected='+$Chosen.gpu_hours_billed_expected+' worst_case_seconds='+$Chosen.worst_case_seconds+' with_eval_wait_seconds='+$Chosen.with_eval_wait_seconds+' cap_after_wait_fits='+$Chosen.cap_after_wait_fits+' eval_wait_seconds='+$Chosen.eval_wait_seconds+' margin_seconds='+$Chosen.margin_seconds+' spare_seconds='+$Chosen.spare_seconds+' mem='+$Pins.slurm.mem+' memory_need_gib='+$Chosen.memory_need_gib+' memory_spare_gib='+$Chosen.memory_spare_gib+' startup_spike='+$Chosen.startup_spike+' job_name='+$Pins.slurm.job_name_prefix+$Plan+' run_label='+$RunLabel+' after_job='+$AfterJob+' mail_user='+$MailUser+' mail_type='+$MailType+' '+$Provisional
# 2026-11-12 (config.tbench_verify; perf-swe32's evidence push): every plan's job pushes its two archives to tbench and
# starts the verification there at its end (best effort, bounded); the line names it (the only new field: recorded
# lines compare with it masked).
$TbenchPush = 'off'
if ($Pins.config.tbench_verify -and $Pins.config.tbench_verify.push) {
    if ($Pins.config.tbench_verify.root -cne '/home/ubuntu/tb16-verify' -or [int]$Pins.config.tbench_verify.total_seconds -gt 300) { throw 'sealed tbench_verify settings differ from the evidence-push decision' }
    $TbenchPush = $Pins.config.tbench_verify.root+':'+$Pins.config.tbench_verify.total_seconds+'s'
}
$Summary += ' tbench_push='+$TbenchPush
if ($SelfTest) {
    $Tokens=$null; $Errors=$null
    [void][Management.Automation.Language.Parser]::ParseFile($PSCommandPath,[ref]$Tokens,[ref]$Errors)
    if ($Errors.Count) { throw 'PowerShell syntax invalid' }
    $Info = New-OneConnectionStartInfo -Executable 'ssh' -RemoteCommand $RemoteCommand
    if ($Info.CreateNoWindow -or $Info.RedirectStandardOutput -or $Info.RedirectStandardError -or -not $Info.RedirectStandardInput) { throw 'visible SSH console contract broken' }
    if (-not $Info.Arguments.Contains('ConnectionAttempts=1')) { throw 'single connection contract broken' }
    $RoundTrip = [Text.Encoding]::UTF8.GetString($Bytes) | ConvertFrom-Json
    if ($RoundTrip.files.'bootstrap.py'.base64 -cne $BootstrapCode -or $RoundTrip.options.plan -cne $Plan -or $RoundTrip.options.run_label -cne $RunLabel -or $RoundTrip.options.rewriter_model -cne $RewriterModel -or $RoundTrip.options.mail_user -cne $MailUser -or $RoundTrip.options.authorization -cne $Chosen.authorization) { throw 'sealed JSON binary transport failed' }
    foreach ($Name in 'job.py','controller.py','tb_checks.py','smoke_checks.py','measure.py','resources.py','tbench_link.py','stock.py','goal_text.py','tbench_known_hosts','tb16-tasks.json','tasks-pool.json') { if (-not $RoundTrip.files.$Name.sha256) { throw ('sealed file missing: '+$Name) } }
    if ($Pins.slurm.account -cne 'bcjw-dtai-gh') { throw 'Slurm pin changed' }
    if ($Pins.clean_env.snapshot_sha256 -cne 'cbfab91650bfbeffb149a2df8ead3787a07b91a2cf332edd7efd0b5b0a07e5f0' -or $Pins.clean_env.release_manifest_sha256 -cne 'b23b5f20e7c85f8f8e3ab93fe51d8519dfd465aa50fecf0cda27c95d18e9af93') { throw 'reused v9 environment pin changed' }
    if ($Pins.wheel_sha256 -cne '700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5' -or -not $Pins.wheel_url.EndsWith('/vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl')) { throw 'official vLLM wheel pin changed' }
    if (([Text.Encoding]::UTF8.GetByteCount($StagePrefix+$Nonce+'/tmp')+1+36) -gt 99) { throw 'stage path exceeds IPC budget' }
    Write-Output ('LOCAL_SELFTEST_PASS connections=0 install=0 submit=0 payload_sha256='+$PinnedPayload+' commit='+$ExpectedCommit+$Summary)
    Write-Output $ReceiptLine
    if ($Blocked) { Write-Output ('BLOCKED plan='+$Plan+' reason='+$Blocked) }
    return
}
if (-not $Authorize) {
    if ($Authorization) { throw 'explicit -Authorize switch also required' }
    Write-Output ('LOCAL_ONLY_PASS connections=0 install=0 submit=0 plan='+$Plan+' '+$Provisional)
    return
}
if ($Authorization -cne $Chosen.authorization) { throw ('exact authorization phrase of plan '+$Plan+' required') }
if ($DryRun) {
    Write-Output ('DRY_RUN connections=0 stage='+$StagePrefix+$Nonce+' payload_bytes='+$Bytes.Length+' commit='+$ExpectedCommit+$Summary)
    Write-Output $ReceiptLine
    if ($PinStatus -cne 'PINNED') { Write-Output ('NOT_SUBMITTABLE runtime_pin='+$PinStatus+' reason='+$PinReason) }
    if ($Blocked) { Write-Output ('NOT_SUBMITTABLE blocked_plan='+$Plan+' reason='+$Blocked) }
    return
}
if ($Blocked) { throw ('plan '+$Plan+' is blocked: '+$Blocked+'. Nothing was submitted.') }
if ($PinStatus -cne 'PINNED') { throw ('the runtime pin is a '+$PinStatus+' ('+$PinReason+'): nothing was submitted. Re-pin with build.py --commit <the chat-protocol commit> once the harness is adapted to its interface (README "Re-pinning").') }
Write-Output ('PLANNED_STAGE='+$StagePrefix+$Nonce+' PLAN='+$Plan)
Write-Output ('One visible SSH/Duo connection: reuse the v9 clean env read-only, private release and the light pre-claim gates (about 3-5 min: the pinned release-check receipt of '+$ExpectedCommit.Substring(0, 7)+' instead of any test suite, bundle verify, git fsck, S0/S1 with the official-wheel RECORD, the C-compiler probe, one serve.py dry run per server, the tbench preflight with Harbor''s RECORD check), then exactly one ghx4 GH200 sbatch of plan '+$Plan+' (--time='+$Chosen.time+', mail on BEGIN, END, FAIL to '+$MailUser+'). Any unknown outcome must not be retried.')
if ($Chosen.partition -ceq 'ghx4-interactive') {
    Write-Output ('ghx4-interactive (urgent: '+$Chosen.urgent+'; 1 GPU at --mem=113560M, billed x'+$Chosen.billing_factor+': at most '+$Chosen.gpu_hours_billed_limit+' billed GPU-hours, '+$Chosen.gpu_hours_billed_expected+' expected): one submitted job per user there; the queue gate stops before the claim (exit 86) while squeue lists one of yours on ghx4-interactive.')
} else {
    Write-Output ('ghx4 batch (1 GPU at --mem=113560M: one GPU-hour per hour, at most '+$Chosen.gpu_hours_limit+' for this plan): the job waits in the queue by fairshare priority (possibly hours; the BEGIN mail says when it starts).')
}
Write-Output 'The queue gate also stops before the claim while another job of a harness sharing tbench (tb16 / swe32 / swe10 / swe16 / gem0 / gswe32: tbench serves one job at a time) is queued or running, unless -AfterJob names it (or the jobs it waits for: the submission chain, QUEUE_GATE_CHAIN) or it is PENDING on your hold (scontrol hold <J>: reason JobHeldUser). A held job is listed as QUEUE_GATE_HELD / RELEASE_AFTER_TB: run scontrol release <J> after this TB job ends. The tbench preflight refuses (no claim) while a TB run is live on tbench: submit the chain before its first job starts.'
$Code = Invoke-OneConnectionTransport -Executable (Get-Command ssh -CommandType Application | Select-Object -First 1).Source -RemoteCommand $RemoteCommand -PayloadBytes $Bytes
if ($Code -eq 86) { throw 'Pre-claim setup failed: remote confirmed NO_SUBMISSION_CLAIM. No sbatch was reached. Read the printed QUALIFY_ERROR/ERROR line, fix it, then start a separately authorized run with a new -RunLabel; no automatic retry.' }
if ($Code -ne 0) { throw ('Setup failed or submission outcome unknown; consult the printed FAILURE_STATUS/CLAIM_STATE. Without confirmed NO_SUBMISSION_CLAIM, keep the stage and claim and reconcile read-only before any replacement. No automatic retry. SSH exit='+$Code) }
Write-Output 'SUBMITTED: the last JSON line above holds stage, job_id and run_id (held_job_ids / release_after_tb: the jobs to release with scontrol release after this TB job ends). After the END mail: the README check block, then postrun\fetch_run.ps1 (both sides, verify, join, tables).'
