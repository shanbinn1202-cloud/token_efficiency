"""Method construction and explicit backend capability checks."""
from dataclasses import asdict, fields, replace
from importlib import import_module

HISTORY_METHODS = {
    "agentdiet": ("baselines.agentdiet.model", "AgentDietModel", "baselines.agentdiet.model", "POLICIES"),
    "vanilla": ("baselines.vanilla.model", "FullHistoryModel", "baselines.vanilla.model", "POLICIES"),
    "sliding_window": ("baselines.sliding_window.model", "SlidingWindowModel", "baselines.sliding_window.model", "POLICIES"),
    "group": ("baselines.group.model", "GroupModel", "baselines.group.memory", "INLINE_SCOPE_POLICIES"),
}
METHODS = (*HISTORY_METHODS, "agentkv", "dram_backed_kv", "repairkv", "intentkv", "random_attention")


def check_support(method, config):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    backend = config["backend"]["name"]
    if backend != "vllm" and config["backend"]["kv_cache_memory_bytes"] is not None:
        raise ValueError("An explicit GPU KV pool size requires backend=vllm")
    if method in {"sliding_window", "agentkv"} and backend == "vllm" and config["backend"]["tensor_parallel_size"] != 1:
        raise ValueError("Local vLLM KV window currently requires tensor_parallel_size=1")
    if backend == "transformers" and config["evaluation"]["workers"] != 1:
        raise ValueError("In-process Transformers requires evaluation.workers=1 (stateful cache)")
    if backend == "transformers" and config["backend"]["tensor_parallel_size"] != 1:
        raise ValueError("In-process Transformers currently supports tensor_parallel_size=1")
    if backend == "transformers" and config["backend"]["cuda_graph"]:
        raise ValueError("Transformers CUDA Graph is not implemented for the aligned paged/stateful caches; use cuda_graph=false")
    if backend == "transformers" and config["backend"]["torch_compile"]:
        raise ValueError("torch_compile=true is not implemented for the aligned stateful adapters; use eager execution")
    if backend == "transformers" and method != "agentkv" and config["backend"]["prefill_chunk_size"] is not None:
        raise ValueError("Explicit prefill chunking is currently supported only by AgentKV; aligned runs use null")
    if method in {"agentkv", "dram_backed_kv"} and config["model"]["dtype"] != "bfloat16":
        raise ValueError(f"{method} adapter currently requires bfloat16")
    if method in {"agentkv", "dram_backed_kv"} and any(config["generation"][k] != v for k, v in
            (("top_p", 1.0), ("top_k", -1), ("repetition_penalty", 1.0))):
        raise ValueError(f"{method} adapter currently requires top_p=1, top_k=-1, repetition_penalty=1")
    if method == "random_attention":
        if backend != "vllm" or config["backend"]["tensor_parallel_size"] != 1:
            raise ValueError("Random Attention requires the local vLLM TP=1 backend")
        options = config["methods"][method]["options"]
        if (set(options) != {"window_size", "random_seed"}
                or options["window_size"] != 64
                or type(options["random_seed"]) is not int):
            raise ValueError("Random Attention requires the upstream 64-token window and an integer selector seed")
    if method == "repairkv":
        from adapters.repairkv.config import check
        check(config)
    if method == "intentkv":
        if backend != 'vllm' or config['backend']['tensor_parallel_size'] != 1:
            raise ValueError('IntentKV requires the local vLLM TP=1 backend')
        options = config['methods'][method]['options']
        if set(options) != {'init_q_window', 'block_size'} or options['block_size'] != 16:
            raise ValueError('IntentKV uses the original evaluation 16-token block setting')
        if options['init_q_window'] < 19 or config['cache']['kv_budget'] <= options['init_q_window']+15:
            raise ValueError('IntentKV requires a valid original JSD window and positive block capacity')
    if method == "dram_backed_kv":
        if config["generation"]["temperature"] != 0:
            raise ValueError("DRAM upstream runtime currently implements greedy generation only")
        if config["cache"]["kv_budget"] % 16:
            raise ValueError("DRAM kv_budget must be divisible by the upstream 16-token page size")
        options = config["methods"][method]["options"]
        allowed = {"query_source", "recent_turns", "recent_quota", "hidden_layer", "projection_dimension", "projection_seed",
                   "budget_mode", "active_ratio", "goal_rewriter_model", "goal_rewriter_max_new_tokens", "goal_rewriter_timeout"}
        if set(options) - allowed:
            raise ValueError(f"Unknown DRAM options: {set(options) - allowed}")
        if options["query_source"] not in {"goal", "observation"}:
            raise ValueError("DRAM query_source must be goal or observation")
        if options["budget_mode"] not in {"fixed", "ratio"} or not 0 < options["active_ratio"] <= 1:
            raise ValueError("DRAM requires budget_mode=fixed|ratio and 0 < active_ratio <= 1")


def policy_for(method, config):
    if method not in HISTORY_METHODS:
        return dict(config["methods"][method]["options"], kv_budget=config["cache"]["kv_budget"])
    _, _, module, attr = HISTORY_METHODS[method]
    policies = getattr(import_module(module), attr)
    settings = config["methods"][method]
    if settings["variant"] not in policies:
        raise ValueError(f"Unknown {method} variant: {settings['variant']}")
    policy = policies[settings["variant"]]
    unknown = settings["options"].keys() - {f.name for f in fields(policy)}
    if unknown:
        raise ValueError(f"Unknown {method} policy options: {sorted(unknown)}")
    return replace(policy, **settings["options"])


def policy_snapshot(method, config):
    policy = policy_for(method, config)
    snapshot = policy if isinstance(policy, dict) else asdict(policy)
    if method in {"vanilla", "agentdiet"} and config["backend"]["name"] == "vllm":
        snapshot.update(engine_overrides={"prefix_caching": config["backend"]["prefix_caching"],
                                           "async_scheduling": config["backend"]["async_scheduling"]})
    if method == "agentdiet":
        snapshot.update(budget_scope="none", selection="CoACT_delayed_history_reduction",
            scheduling="native_vllm_request_lifecycle" if config["backend"]["name"] == "vllm" else "single_worker",
            reflection_model="same_as_agent", reflection_temperature=0.0,
            reflection_max_tokens=config["generation"]["max_tokens"],
            reflection_thinking=snapshot["reflection_enable_thinking"],
            reflection_stop=["</step>"], token_counter="CoACT_litellm_token_counter")
    if method == "sliding_window":
        snapshot.update(kv_budget=config["cache"]["kv_budget"], budget_scope="input",
                        selection="recent_tokens",
                        storage="rerendered_tokens", linear_state="recomputed")
        if config["backend"]["name"] == "vllm":
            snapshot.update(budget_scope="fixed_prefix_plus_recent_KV_compaction_target",
                            selection="initial_system_and_task_prefix_plus_recent_tokens",
                            storage="native_full_attention_pages_with_CPU_turn_checkpoint",
                            trigger="after_prefill_and_decode_physical_length_divisible_by_64",
                            attention_path="vanilla_native_fa3_gqa",
                            linear_state="retained", logical_positions="preserved",
                            scheduling="native_async_fcfs_and_recompute_preemption")
            snapshot["engine_overrides"] = {"prefix_caching": False,
                                            "attention_backend": "FLASH_ATTN", "flash_attn_version": 3,
                                            "async_scheduling": config["backend"]["async_scheduling"]}
    if method == "agentkv" and config["backend"]["name"] == "vllm":
        snapshot.update(storage="persistent_per_layer_head_kv", budget_scope="decode_compaction_target",
            scheduling="native_vllm_request_lifecycle_and_recompute_preemption",
            session_admission="native_fcfs_no_extra_reservation",
            idle_storage="lossless_cpu_turn_checkpoint_via_native_KVConnector",
            upstream_commit="b5280b7c42bd90962712c19f8e0047c89ca40d6d",
            selection="upstream_StageQSnapKVEvictionPlan", linear_state="retained",
            engine_overrides={"prefix_caching": False, "async_scheduling": config["backend"]["async_scheduling"],
                              "attention_backend": "FLASH_ATTN"})
    if method == "dram_backed_kv" and config["backend"]["name"] == "vllm":
        snapshot.update(storage="native_vllm_blocks_with_upstream_cpu_dram_runtime", budget_scope="historical_pages",
            selection="upstream_online_selector_v2_hidden_page_vectors",
            page_size=16, linear_state="retained", recompute_selected_pages=False,
            engine_overrides={"prefix_caching": False, "async_scheduling": config["backend"]["async_scheduling"],
                              "attention_backend": "FLASH_ATTN", "goal_query_scratch_slots": 1})
    if method == "repairkv":
        snapshot.update(storage="upstream_model_device_pool_and_cpu_evicted_partition",
            budget_scope="resumed_history_after_each_turn",
            protocol="upstream_full_pool_repartition_with_growing_history_budget",
            base_selection="all_committed_history_with_fresh_scores_each_turn",
            scoring="upstream_score_evicted_positions_default_cpu",
            cpu_execution="allocated_cores_during_upstream_calls_restore_vllm_threads_afterward",
            restoration="upstream_host_fragment_transfer_and_inject_kv",
            selection="upstream_SnapKV_and_exact_Q_burst_repair",
            probe="rollback_full_attention_and_recurrent_state_then_replay",
            scheduling="native_vllm_nonresumable_requests_and_recompute_preemption",
            cache_loading="native_KVConnector",
            engine_overrides={"prefix_caching": False, "async_scheduling": config["backend"]["async_scheduling"],
                              "attention_backend": "FLASH_ATTN"})
    if method == 'intentkv':
        snapshot.update(upstream_commit='065f394dbb14f317d2b4dba252e8e66b91afa8c7',
            selection='upstream_IntentKV_scores_then_mean_block_scores_within_GQA_group',
            algorithm_variant='user_requested_GQA_shared_selection',
            budget_scope='initial_prompt_per_KV_head_then_append_decode',
            protocol='fresh_cache_each_generate_request',
            storage='compacted_per_KV_head_native_pages',
            attention_path='vanilla_native_fa3_gqa',
            scheduling='native_request_lifecycle_and_recompute_preemption',
            engine_overrides={'prefix_caching': False, 'async_scheduling': config['backend']['async_scheduling'],
                              'attention_backend': 'FLASH_ATTN', 'flash_attn_version': 3})
    if method == "random_attention":
        snapshot.update(selection="verbatim_upstream_random_pp_per_layer_KV_head",
            upstream_commit="64db9688a12f0926db45fec039370f6ceb1ab4fe",
            protocol="persistent_across_turns_initial_system_and_task_prefix",
            budget_scope="total_retained_KV_including_prefix_and_recent_64",
            trigger="decode_only_physical_length_above_budget_and_divisible_by_64",
            storage="native_full_attention_pages_with_lossless_CPU_turn_checkpoint",
            discarded_KV="not_archived", linear_state="retained", logical_positions="preserved",
            scheduling="native_async_fcfs_and_recompute_with_recorded_keep_sets",
            engine_overrides={"prefix_caching": False, "async_scheduling": config["backend"]["async_scheduling"],
                              "attention_backend": "FLASH_ATTN"})
    return snapshot


def create_model(method, config, backend, **kwargs):
    if method not in HISTORY_METHODS:
        module, cls, variant = "baselines.vanilla.model", "FullHistoryModel", "full"
        return getattr(import_module(module), cls)(variant=variant, backend=backend, **kwargs)
    module, cls, _, _ = HISTORY_METHODS[method]
    if method == "sliding_window":
        kwargs["kv_budget"] = config["cache"]["kv_budget"]
        if config["backend"]["name"] == "vllm":
            cls = "KVSlidingWindowModel"
    return getattr(import_module(module), cls)(variant=config["methods"][method]["variant"],
        policy=policy_for(method, config), backend=backend, **kwargs)
