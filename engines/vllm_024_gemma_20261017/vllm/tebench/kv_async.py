"""Storage dependencies shared by policies using native AsyncScheduler.

No request priorities, capacity reservations, or admission limits live here.
Only batches which change the physical address space need a commit fence.
"""

import os


def needs_sample_values():
    return any(
        os.environ.get(k) == "1"
        for k in (
            "VLLM_AGENTKV",
            "VLLM_REPAIRKV",
            "VLLM_NATIVE_WINDOW",
            "VLLM_RANDOM_ATTENTION",
            "VLLM_COMPACT_WINDOW",
            "VLLM_INTENTKV",
        )
    )


def needs_commit(scheduler, output):
    from vllm.tebench.kv_recompute import BOUNDARIES, CURSORS
    from vllm.tebench.kv_session import OFFSETS
    from vllm.tebench.repairkv import REPLAY_BOUNDARIES

    agent = os.environ.get("VLLM_AGENTKV") == "1"
    intent = os.environ.get("VLLM_INTENTKV") == "1"
    window = os.environ.get("VLLM_COMPACT_WINDOW") == "1"
    random = os.environ.get("VLLM_RANDOM_ATTENTION") == "1" or window
    if not (agent or intent or random or os.environ.get("VLLM_REPAIRKV") == "1"):
        return False
    if agent:
        import json

        interval = json.loads(os.environ["VLLM_AGENTKV_OPTIONS"]).get(
            "eviction_interval", 128
        )
        budget = int(os.environ["VLLM_AGENTKV_BUDGET"])
    for rid, count in output.num_scheduled_tokens.items():
        req = scheduler.requests[rid]
        end = req.num_computed_tokens + count
        cursor = CURSORS.get(rid)
        if cursor is not None and end == BOUNDARIES[rid][cursor]:
            return True
        if end == REPLAY_BOUNDARIES.get(rid):
            return True
        if (
            intent
            and end == req.num_prompt_tokens
            and end - OFFSETS.get(rid, 0) >= int(os.environ["VLLM_INTENTKV_BUDGET"])
        ):
            return True
        if random:
            if window:
                from adapters.sliding_window.policy import should_compact
            else:
                from adapters.random_attention.policy import should_compact

            if (not window or end >= req.num_prompt_tokens) and should_compact(
                physical=end - OFFSETS.get(rid, 0),
                budget=int(
                    os.environ[
                        "VLLM_COMPACT_WINDOW_BUDGET"
                        if window
                        else "VLLM_RANDOM_ATTENTION_BUDGET"
                    ]
                ),
                new_tokens=count,
                decoding=end > req.num_prompt_tokens,
            ):
                return True
        generated = end - req.num_prompt_tokens + 1
        if (
            agent
            and generated > 0
            and generated % interval == 0
            and end - OFFSETS.get(rid, 0) > budget
        ):
            return True
    return False


def stopped(runner, request):
    from types import SimpleNamespace

    from vllm.v1.core.sched.utils import check_stop

    view = SimpleNamespace(
        sampling_params=request.sampling_params,
        pooling_params=None,
        num_output_tokens=len(request.output_token_ids),
        output_token_ids=request.output_token_ids,
        num_tokens=request.num_tokens,
        max_tokens=request.sampling_params.max_tokens,
    )
    return check_stop(view, runner.max_model_len)
