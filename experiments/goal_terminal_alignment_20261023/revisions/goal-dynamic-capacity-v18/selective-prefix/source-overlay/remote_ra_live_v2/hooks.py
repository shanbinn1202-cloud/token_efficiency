"""Tiny installed vLLM hook surface; inert when the private bridge is absent."""
from .contract import require


def scheduler_split(scheduler, request, cursor, proposed):
    bridge = getattr(scheduler, "_remote_ra_live_v2_bridge", None)
    if bridge is None:
        return proposed
    return bridge.scheduler_split(request, cursor, proposed)


def before_forward(runner, scheduler_output, positions):
    from .native_probe import active_probe
    probe=active_probe(runner)
    if probe is not None:return probe.before_forward(positions)
    bridge = getattr(runner, "_remote_ra_live_v2_bridge", None)
    if bridge is None:
        return None
    from .positions import runtime_mode
    runtime_mode(runner,'generation forward')
    # Several requests may share the step; the batch holds exactly the scheduled ones.
    require(len(runner.input_batch.req_ids)==len(scheduler_output.num_scheduled_tokens)>0 and
            set(runner.input_batch.req_ids)==set(scheduler_output.num_scheduled_tokens),
            'generation forward: batch differs from the scheduled requests')
    from .native_probe import mixed_probe
    mixed = mixed_probe(runner)
    if mixed is None:
        return bridge.before_forward(runner, scheduler_output, positions)
    leases = bridge.before_forward(runner, scheduler_output, positions)
    # --phase1-in-step: the armed probe row of this step (phase1_step).
    from .phase1_step import probe_before_forward
    probe_before_forward(runner, mixed, scheduler_output, positions)
    return leases


def after_forward(runner, scheduler_output, hidden_states):
    from .native_probe import active_probe
    from .vector_space import pre_final_norm
    probe=active_probe(runner)
    # Vector consumers get pre-final-norm rows; logits keep the native output.
    if probe is not None:return probe.after_forward(pre_final_norm(runner, hidden_states))
    bridge = getattr(runner, "_remote_ra_live_v2_bridge", None)
    if bridge is None:
        return None
    from .native_probe import mixed_probe
    mixed = mixed_probe(runner)
    if mixed is None:
        return bridge.after_forward(runner, scheduler_output, pre_final_norm(runner, hidden_states))
    # --phase1-in-step: the generation rows, then the armed probe row (phase1_step).
    hidden = pre_final_norm(runner, hidden_states)
    receipts = bridge.after_forward(runner, scheduler_output, hidden)
    from .phase1_step import probe_after_forward
    probe_after_forward(runner, mixed, scheduler_output, hidden)
    return receipts


def force_eager(runner):
    """Runner hook (execute_model): True dispatches this step to CUDAGraphMode.NONE.

    Only with CUDA graphs on (cuda_graphs.attach) and only for a probe-only
    pass; False otherwise, which is vLLM's default."""
    from .cuda_graphs import force_eager as probe_pass
    return probe_pass(runner)


def output_stop(scheduler, request):
    """Protocol stop checked after vLLM's own EOS/stop-token/length rules."""
    policy = getattr(scheduler, "_remote_ra_output_stop", None)
    return False if policy is None else policy(request)


def terminal(runner, request_id, operation):
    bridge = getattr(runner, "_remote_ra_live_v2_bridge", None)
    if bridge is not None:
        return bridge.terminal(request_id, operation)

