"""Host-call and device-join lifecycle around the real runner entry point.

No exception or transport timeout constitutes a device-drain acknowledgement.
Only a successful backend drain allows a failed generation to become terminal.
"""
from functools import wraps
from .contract import require


def guarded_execute(runner, execute):
    @wraps(execute)
    def invoke(output, *args, **kwargs):
        from .native_probe import active_probe
        if active_probe(runner) is not None:
            return execute(output, *args, **kwargs)
        owner = runner._remote_ra_live_v2_allocation_owner
        bridge = runner._remote_ra_live_v2_bridge
        scheduled = output.num_scheduled_tokens
        require(not any(rid in owner.terminals for rid in scheduled),
                'delayed native execute for terminal generation')
        owned = tuple(rid for rid in scheduled if rid in owner.records)
        if not owned:
            return execute(output, *args, **kwargs)
        # Batched decode: every scheduled request must be an owned generation.
        require(len(owned) == len(scheduled),
                'generation guard requires owned requests only')
        identities = tuple(owner.records[rid]['identity'] for rid in owned)
        bridge.qwen.enter_execute(*identities)
        try:
            if owner.startup.get('source_closure_verified'):
                from .connector_guard import empty_scheduler_metadata
                empty_scheduler_metadata(output)
                owner.io.drain()
            result = execute(output, *args, **kwargs)
            bridge.qwen.leave_execute(*identities)
        except BaseException as original:
            # Original call has returned, but queued device work may remain.
            bridge.qwen.leave_failed_execute(*identities)
            try:
                bridge.qwen.recover_failed_execute(*identities)
            except BaseException as drain_error:
                raise ExceptionGroup('generation failed; device completion unknown; leases retained',
                                     [original, drain_error]) from original
            raise
        else:
            return result
    return invoke
