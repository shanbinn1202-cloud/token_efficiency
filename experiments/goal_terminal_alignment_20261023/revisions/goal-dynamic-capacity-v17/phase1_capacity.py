"""Wait for actual long-Goal KV before touching this turn's native state."""
from types import SimpleNamespace

from .contract import require
from .goal_async import LeaseWait


def prepared_only(record):
    return (record.get('phase1') is None and 'plan' not in record
            and 'native_rows' not in record and 'native_interval' not in record)


class CapacityLeaseWait(LeaseWait):
    def __init__(self, coordinator, binding, blocks):
        super().__init__(coordinator, binding.request)
        self.binding, self.blocks = binding, blocks

    def figures(self):
        from .capacity import headroom_by_pool
        c = self.coordinator
        lifecycle = c.core._remote_ra_live_v2_lifecycle
        s = c.adapter.sessions[self.binding.session]
        restore = len(s.evicted) * len(c.groups()[0])
        headroom = headroom_by_pool(lifecycle)
        need = (self.blocks[0] + restore + headroom[0],
                self.blocks[1] + headroom[1])
        free = (c.pool.get_num_free_blocks(), c.mamba_pool.get_num_free_blocks())
        return lifecycle, need, free

    def done(self):
        if self.coordinator.records:
            return False
        lifecycle, need, free = self.figures()
        if need[1] > free[1]:
            return False
        if need[0] <= free[0]:
            return True
        release = getattr(self.coordinator.core, '_remote_ra_idle_release', None)
        if release is None:
            return False
        # Pure readiness check. Worker RPCs happen only after the engine resumes
        # this generator at its normal quiescent registered-command boundary.
        available = sum(len(s.resident) * release.groups()
                        for _, s in release.idle(lifecycle, self.binding.session))
        return need[0] <= free[0] + available


def wait_capacity(coordinator, binding, geo, goal_payload):
    blocks = coordinator.capacity_need(binding, geo, goal_payload)
    work = CapacityLeaseWait(coordinator, binding, blocks)
    core, adapter = coordinator.core, coordinator.adapter
    session = adapter.sessions[binding.session]
    release = getattr(core, '_remote_ra_idle_release', None)
    groups = len(coordinator.groups()[0])
    # Tails have no committed full-page mirror and are never released.
    floor = sum(groups * ((s.tail is not None) +
                         (len(s.resident) if s.poisoned else 0))
                for sid, s in adapter.sessions.items() if sid != binding.session)
    own = groups * (len(session.resident) + len(session.evicted) +
                    (session.tail is not None))
    from .idle_release import CapacityExceeded
    if own + blocks[0] > coordinator.pool.num_gpu_blocks - 1 - floor:
        raise CapacityExceeded('Actual long-Goal phase1 exceeds available full-attention pool')
    if blocks[1] > coordinator.mamba_pool.num_gpu_blocks - 1:
        raise CapacityExceeded('Actual phase1 exceeds available sliding-window pool')
    while True:
        lifecycle, need, free = work.figures()
        if not coordinator.records and all(n <= f for n, f in zip(need, free)):
            return
        # No phase1/reserve/native row has run for this request. Drop its
        # committed GPU mirror while parked so several large arrivals cannot
        # deadlock by each retaining history that another needs room to read.
        if release is not None and session.resident:
            require(not session.pending and binding.request not in adapter.requests
                    and not coordinator.pending(binding.session),
                    'capacity wait may release only an unstarted turn')
            release.evict(binding.session, binding.request, max(0, need[0] - free[0]))
            continue
        if not coordinator.records and release is not None and need[1] <= free[1]:
            command = SimpleNamespace(session_id=binding.session, request_id=binding.request)
            short = max(0, need[0] - free[0])
            victims = release.victims(lifecycle, command, short) if short else None
            if victims:
                release.release(victims, command, short)
                continue
        yield work
