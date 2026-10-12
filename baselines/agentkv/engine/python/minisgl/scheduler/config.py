from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    prefill_chunk_size: int = 0
    offline_mode: bool = False
    eviction_plan: str = "identity"
    compact_interval: int = 128
    compact_after_prefill: bool = False
    compact_before_park: bool = False
    compact_aware_kv_reserve: bool = False
    compact_aware_prefill_reserve: bool = False
    compact_debug: bool = False
    compact_debug_dir: str = ""
    compact_debug_jsonl: bool = False
    compact_debug_max_items: int = 0
    eviction_trace: bool = False
    eviction_trace_dir: str = ""
    query_ring_capacity: int = 0
    query_ring_num_stages: int = 1

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    def __post_init__(self) -> None:
        if self.query_ring_capacity != 0:
            return
        from minisgl.eviction import create_eviction_plan

        plan = create_eviction_plan(self.eviction_plan)
        if plan.requires_query_observation:
            object.__setattr__(
                self,
                "query_ring_capacity",
                getattr(plan, "query_ring_capacity", getattr(plan, "observation_window")),
            )
            object.__setattr__(
                self,
                "query_ring_num_stages",
                getattr(plan, "query_ring_num_stages", 1),
            )

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
