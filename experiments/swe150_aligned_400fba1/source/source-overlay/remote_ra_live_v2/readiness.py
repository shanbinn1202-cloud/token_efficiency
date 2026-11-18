"""Capability is a conjunction of concrete installed/native evidence only."""
from __future__ import annotations

from dataclasses import dataclass
from .contract import require


REQUIRED = (
    "source_identity",
    "private_transport_v2",
    "engine_lifecycle_callbacks",
    "worker_backend",
    "qwen_align_forward",
    "suffix_decode_leases",
    "mamba_relocation",
    "page_query_publication",
    "cancel_drain",
    "authenticated_growing_query_selection",
    "native_sparse_position_mapping",
)


def candidate_evidence():
    return {name: name not in {"authenticated_growing_query_selection",
                               "native_sparse_position_mapping"} for name in REQUIRED}


@dataclass(frozen=True)
class NativeReadiness:
    evidence: dict[str, bool]
    default_enabled: bool = False
    cuda_qualified: bool = False
    growing_smoke_qualified: bool = False

    def __post_init__(self):
        require(type(self.evidence) is dict and
                set(self.evidence) == set(REQUIRED) and
                all(type(v) is bool for v in self.evidence.values()),
                "exact readiness evidence required")
        require(self.default_enabled is False,
                "growing history must remain default OFF")

    @property
    def code_ready(self):
        return all(self.evidence.values())

    @property
    def native_ready(self):
        return self.code_ready and self.cuda_qualified

    @property
    def live_multi_turn_supported(self):
        return self.native_ready and self.growing_smoke_qualified

    def require_native(self):
        missing = [name for name in REQUIRED if not self.evidence[name]]
        if not self.cuda_qualified:
            missing.append("cuda_qualification")
        require(not missing, "native readiness missing: " + ", ".join(missing))

    def report(self):
        return dict(default_enabled=False, code_ready=self.code_ready,
                    native_ready=self.native_ready,
                    live_multi_turn_supported=self.live_multi_turn_supported,
                    cuda_qualified=self.cuda_qualified,
                    growing_smoke_qualified=self.growing_smoke_qualified,
                    missing=[name for name in REQUIRED if not self.evidence[name]])


@dataclass(frozen=True)
class FrontendReadiness:
    """Read-only parent view of the worker-qualified B0 readiness receipt."""
    report_value: dict

    def __post_init__(self):
        require(type(self.report_value) is dict,
                "frontend readiness receipt required")
        require(self.report_value.get("default_enabled") is False,
                "growing history must remain default OFF")

    def require_native(self):
        require(self.report_value.get("native_ready") is True,
                "frontend requires worker-qualified native readiness")

    def report(self):
        return dict(self.report_value)
