"""Default-OFF pre-B0 engine and trusted-frontend attachment."""
from __future__ import annotations

from .contract import require
from .readiness import NativeReadiness, REQUIRED, candidate_evidence


def attach_engine_dormant(core, ticket):
    """Attach code seams only; never makes an engine live."""
    if ticket is None:
        return None
    owner = core._remote_ra_growing_owner
    owner.owner_check(); owner.verify_identity()
    require(ticket is core._remote_ra_startup_ticket and ticket.phase == "active"
            and not hasattr(core, "output_thread") and
            not getattr(core, "engines_running", False),
            "live-v2 attachment must be canonical pre-B0")
    require(not hasattr(core, "_remote_ra_live_v2"),
            "live-v2 attachment collision")
    for name in ("remote_ra_live_v2.contract", "remote_ra_live_v2.lifecycle",
                 "remote_ra_live_v2.qwen", "remote_ra_live_v2.worker_backend",
                 "remote_ra_live_v2.controller", "remote_ra_live_v2.readiness"):
        # spawn does not inherit frontend imports. Import this fixed,
        # reviewed module before verifying its actual loaded identity.
        import importlib
        importlib.import_module(name)
        owner.loaded.verify_module(name)
    # Presence of audited code is not CUDA qualification or a successful smoke.
    state = dict(readiness=NativeReadiness(candidate_evidence(),
                                           cuda_qualified=False,
                                           growing_smoke_qualified=False),
                 qwen=None, lifecycle=None, worker=None)
    core._remote_ra_live_v2 = state
    return state


def bind_loaded_worker(core, *, worker_backend, qwen_lifecycle,
                       engine_lifecycle, cuda_evidence):
    """Called by the loaded GPU worker path after concrete layout checks."""
    state = core._remote_ra_live_v2
    owner = core._remote_ra_growing_owner
    owner.owner_check(); owner.verify_identity()
    require(state["worker"] is None and
            getattr(worker_backend.io, "evidence_kind", None) ==
            "CUDA_PHYSICAL_TENSOR_OPERATIONS" and
            type(cuda_evidence) is dict and
            cuda_evidence.get("copies_drained") is True and
            cuda_evidence.get("loaded_sources_verified") is True,
            "concrete loaded CUDA evidence required")
    from remote_ra_http.evidence import AttachedReadiness, worker_attached
    import os
    report = (worker_attached(core, worker_backend, qwen_lifecycle, engine_lifecycle, cuda_evidence)
              if os.environ.get("REMOTE_RA_HTTP_RUNTIME", "0") == "1" else None)
    state.update(worker=worker_backend, qwen=qwen_lifecycle, lifecycle=engine_lifecycle,
                 readiness=AttachedReadiness(report) if report is not None else NativeReadiness(candidate_evidence()))
    return state["readiness"].report()


def mark_growing_smoke(core, receipt):
    state = core._remote_ra_live_v2
    state["readiness"].require_native()
    require(type(receipt) is dict and receipt.get("turns", 0) >= 3 and
            receipt.get("reactivated_old_page") is True and
            receipt.get("retained_intersection") is True and
            receipt.get("cancel_drain_pass") is True,
            "growing-history smoke evidence incomplete")
    state["readiness"] = NativeReadiness({name: True for name in REQUIRED},
        cuda_qualified=True, growing_smoke_qualified=True)
    return state["readiness"].report()

