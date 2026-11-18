"""Explicit adapter for future audited EngineCore hooks, never a v5 bypass."""
from .controller import Abort, Capabilities, Receipt, REQUIRED_HOOKS, require


class EngineCoreBridge:
    """Construct only inside the private authenticated engine authority.

    These methods do not exist in frozen v5. Missing hooks are fatal at creation.
    Hooks must execute synchronously on the scheduler thread; their native
    implementation owns block tables, DRAM, hybrid state and CUDA completion.
    No HTTP dictionary can serve as the core or supply these attestations.
    """

    METHODS = ("remote_ra_growing_capabilities", "remote_ra_growing_apply",
               "remote_ra_growing_complete", "remote_ra_growing_rollback")

    def __init__(self, core):
        require(all(callable(getattr(core, name, None)) for name in self.METHODS),
                "EngineCore lacks audited growing-history native hooks")
        self._core = core
        self.capabilities()

    def capabilities(self):
        result = self._core.remote_ra_growing_capabilities()
        require(type(result) is Capabilities and REQUIRED_HOOKS <= result.hooks,
                "native hook capabilities incomplete")
        return result

    def apply(self, binding, plan, pages):
        return self._core.remote_ra_growing_apply(binding, plan, pages)

    def complete(self, binding, plan):
        return self._core.remote_ra_growing_complete(binding, plan)

    def rollback(self, binding, plan):
        return self._core.remote_ra_growing_rollback(binding, plan)
