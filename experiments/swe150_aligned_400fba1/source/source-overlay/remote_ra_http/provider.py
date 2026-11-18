"""No fixed messages: growing admission is owned by SplitPrivateEndpoint."""
from dynamic_runtime import DynamicContextProvider

class RequestContextProvider(DynamicContextProvider):
    def register(self,**kwargs):
        raise RuntimeError('HTTP requests must use authenticated growing registered-add')

def reject_legacy_selection(*args,**kwargs):
    raise RuntimeError('frozen three-stage selector is disabled in HTTP runtime')
