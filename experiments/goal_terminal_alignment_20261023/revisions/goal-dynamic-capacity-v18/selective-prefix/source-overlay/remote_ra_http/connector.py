"""Isolate native growing ownership from the legacy async offloader."""
from types import MethodType
from .config import require

def isolate_connector(core):
    adapter=core._remote_ra_growing_adapter
    connector=core.scheduler.connector.connector_scheduler
    require(not connector._req_status and not connector.has_pending_push_work(),
            'connector must be fresh before growing isolation')
    native={n:getattr(connector,n) for n in ('on_new_request','get_num_new_matched_tokens',
            'update_state_after_alloc','request_finished','build_connector_meta')}
    admitted=set()
    def owns(request):
        return request.request_id in admitted or adapter.owns(request.request_id)
    def on_new(self,request):
        require(adapter.owns(request.request_id),'HTTP-only server refuses unowned connector request')
        admitted.add(request.request_id)
    def matched(self,request,num_computed_tokens):
        require(owns(request),'unowned prefix lookup');return 0,False
    def updated(self,request,blocks,num_external_tokens):
        require(owns(request) and num_external_tokens==0,'external offload transfer forbidden')
    def finished(self,request):
        require(owns(request),'unowned connector completion');return False,None
    def meta(self,output):
        require(not output.preempted_req_ids,'preemption unsupported for exclusive growing session')
        require(set(output.num_scheduled_tokens)<=admitted,'unowned scheduled request')
        require(not self.has_pending_push_work(),'legacy offloader work escaped isolation')
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingConnectorMetadata
        return OffloadingConnectorMetadata(load_jobs={},store_jobs={},jobs_to_flush=set(),selective_load_plans={})
    for name,fn in zip(native,(on_new,matched,updated,finished,meta)):
        setattr(connector,name,MethodType(fn,connector))
    connector._remote_ra_http_native_methods=native
