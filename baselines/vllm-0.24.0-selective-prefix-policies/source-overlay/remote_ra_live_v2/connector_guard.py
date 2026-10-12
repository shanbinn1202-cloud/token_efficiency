"""Captured OffloadingConnector quiescence, not wait_for_save (a no-op)."""
from .contract import require

def quiescent(connector):
    require(type(connector).__name__=='OffloadingConnector','unsupported loaded connector')
    worker=connector.connector_worker
    require(worker is not None,'offloading worker missing')
    require(not worker._load_jobs and not worker._unsubmitted_store_jobs,
            'connector load/deferred-store ownership remains: retain leases')
    handlers=tuple(worker.worker.handlers)
    require(handlers,'offloading handlers missing')
    for handler in handlers:
        require(type(handler).__name__=='SingleDirectionOffloadingHandler',
                'unknown connector handler completion semantics')
        require(not handler._transfers and not handler._transfer_events,
                'connector CUDA transfer completion unknown: retain leases')
    meta=getattr(connector,'_connector_metadata',None)
    if meta is not None:
        require(not meta.load_jobs and not meta.store_jobs and not meta.jobs_to_flush,
                'active connector metadata outside fixed quiescent fixture')
    require(not worker._connector_worker_meta.completed_jobs,
            'connector completion still awaits scheduler acknowledgement')
    return dict(loads=0,deferred_stores=0,transfer_events=0,pending_scheduler_acks=0)

def empty_scheduler_metadata(output):
    meta=getattr(output,'kv_connector_metadata',None)
    require(meta is not None and not meta.load_jobs and not meta.store_jobs and not meta.jobs_to_flush,
            'qualification requires empty native connector jobs')
