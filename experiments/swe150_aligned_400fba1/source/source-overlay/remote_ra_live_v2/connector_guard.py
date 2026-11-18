"""Captured OffloadingConnector quiescence, not wait_for_save (a no-op).

Also the connector's CPU pool size (serve.py --connector-pool-gib), checked once at
startup. vLLM's CPUOffloadingSpec turns kv_connector_extra_config.cpu_bytes_to_use into
``cpu_bytes // block bytes`` offloaded blocks (vllm/v1/kv_offload/cpu/spec.py) and the
worker pins and zeroes that many blocks per KV tensor at registration
(vllm/v1/kv_offload/cpu/gpu_worker.py, CpuGpuOffloadingHandlers). On the HTTP runtime
nothing ever moves through that pool: isolate_connector (remote_ra_http/connector.py)
answers every prefix lookup with 0 tokens, refuses any external transfer, finishes
requests without a store and builds only empty connector metadata, and ``quiescent``
below holds the worker to no load, store or transfer. Captures and restores use the
runtime's own pinned memory (the Page arena or per-capture buffers and the GDN carries;
remote_ra_growing_worker/tensors.py). The pool only has to exist, with at least one block.

Without the option (the default) startup.json has no ``connector_pool_bytes`` and vLLM
gets 32 GiB, as before; nothing here runs. With it, ``connector_pool_bytes`` is pinned
(N GiB, 1 <= N <= 31) and vLLM gets cpu_bytes_to_use = N GiB; remote_ra_http.config.load
checks the value, compose_engine checks the engine's vLLM config and attach_loaded_worker
the worker's allocated pool, once each. Nothing runs per request or step.
"""
import os

from .contract import require

POOL_KEY = 'connector_pool_bytes'    # startup.json, only with --connector-pool-gib
DEFAULT_POOL_BYTES = 32 << 30        # vLLM's cpu_bytes_to_use without the option (serve.py CONNECTOR_BYTES)
POOL_GIB = range(1, 32)              # the option's whole-GiB choices, below the default


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


def pinned_pool(config):
    """The pinned connector pool in bytes (--connector-pool-gib), or None (the default 32 GiB: no key)."""
    if config is None or POOL_KEY not in config:
        return None
    value = config[POOL_KEY]
    require(type(value) is int and value > 0 and value % (1 << 30) == 0 and value >> 30 in POOL_GIB,
            'connector_pool_bytes must be 1-31 GiB in whole GiB; the default 32 GiB is pinned by leaving it out')
    return value


def check_engine_pool(vllm_config, config):
    """compose_engine (HTTP runtime), once: the engine's OffloadingConnector gets the pinned pool."""
    pin = pinned_pool(config)
    if pin is None:
        return None
    transfer = getattr(vllm_config, 'kv_transfer_config', None)
    require(getattr(transfer, 'kv_connector', None) == 'OffloadingConnector',
            'connector_pool_bytes needs the OffloadingConnector')
    actual = (getattr(transfer, 'kv_connector_extra_config', None) or {}).get('cpu_bytes_to_use')
    require(type(actual) is int and actual == pin,
            'engine connector pool %r differs from the pinned %d bytes' % (actual, pin))
    return pin


def check_worker_pool(connector):
    """attach_loaded_worker (HTTP runtime), once: the loaded connector allocated the pinned pool,
    at least one block per KV tensor and at most the pinned bytes. None without the pin."""
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    pin = pinned_pool(load())
    if pin is None:
        return None
    require(type(connector).__name__ == 'OffloadingConnector', 'unsupported loaded connector')
    worker = connector.connector_worker
    require(worker is not None, 'offloading worker missing')
    spec = worker.spec
    require(spec.extra_config.get('cpu_bytes_to_use') == pin,
            'worker connector pool %r differs from the pinned %d bytes' % (spec.extra_config.get('cpu_bytes_to_use'), pin))
    handlers = getattr(spec, '_handlers', None)
    require(handlers is not None, 'connector CPU pool not allocated')
    tensors = tuple(handlers.gpu_to_cpu_handler.dst_tensors)
    blocks = spec.num_blocks
    size = sum(t.numel() * t.element_size() for t in tensors)
    require(tensors and type(blocks) is int and blocks >= 1 and all(t.shape[0] == blocks for t in tensors),
            'connector CPU pool must hold at least one block per KV tensor')
    require(size <= pin, 'connector CPU pool pins %d bytes, more than the pinned %d' % (size, pin))
    return dict(pool_bytes=pin, blocks=blocks, allocated_bytes=size)
