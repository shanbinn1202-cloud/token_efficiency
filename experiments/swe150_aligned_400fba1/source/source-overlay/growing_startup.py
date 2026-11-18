"""Private, dormant attachment before B0. No environment/client enabling API."""
from remote_ra_native_growing.adapter import EngineCoreAdapter, NativePort
from remote_ra_native_growing.contract import Authority, require


def attach_dormant(core, ticket):
    # Normal vLLM construction never imports native authority or changes state.
    if ticket is None:
        return None
    from remote_ra_linux.native_authority import current, NativeEngine
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    owner = current()
    require(type(owner) is NativeEngine and core._remote_ra_authority is owner
            and owner.core is core, 'canonical engine-owned authority required')
    owner.owner_check(); owner.verify_identity()
    import remote_ra_live_v2.kv_layout  # noqa: F401 -- verified below like the rest
    for name in ('growing_startup', 'remote_ra_native_growing.adapter',
                 'remote_ra_native_growing.contract', 'remote_ra_native_growing.ownership',
                 'remote_ra_live_v2.kv_layout', 'vllm.v1.core.single_type_kv_cache_manager'):
        owner.loaded.verify_module(name)
    require(ticket.core is core and ticket.phase == 'active'
            and core._remote_ra_startup_ticket is ticket,
            'growing attachment requires pre-B0 startup ticket')
    require(not hasattr(core, 'output_thread') and not getattr(core, 'engines_running', False),
            'growing attachment after engine serving')
    scheduler = core.scheduler
    require(all(len(getattr(scheduler, n)) == 0 for n in ('requests','waiting','running')),
            'growing attachment requires empty scheduler')
    manager = scheduler.kv_cache_manager
    offload = getattr(getattr(scheduler, 'connector', None), 'connector_scheduler', None)
    require(offload is not None, 'native offloading scheduler required')
    targets = (core, scheduler, manager, offload)
    require(all(not hasattr(t, '_remote_ra_growing_adapter') for t in targets),
            'growing attachment collision')
    managers = tuple(m for m in manager.coordinator.single_type_managers
                     if type(m) is FullAttentionManager)
    authority = Authority(epoch=owner.identity.epoch, loaded_source=owner.loaded.manifest_hash)
    # The startup KV layout (interface I0), built once: the pinned startup config
    # and vLLM's additional_config must name the same layout (absent: legacy).
    from remote_ra_live_v2.contract import startup_layout
    from remote_ra_live_v2.kv_layout import from_vllm_config, legacy
    geometry = owner.smoke_config
    layout = startup_layout(geometry) or legacy()
    require(layout == from_vllm_config(core.vllm_config),
            'startup KV layout differs from the vLLM configuration')
    adapter = EngineCoreAdapter(authority, manager,
        page_tokens=geometry['page_tokens'], full_attention_managers=managers, layout=layout)
    require(adapter.capabilities()['native_ready'] is False, 'unexpected live backend')
    for target in targets:
        target._remote_ra_growing_adapter = adapter
    core._remote_ra_growing_port = NativePort(adapter)
    core._remote_ra_growing_owner = owner
    return adapter


def negotiate(core, nonce):
    """In-process authenticated owner seam, not a public route or receipt parser.

    Future transport must verify this MAC inside the same private authority and
    bind it to its own challenge. This candidate exposes no such transport.
    """
    from remote_ra_linux.native_authority import current, NativeEngine
    owner = current()
    require(type(owner) is NativeEngine and owner is core._remote_ra_growing_owner
            and owner.core is core, 'canonical capability owner required')
    owner.owner_check(); owner.verify_identity()
    require(type(nonce) is str and len(nonce) == 64 and
            all(c in '0123456789abcdef' for c in nonce), 'challenge must be 32-byte hex')
    adapter = core._remote_ra_growing_adapter
    result = dict(adapter.capabilities(), challenge=nonce, enabled=False)
    return {'capability': result, 'mac': adapter.authority.sign(result)}


def require_live(core, *args, **kwargs):
    # A bool, env flag, HTTP option or even a valid capability MAC cannot enable it.
    raise RuntimeError('NATIVE_BACKEND_UNAVAILABLE: compiled mixed-table/private DRAM/'
                       'hybrid-Mamba/DMA/signed-drain worker is absent')
