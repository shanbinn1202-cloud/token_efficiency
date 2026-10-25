"""Pre-B0 dormant attachment using the existing two-process authority channel."""
import hashlib
from .transport import Channel, require
from .lifecycle import MISSING


def channels(owner, directions):
    owner.owner_check(); owner.verify_identity()
    require(owner.channel is not None, 'existing private startup channel required')
    # Domain separation from v1 registration; never supplied by a request/env flag.
    key = hashlib.sha256(owner.channel._key + b'remote-ra-private-growing-v2').digest()
    return tuple(Channel(key, owner.identity.epoch, owner.loaded.manifest_hash, d) for d in directions)


def attach_dormant(core, ticket):
    if ticket is None:
        return None
    from remote_ra_linux.native_authority import current, NativeEngine
    owner = current()
    require(type(owner) is NativeEngine and owner is core._remote_ra_growing_owner
            and ticket is core._remote_ra_startup_ticket and ticket.phase == 'active',
            'canonical pre-B0 startup authority required')
    require(not hasattr(core, 'output_thread') and not getattr(core, 'engines_running', False),
            'private v2 attachment must precede B0 and IO threads')
    executor = core.model_executor
    require(type(executor).__module__ == 'vllm.v1.executor.uniproc_executor'
            and type(executor).__name__ == 'UniProcExecutor', 'UniProc-only private worker')
    require(not hasattr(core, '_remote_ra_private_v2'), 'private v2 attachment collision')
    for name in ('remote_ra_private_v2.startup', 'remote_ra_private_v2.transport',
                 'remote_ra_private_v2.lifecycle', 'remote_ra_private_v2.frontend',
                 'remote_ra_private_v2.worker'):
        owner.loaded.verify_module(name)
    incoming, outgoing = channels(owner, ('frontend-engine', 'engine-frontend'))
    state = dict(incoming=incoming, outgoing=outgoing, native_ready=False,
                 missing=tuple(MISSING), dispatcher=None)
    # One assignment after all validation: failed attachment leaves no partial state.
    core._remote_ra_private_v2 = state
    return state


def dispatch(core, frame):
    state = core._remote_ra_private_v2
    dispatcher = state['dispatcher']
    require(dispatcher is not None, 'PRIVATE_V2_ACTIVE_COMPOSITION_UNAVAILABLE')
    return dispatcher.handle(frame)


def detach(core):
    state = getattr(core, '_remote_ra_private_v2', None)
    if state is None:
        return
    if state['dispatcher'] is not None:
        state['dispatcher'].disconnect()
    del core._remote_ra_private_v2
