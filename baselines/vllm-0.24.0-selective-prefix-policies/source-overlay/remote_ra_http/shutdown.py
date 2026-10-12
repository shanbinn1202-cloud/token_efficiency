"""Cleanup-only owners: fence admission, join owned resources, persist, then ACK."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time

GRACE_SECONDS=35.0  # One budget inside the qualification controller's 45 s.

class CleanupOwner:
    def __init__(self):
        self.thread=threading.get_ident();self.state='open';self.result=None;self.error=None
    def run(self,action):
        if threading.get_ident()!=self.thread:raise RuntimeError('cleanup owner thread changed')
        if self.state=='done':return self.result
        if self.state=='failed':raise self.error
        if self.state=='closing':raise RuntimeError('recursive cleanup owner')
        self.state='closing'
        try:self.result=action()
        except BaseException as exc:self.error=exc;self.state='failed';raise
        self.state='done';return self.result

def once(owner,name,action):
    state=getattr(owner,name,None)
    if state is None:state=CleanupOwner();setattr(owner,name,state)
    return state.run(action)

def stages(label,actions):
    errors=[];results=[]
    for name,action in actions:
        if action is None:continue
        try:results.append(action())
        except BaseException as exc:
            exc.add_note('shutdown stage: '+name);errors.append(exc)
    if errors:raise BaseExceptionGroup(label,errors)
    return results

def save_receipt(path,value):
    from .config import encoded
    raw=encoded(value);path=Path(path)
    try:
        with path.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes()!=raw:raise RuntimeError('conflicting cleanup receipt')
    return hashlib.sha256(raw).hexdigest()

def request_close(authority):
    authority.owner_check()
    if getattr(authority,'closing',False):return
    authority.closing=True;authority.deadline=time.monotonic()+GRACE_SECONDS
    # Close admission while existing owned executes/drains may still finish.
    # The final cleanup owner revokes live capability before freeing anything.
    from vllm.v1.engine.core import EngineShutdownState
    authority.core.shutdown_state=EngineShutdownState.REQUESTED

def shutdown_engine(core,native):
    authority=getattr(core,'_remote_ra_authority',None)
    if authority is None and not getattr(core,'_remote_ra_dynamic_mode',False):return native()
    def perform():
        errors=[];receipt_hash=None
        ticket=getattr(core,'_remote_ra_startup_ticket',None)
        if ticket is not None:ticket.phase='closing'
        if authority is not None:
            authority.owner_check();authority.closing=True
            authority.deadline=time.monotonic()+GRACE_SECONDS
            authority.fence.revoke_and_cleanup()
        assembly=getattr(core,'_remote_ra_assembly',None)
        gate=getattr(authority,'gate',None)
        try:
            stages('owned engine cleanup failed',[
                ('request/session drain',assembly.close if assembly is not None else None),
                ('worker gate',gate.close if gate is not None else None),
                ('native resources',native)])
        except BaseException as exc:errors.append(exc)
        if not errors:
            try:
                from .evidence import engine_closed
                receipt_hash=engine_closed(core)
            except BaseException as exc:errors.append(exc)
        if authority is not None:
            if errors:
                from .diagnostics import failure
                failure('ENGINE_TEARDOWN_FAILED',BaseExceptionGroup('engine teardown failed',errors),authority)
            try:authority.close(receipt_hash=receipt_hash)
            except BaseException as exc:errors.append(exc)
        if ticket is not None:ticket.phase='failed_shutdown' if errors else 'closed'
        if errors:
            error=BaseExceptionGroup('engine closure failed',errors)
            from .diagnostics import failure
            failure('ENGINE_SHUTDOWN_FAILED',error,authority)
            raise error
        return receipt_hash
    return once(core,'_remote_ra_cleanup_owner',perform)

def close_authority(authority,receipt_hash=None):
    authority.owner_check()
    def perform():
        from private_authority.protocol import hex64
        authority.fence.revoke_and_cleanup();authority.stop.set()
        errors=[]
        try:
            if authority.gate is not None:authority.gate.close()
        except BaseException as exc:errors.append(exc)
        try:
            authority.observer.join(1)
            if authority.observer.is_alive():raise RuntimeError('engine observer did not stop')
        except BaseException as exc:errors.append(exc)
        try:
            if not errors and receipt_hash is not None:
                if not hex64(receipt_hash):raise RuntimeError('invalid cleanup receipt digest')
                authority.channel.send('CLOSED',{'cleanup_sha256':receipt_hash})
        except BaseException as exc:errors.append(exc)
        try:authority.channel.pipe.close()
        except BaseException as exc:errors.append(exc)
        authority.closed=not errors
        if errors:raise BaseExceptionGroup('engine authority close failed',errors)
        from remote_ra_linux.native_authority import event
        event('ENGINE_AUTHORITY_CLOSED',epoch=authority.identity.epoch,
              native_gpu_drain_attested=receipt_hash is not None,cleanup_sha256=receipt_hash)
        return receipt_hash
    return once(authority,'_cleanup_owner',perform)

def accept_closed(manager,payload):
    raw=Path(manager.cleanup_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=payload['cleanup_sha256']:
        raise RuntimeError('CLOSED cleanup digest mismatch')
    value=json.loads(raw)
    if (value.get('joined') is not True or value.get('epoch')!=manager.identity.epoch or
            value.get('session_id')!=manager.session_id or
            value.get('runtime_manifest')!=manager.identity.manifest):
        raise RuntimeError('CLOSED cleanup ownership mismatch')
    manager.closed_receipt=payload['cleanup_sha256']

def send_or_closing(channel,kind,payload):
    # A signal-triggered child may have queued CLOSED and closed its write end.
    # The sole reader must still authenticate that frame, never infer success
    # from EOF. The caller always performs the bounded authenticated receive.
    if channel.pipe.poll(0):return
    try:channel.send(kind,payload)
    except (OSError,EOFError):return

def manager_shutdown(manager):
    # May be entered by MPClient's monitor thread as well as the frontend owner.
    with manager.close_lock:
        state=getattr(manager,'close_state','open')
        if state=='done':return getattr(manager,'closed_receipt',None)
        if state=='failed':raise manager.close_error
        manager.close_state='closing';errors=[]
        deadline=time.monotonic()+GRACE_SECONDS
        manager.closing=True;manager.deadline=deadline
        try:
            if manager.reader is not None and manager.b0.is_set() and not manager.done.is_set():
                if not manager.revoked.is_set():manager.commands.put(('CLOSE',None),timeout=max(.01,deadline-time.monotonic()))
                if not manager.done.wait(max(0,deadline-time.monotonic())):
                    raise TimeoutError('private CLOSED deadline expired')
            if manager.error:raise RuntimeError(manager.error)
            if not getattr(manager,'closed_receipt',None):raise RuntimeError('private CLOSED receipt missing')
            manager.process.join(max(0,deadline-time.monotonic()))
            if manager.process.is_alive():raise TimeoutError('engine natural exit deadline expired')
            if manager.process.exitcode!=0:raise RuntimeError('engine exited abnormally: '+str(manager.process.exitcode))
        except BaseException as exc:
            errors.append(exc)
            try:manager._reap()
            except BaseException as cleanup:errors.append(cleanup)
        manager.revoked.set();manager.done.set()
        for pipe in (manager.pipe,manager.child_pipe):
            try:pipe.close()
            except BaseException as exc:errors.append(exc)
        for thread in (manager.reader,manager.watchdog):
            if thread is not None and thread is not threading.current_thread():
                thread.join(2)
                if thread.is_alive():errors.append(RuntimeError('authority thread stuck'))
        if not errors:
            try:
                # Multi-session stage 1: the parent receipt names every session the server held.
                from .config import load
                from remote_ra_live_v2.sessions import session_ids
                sessions=session_ids(manager.session_id,len(load()['tasks']))
                save_receipt(Path(manager.cleanup_path).with_name('parent-cleanup.json'),
                    dict(schema='remote-ra-parent-cleanup-v1',epoch=manager.identity.epoch,
                         session_id=manager.session_id,sessions=sessions,cleanup_sha256=manager.closed_receipt,
                         child_exitcode=manager.process.exitcode,natural_exit=True,joined=True))
            except BaseException as exc:errors.append(exc)
        if errors:
            manager.close_state='failed';manager.close_error=BaseExceptionGroup('parent closure failed',errors)
            raise manager.close_error
        manager.close_state='done';manager.closed=True
        import remote_ra_linux.native_authority as module
        if module._authority is manager:module._authority=None
        module.event('PARENT_AUTHORITY_CLOSED',native_gpu_drain_attested=True,
                     cleanup_sha256=manager.closed_receipt,natural_exit=True)
        return manager.closed_receipt
