"""One real manager/engine private authority for bounded native smoke.

No CPU engine entrypoint or synthetic capability is used. The engine owner
thread alone pumps channel messages and scheduler state; observers only revoke.
This module does not claim a successful GPU run or complete native attestation.
"""
from dataclasses import dataclass, asdict
import hashlib
import multiprocessing as mp
import os
import queue
import secrets
import threading
import time

from private_authority.protocol import (EngineFence, Topology, child_provision,
    parent_provision, capability_from_worker_gate, digest, hex64, require)

from remote_ra_http.diagnostics import failure, phase

_authority = None


def current():
    require(type(_authority) in (NativeManager, NativeEngine),
            'native authority is not installed in this process')
    _authority.owner_check()
    require(not _authority.revoked.is_set(), 'native authority revoked')
    return _authority


def install(authority):
    global _authority
    require(_authority is None and type(authority) in (NativeManager, NativeEngine),
            'one canonical native authority per process')
    _authority = authority


def event(name, **fields):
    import json
    print('REMOTE_RA_LIVE ' + json.dumps(dict(event=name, pid=os.getpid(), **fields),
                                      sort_keys=True), flush=True)


@dataclass(frozen=True)
class NativeEndpoint:
    pipe: object
    manifest: str
    policy: str


class Common:
    selected_count = 2
    projection_seed = 42

    def owner_check(self):
        require(threading.get_ident() == self.owner, 'native authority owner changed')

    def verify_identity(self):
        self.owner_check()
        require(not self.revoked.is_set(), 'native authority revoked')
        return self.loaded.verify_loaded()

    def revoke(self, reason):
        self.revoked.set()
        event('NATIVE_AUTHORITY_REVOKED', reason=str(reason)[:200])

    @property
    def smoke_config(self):
        from remote_ra_http.config import legacy_geometry
        return legacy_geometry()

    @property
    def site(self):
        return self.loaded.site

    @property
    def disk_pins(self):
        m = self.loaded.manifest
        return dict(sources=m['baseline_files'], native=m['native_pins'], model_pins=m['model_pins'])

    @property
    def release_source(self):
        return self.loaded.source

    @property
    def release_hashes(self):
        return self.loaded.pins

    @property
    def codec(self):
        from registered_add_protocol import ProtocolCodec
        require(self.channel is not None, 'private channel not provisioned')
        if self._codec is None:
            key = hashlib.sha256(self.channel._key + b'registered-add-v1').digest()
            self._codec = ProtocolCodec(key=key, identity=self.identity.manifest,
                                        epoch=self.identity.epoch)
        return self._codec

    @property
    def model_root(self):
        return self.loaded.model_root

    @property
    def session_id(self):
        return 'http-' + self.identity.epoch[:32]


class NativeManager(Common):
    """Transport owner attached to the actual CoreEngineProcManager."""
    def __init__(self, manager, context, config, local_engine_count):
        from vllm.v1.engine.utils import CoreEngineProcManager
        from bootstrap_contract import validate_topology
        from .live_identity import LiveIdentity
        require(type(manager) is CoreEngineProcManager and local_engine_count == 1,
                'actual single native manager required')
        Topology().validate(context); validate_topology(config)
        require(os.environ.get('VLLM_POLICY_RUNTIME_V1') == '1', 'worker hooks required')
        require(not os.environ.get('VLLM_POLICY_RUNTIME_MANIFEST'), 'static manifest conflict')
        from remote_ra_http.config import load
        self.policy = load()['policy']
        from pathlib import Path
        self.cleanup_path=Path(load()['output_directory'])/'engine-cleanup.json'
        self.closing=False;self.closed_receipt=None;self.close_state='open'
        require(self.policy in ('full', 'random50', 'goal_query', 'observation_query', 'random_fill'), 'explicit four-policy smoke required')
        self.loaded = LiveIdentity(); self.loaded.verify_module('vllm.v1.engine.utils')
        self.owner = threading.get_ident(); self.manager = manager; self.context = context
        self.revoked = threading.Event(); self.done = threading.Event()
        self.b0 = threading.Event(); self.f1 = threading.Event()
        self.commands = queue.Queue(maxsize=1); self.lock = threading.RLock()
        self.close_lock = threading.Lock()
        self.channel = self.identity = self.process = self.reader = self.watchdog = None
        self._codec = None; self.closed = False; self.taken = False
        self.live_v2_readiness = None
        self.error = None; self.started_at = time.monotonic()
        self.deadline = self.started_at + 600
        self.diagnostic_phase = "parent-created"
        self.pipe, self.child_pipe = context.Pipe(duplex=True)
        manager._remote_ra_authority = self
        install(self)

    def child_endpoint(self):
        self.owner_check(); require(not self.taken, 'endpoint is single use')
        self.taken = True
        return NativeEndpoint(self.child_pipe, self.loaded.manifest_hash, self.policy)

    @property
    def _process(self):
        return self.process

    @_process.setter
    def _process(self, value):
        self.process = value

    def after_spawn(self, process):
        self.owner_check()
        require(self.taken and self.process is None and process.pid is not None
                and process._parent_pid == os.getpid() and process._start_method == 'spawn',
                'manager must own its real spawned engine')
        self.process = process; self.child_pipe.close()
        self.reader = threading.Thread(target=self._run, name='native-private-authority')
        self.watchdog = threading.Thread(target=self._watch, name='native-authority-watchdog')
        self.reader.start(); self.watchdog.start()

    def _fail(self, why):
        self.error = self.error or why; self.revoked.set()

    def _watch(self):
        while not self.done.wait(.1):
            if not self.process.is_alive():
                # Let the sole reader consume a CLOSED frame already in the pipe.
                if self.done.wait(.5):return
                self._fail('engine peer exited: exitcode=' + str(self.process.exitcode))
            elif time.monotonic() >= self.deadline:
                self._fail('authority deadline expired in ' + self.diagnostic_phase)
            if self.revoked.is_set() and self.error:
                failure('PARENT_AUTHORITY_WATCHDOG', RuntimeError(self.error), self)
            if self.revoked.is_set():
                self._reap(); return

    def _run(self):
        try:
            phase(self, "parent-provision")
            channel = parent_provision(self.pipe, child_pid=self.process.pid,
                manifest=self.loaded.manifest_hash, deadline=self.deadline)
            self.channel = channel; self.identity = channel.identity
            phase(self, 'parent-await-prepared-live')
            _, prepared = channel.receive({'PREPARED_LIVE'}, self.deadline)
            live_report = prepared.get('live_v2_readiness')
            require(type(live_report) is dict and live_report.get('native_ready') is True
                    and live_report.get('default_enabled') is False,
                    'worker-qualified live v2 readiness required')
            self.live_v2_readiness = dict(live_report)
            phase(self, 'parent-commit')
            receipt = {'receipt_hash': digest(prepared)}
            channel.send('COMMIT', receipt)
            require(channel.receive({'COMMIT_ACK'}, self.deadline)[1] == receipt, 'commit ACK differs')
            phase(self, 'parent-release')
            channel.send('RELEASE', receipt)
            require(channel.receive({'RELEASE_ACK'}, self.deadline)[1] == receipt, 'release ACK differs')
            channel.handoff_to_monitor(); self.b0.set()
            event('B0_NATIVE_WORKER_ACK', epoch=self.identity.epoch,
                  worker_nonce=prepared['worker_nonce'], capability_only=True)
            self.deadline = time.monotonic() + 90
            f1_deadline = time.monotonic() + 120
            bound = False
            phase(self, "parent-monitor-await-f1")
            while not self.revoked.is_set():
                try: command, value = self.commands.get(timeout=.2)
                except queue.Empty: command, value = 'PING', None
                if command == 'CLOSE':
                    from remote_ra_http.shutdown import accept_closed, GRACE_SECONDS
                    self.closing=True;self.deadline=time.monotonic()+GRACE_SECONDS
                    from remote_ra_http.shutdown import send_or_closing
                    send_or_closing(channel,'CLOSE',{})
                    accept_closed(self,channel.receive({'CLOSED'},self.deadline)[1]);break
                if command == 'BIND':
                    require(not bound, 'frontend binding repeated')
                    payload = dict(frontend_nonce=secrets.token_hex(32), binding_hash=value,
                                   worker_nonce=prepared['worker_nonce'])
                    channel.send('FRONTEND_BOUND', payload)
                    require(channel.receive({'FRONTEND_ACK'}, self.deadline)[1] == payload,
                            'frontend ACK differs')
                    bound = True; self.f1.set()
                    phase(self, "parent-admitted")
                    event('F1_NATIVE_FRONTEND_ACK', epoch=self.identity.epoch, binding_hash=value,
                          session_id=self.session_id, policy=self.policy,
                          manifest=self.identity.manifest)
                else:
                    require(bound or time.monotonic() < f1_deadline, 'frontend binding expired')
                    payload = {'ping_nonce': secrets.token_hex(32)}
                    from remote_ra_http.shutdown import send_or_closing
                    send_or_closing(channel,'PING',payload)
                    kind,reply=channel.receive({'PONG','CLOSED'},self.deadline)
                    if kind=='CLOSED':
                        from remote_ra_http.shutdown import accept_closed
                        self.closing=True;accept_closed(self,reply);break
                    require(reply == payload, 'heartbeat differs')
                self.deadline = time.monotonic() + 90
        except BaseException as exc:
            self._fail(type(exc).__name__ + ': ' + str(exc))
            failure('PARENT_AUTHORITY_REFUSED', exc, self)
            self._reap()
        finally:
            self.revoked.set(); self.done.set()

    def _wait(self, ready):
        self.owner_check()
        while not ready.wait(.05):
            require(not self.revoked.is_set(), self.error or 'authority revoked')
        require(not self.revoked.is_set(), self.error or 'authority revoked')

    def verify_engine_handshake(self):
        self._wait(self.b0); return self.verify_identity()

    def bind_frontend(self, binding_hash):
        self.verify_engine_handshake()
        require(hex64(binding_hash) and not self.f1.is_set(), 'invalid/repeated frontend binding')
        self.commands.put_nowait(('BIND', binding_hash)); self._wait(self.f1)

    def verify_admission(self):
        self.verify_identity(); require(not getattr(self,'closing',False),'frontend admission closing')
        require(self.f1.is_set(), 'frontend ACK missing'); return True

    require_frontend_admission = verify_admission

    def _reap(self):
        with self.lock:
            if self.process is None or self.process.pid is None:return
            if self.process.is_alive():self.process.terminate()
            self.process.join(2)
            if self.process.is_alive():self.process.kill(); self.process.join(2)
            require(not self.process.is_alive(), 'owned engine did not stop')

    def close(self):
        self.owner_check()
        self.shutdown()

    def shutdown(self):
        from remote_ra_http.shutdown import manager_shutdown
        return manager_shutdown(self)


class NativeEngine(Common):
    def __init__(self, endpoint, config):
        from .live_identity import LiveIdentity
        from bootstrap_contract import validate_topology
        require(type(endpoint) is NativeEndpoint, 'native private endpoint required')
        parent = mp.parent_process()
        require(parent is not None and mp.get_start_method() == 'spawn', 'real spawn parent required')
        validate_topology(config)
        self.loaded = LiveIdentity(); require(endpoint.manifest == self.loaded.manifest_hash, 'launch manifest differs')
        self.owner = threading.get_ident(); self.revoked = threading.Event(); self.stop = threading.Event()
        self.policy = endpoint.policy; self._codec = None; self.core = self.gate = self.assembly = None
        self.started_at = time.monotonic()
        self.deadline = self.started_at + 600
        self.diagnostic_phase = "engine-provision"
        self.channel = child_provision(endpoint.pipe, parent_pid=parent.pid,
            manifest=endpoint.manifest, deadline=self.deadline)
        self.identity = self.channel.identity; self.fence = EngineFence(self.revoked)
        self.parent = parent
        install(self)
        phase(self, "engine-native-model-init")
        self.observer = threading.Thread(target=self._watch, name='native-engine-parent-watch', daemon=True)
        self.observer.start()

    def _watch(self):
        while not self.stop.wait(.2):
            if not self.parent.is_alive() or time.monotonic() > self.deadline:
                why = "parent peer exited" if not self.parent.is_alive() else "authority deadline expired"
                failure("ENGINE_AUTHORITY_WATCHDOG", RuntimeError(why), self)
                self.revoked.set()
                if not self.stop.wait(5):os._exit(73)
                return

    def build(self, *, core, verify_quiescent):
        from .live_factory import build_engine_components
        from native_worker_registration import register_uniproc
        from policy_runtime.goal_contract_v13 import ProducerProvenance
        from engine_assembly import EngineAssembly, AssemblyConfig
        from vllm.v1.engine import EngineCoreRequest
        from vllm.v1.serial_utils import MsgpackDecoder
        self.verify_identity(); self.core = core
        phase(self, 'engine-renderer-provider')
        parts = build_engine_components(core, self, verify_quiescent)
        phase(self, 'engine-register-worker')
        self.gate = register_uniproc(executor=core.model_executor,
            channel_epoch=self.identity.epoch, coordinator_epoch=parts['provider'].coordinator._epoch,
            provenance=ProducerProvenance(**asdict(parts['provenance'])),
            projection_seed=self.projection_seed, verify_loaded=self.verify_identity, enabled=True)
        def worker_check():
            self.gate.before_execute(None); return True
        phase(self, "engine-dormant-assembly")
        self.assembly = EngineAssembly(config=AssemblyConfig(enabled=True,
            identity=self.identity.manifest, epoch=self.identity.epoch), core=core,
            provider=parts['provider'], renderer=parts['renderer'], codec=self.codec,
            core_type=EngineCoreRequest, decoder_type=MsgpackDecoder,
            verify_source=self.verify_identity, verify_worker=worker_check,
            verify_quiescent=verify_quiescent, selector_bridge=parts['selector_bridge'])
        return self.assembly

    def release_b0(self, core):
        self.verify_identity()
        require(core is self.core and self.assembly is core._remote_ra_assembly
                and self.assembly.require_active(), 'native engine assembly owner differs')
        capability = capability_from_worker_gate(self.gate, channel_epoch=self.identity.epoch,
            coordinator_epoch=self.gate.request['coordinator_epoch'], challenge=self.gate.request['challenge'])
        prepared = capability.payload(); self.worker_nonce = capability.worker_nonce
        live_state = getattr(core, '_remote_ra_live_v2', None)
        require(type(live_state) is dict, 'live v2 engine composition missing')
        prepared['live_v2_readiness'] = live_state['readiness'].report()
        require(prepared['live_v2_readiness'].get('native_ready') is True,
                'loaded worker readiness not qualified')
        phase(self, 'engine-send-prepared-live')
        self.channel.send('PREPARED_LIVE', prepared)
        receipt = {'receipt_hash': digest(prepared)}
        require(self.channel.receive({'COMMIT'}, self.deadline)[1] == receipt, 'commit differs')
        self.channel.send('COMMIT_ACK', receipt)
        require(self.channel.receive({'RELEASE'}, self.deadline)[1] == receipt, 'release differs')
        self.fence.capable(); self.channel.send('RELEASE_ACK', receipt)
        self.channel.handoff_to_monitor(); self.deadline = time.monotonic() + 90
        phase(self, 'engine-b0-released')
        event('ENGINE_B0_RELEASED', epoch=self.identity.epoch, worker_nonce=self.worker_nonce,
              physical_capture_verified=False)

    def native_ready(self, kind):
        self.owner_check(); self.fence.native_ready(kind)
        event(kind, epoch=self.identity.epoch)

    def verify_admission(self):
        self.verify_identity(); self.channel._check()
        require(not getattr(self,'closing',False),'engine admission closing')
        require(self.channel.phase == 'MONITOR' and self.channel.identity == self.identity,
                'private engine channel phase/epoch changed')
        self.fence.require_add('registered')
        return True

    def poll(self):
        self.owner_check()
        if getattr(self,'closing',False):return
        self.fence.check()
        if not self.channel.pipe.poll(0):return
        kind, payload = self.channel.receive({'PING', 'FRONTEND_BOUND', 'CLOSE'}, self.deadline)
        self.deadline = time.monotonic() + 90
        if kind == 'PING':self.channel.send('PONG', payload)
        elif kind == 'FRONTEND_BOUND':
            require(payload['worker_nonce'] == self.worker_nonce, 'frontend worker nonce mismatch')
            self.fence.bind_frontend(); self.channel.send('FRONTEND_ACK', payload)
            event('ENGINE_F1_ACK', epoch=self.identity.epoch)
        else:
            from remote_ra_http.shutdown import request_close
            request_close(self)

    def close(self, receipt_hash=None):
        from remote_ra_http.shutdown import close_authority
        return close_authority(self,receipt_hash)


def engine_bootstrap(endpoint, config):
    if os.environ.get('REMOTE_RA_DYNAMIC_STARTUP', '0') == '0':
        require(endpoint is None, 'unexpected authority in static mode'); return None
    authority = NativeEngine(endpoint, config)
    import startup_runtime
    startup_runtime.install_server_factory(authority.build)
    return authority
