"""Real manager-owned stdlib spawn transport qualification.

The same early resource ownership seam is emitted into CoreEngineProcManager.
Zero-GPU qualification uses the approved CPU engine entrypoint, never EngineCore
or a worker. Production stays smoke-gated; this receipt is not a live worker B0.
"""
import multiprocessing as mp
import functools
import os
import threading
import time
from dataclasses import dataclass
from private_authority.cpu_launch import CpuAuthorityLaunch, Scenario, engine_entry
from private_authority.protocol import Topology, require
from private_authority.evidence import verify_sources

def guarded_manager_constructor(fn):
    @functools.wraps(fn)
    def construct(manager, *args, **kwargs):
        try:
            return fn(manager, *args, **kwargs)
        except BaseException as original:
            plan = getattr(manager, '_remote_ra_authority', None)
            if plan is not None:
                try:
                    if plan._process is None:
                        started = [p for p in getattr(manager, 'processes', ()) if p.pid is not None]
                        require(len(started) <= 1, 'single-child cleanup owner required')
                        if started:
                            require(started[0]._parent_pid == os.getpid(), 'process cleanup owner differs')
                            plan._process = started[0]
                    plan.close()
                except BaseException as cleanup:
                    raise BaseExceptionGroup('manager construction and authority cleanup failed',
                                             [original, cleanup]) from None
            raise
    return construct

@dataclass(frozen=True)
class ChildEndpoint:
    pipe: object
    report_pipe: object
    native_init: object
    scenario: object

def preflight_child(endpoint):
    require(type(endpoint) is ChildEndpoint, 'exact manager endpoint required')
    engine_entry(endpoint.pipe, endpoint.report_pipe, endpoint.native_init, endpoint.scenario)

class ManagerAuthority(CpuAuthorityLaunch):
    def __init__(self, *, manager, context, local_engine_count, timeout=5.0, scenario=None):
        require(type(local_engine_count) is int and local_engine_count == 1,
                'exact one-child count required before resource allocation')
        Topology().validate(context)
        require(not hasattr(manager, '_remote_ra_authority'), 'manager already owns authority')
        super().__init__(context=context, timeout=timeout, scenario=scenario)
        self.manager = manager
        self._child_pipe = self._send_report = None
        self._endpoint_taken = False
        manager._remote_ra_authority = self  # before first Pipe/Event allocation
        try:
            verify_sources()
            self._pipe, self._child_pipe = context.Pipe(duplex=True)
            self._telemetry, self._send_report = context.Pipe(duplex=False)
            self.native_init = context.Event()
        except BaseException:
            self.close()
            raise

    def child_endpoint(self):
        self._check_owner()
        require(not self._closed and not self._endpoint_taken, 'child endpoint is single-use')
        self._endpoint_taken = True
        return ChildEndpoint(self._child_pipe, self._send_report, self.native_init, self.scenario)

    def after_spawn(self, process):
        self._check_owner()
        require(self._endpoint_taken and self._process is None and process.pid is not None
                and process._start_method == 'spawn' and process._parent_pid == os.getpid(),
                'same manager must own one actual spawned child')
        self._process = process
        self._started = True
        self._child_pipe.close()
        self._send_report.close()
        self._deadline = time.monotonic() + self.timeout
        self._sideband = threading.Thread(target=self._run, name='manager-private-sideband')
        self._watchdog = threading.Thread(target=self._watch, name='manager-private-watchdog')
        try:
            self._sideband.start()
            self._watchdog.start()
        except BaseException:
            self._fence('thread creation failed')
            self._reap()
            self._close_transport()
            self._done.set()
            for thread in (self._sideband, self._watchdog):
                if thread.ident is not None:
                    thread.join(2)
            self._closed = True
            raise

    def close(self):
        self._check_owner()
        if self._closed:
            return
        if self._started and self._sideband is not None and self._watchdog is not None:
            super().close()
        else:
            self._revoked.set()
            if self._process is not None and self._process.pid is not None:
                self._reap()
            for pipe in (self._pipe, self._child_pipe, self._telemetry, self._send_report):
                if pipe is not None and not pipe.closed:
                    pipe.close()
            self._done.set()
            self._closed = True

def prepare_launch(vllm_config, context, *, manager, local_engine_count):
    value = os.environ.get('REMOTE_RA_DYNAMIC_STARTUP', '0')
    require(value in ('0', '1'), 'canonical feature flag required')
    requested = getattr(vllm_config, '_remote_ra_dynamic_requested', False)
    require(type(requested) is bool, 'canonical dynamic marker required')
    if value == '0' and not requested:
        return None
    require(type(local_engine_count) is int and local_engine_count == 1,
            'one engine required before allocation')
    Topology().validate(context)
    # Real launch has its own authority and worker gate, never the CPU fixture.
    from .native_authority import NativeManager
    return NativeManager(manager, context, vllm_config, local_engine_count)

def run_transport_preflight(*, scenario=None, timeout=5):
    class Owner:
        pass
    owner = Owner()
    plan = ManagerAuthority(manager=owner, context=mp.get_context('spawn'),
                            local_engine_count=1, timeout=timeout, scenario=scenario)
    process = None
    try:
        process = plan.context.Process(target=preflight_child, args=(plan.child_endpoint(),),
                                       name='remote-ra-zero-gpu-private-engine')
        # Own the Process before start so any partial start failure is cleanup-owned.
        process.start()
        plan.after_spawn(process)
        plan.advance_native_init()
        plan.wait_b0()
        plan.bind_frontend('f' * 64)
        plan.require_frontend_admission()
        result = {'transport': 'REAL_STDLIB_SPAWN_PRIVATE_PIPE_PASS',
                  'manager_owned': owner._remote_ra_authority is plan,
                  'native_vllm_manager_constructed': False,
                  'worker_capability': 'CPU_FIXTURE_ONLY', 'system_valid': False}
    finally:
        # A Process.start failure may occur before after_spawn attached it.
        if process is not None and process.pid is not None and plan._process is None:
            plan._process = process
        plan.close()
    return result
