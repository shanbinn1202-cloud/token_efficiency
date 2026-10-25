"""CPU regression coverage of real startup functions and authenticated pipes.

Workers below are explicit synthetic doubles; no CUDA/READY claim is emitted.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE=HERE/'source-overlay'
sys.path.insert(0, str(SOURCE))

def report():
    return dict(default_enabled=False, code_ready=True, native_ready=True, worker_attached=True,
        cuda_qualified=False, growing_smoke_qualified=False, live_multi_turn_supported=False,
        system_valid=False, missing=[], worker_receipt_sha256='a' * 64)

def cold_attach(source, output):
    """No composition/controller/qwen imports may precede the dormant hook."""
    sys.path.insert(0, source)
    from remote_ra_live_v2.startup import attach_engine_dormant
    manifest = json.loads((Path(source).parent / 'INSTALL_RELEASE.json').read_bytes())
    verified = []
    def verify(name):
        module = sys.modules.get(name)
        if module is None: raise RuntimeError('required live module missing: ' + name)
        relative = Path(module.__file__).relative_to(source).as_posix()
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != manifest['source_files'][relative]:
            raise RuntimeError('loaded SHA mismatch: ' + name)
        verified.append(name)
    owner = NS(owner_check=lambda: None, verify_identity=lambda: True, loaded=NS(verify_module=verify))
    ticket = NS(phase='active')
    core = NS(_remote_ra_growing_owner=owner, _remote_ra_startup_ticket=ticket, engines_running=False)
    try:
        state = attach_engine_dormant(core, ticket)
        output.send(dict(ok=True, verified=verified, native_ready=state['readiness'].native_ready))
    except BaseException as exc:
        output.send(dict(ok=False, error=type(exc).__name__, reason=str(exc)))
    finally: output.close()

def pipe_engine(pipe, output, policy, directory):
    from private_authority.protocol import child_provision, EngineFence
    from remote_ra_linux.native_authority import NativeEngine
    import remote_ra_linux.native_authority as native
    from types import ModuleType
    stub=ModuleType('vllm.v1.engine.core');stub.EngineShutdownState=NS(REQUESTED='REQUESTED')
    sys.modules[stub.__name__]=stub
    child_closing=False
    child = NativeEngine.__new__(NativeEngine)
    child.owner = threading.get_ident()
    child.started_at = time.monotonic()
    child.deadline = time.monotonic() + 8
    child.diagnostic_phase = 'synthetic-engine'
    child.revoked, child.stop = threading.Event(), threading.Event()
    child.policy = policy
    child.channel = child_provision(pipe, parent_pid=os.getppid(), manifest='b' * 64, deadline=child.deadline)
    child.identity = child.channel.identity
    child.fence = EngineFence(child.revoked)
    child.verify_identity = lambda: True
    child.assembly = NS(require_active=lambda: True, close=lambda: None)
    child.core = NS(_remote_ra_assembly=child.assembly,
        _remote_ra_live_v2={'readiness': NS(report=report)})
    child.gate = NS(request=dict(coordinator_epoch='c'*64, challenge='d'*64))
    capability = NS(worker_nonce='e'*64, payload=lambda: dict(worker_nonce='e'*64,
        capability_hash='f'*64, physical_capture_verified=False, gpu_drain_verified=False))
    native.capability_from_worker_gate = lambda *a, **kw: capability
    try:
        child.release_b0(child.core)  # exact emitted production B0 method
        child.native_ready('INPUT_READY')
        child.native_ready('HANDSHAKE_READY')
        while not getattr(child,'closing',False):
            child.poll()
            time.sleep(.005)
        from remote_ra_http.shutdown import save_receipt
        digest=save_receipt(Path(directory)/'engine-cleanup.json',dict(joined=True,
            epoch=child.identity.epoch,session_id=child.session_id,runtime_manifest='b'*64))
        child.channel.send('CLOSED',{'cleanup_sha256':digest});child.stop.set()
    except RuntimeError as exc:
        if str(exc) != 'native authority closed by parent': raise
    finally:
        output.send(dict(policy=policy, closed=child.stop.is_set(), phase=child.channel.phase))
        output.close()
        pipe.close()

class StartupTests(unittest.TestCase):
    def spawned_attach(self, source):
        ctx = mp.get_context('spawn')
        parent, child = ctx.Pipe(duplex=False)
        process = ctx.Process(target=cold_attach, args=(str(source), child))
        process.start(); child.close()
        try:
            self.assertTrue(parent.poll(10), 'cold attach child timed out')
            value = parent.recv()
            process.join(10)
            self.assertEqual(process.exitcode, 0)
            return value
        finally:
            if process.is_alive(): process.terminate(); process.join(5)
            parent.close()


    def test_v24_cold_spawn_imports_then_verifies(self):
        result = self.spawned_attach(SOURCE)
        self.assertTrue(result['ok'], result)
        self.assertEqual(len(result['verified']), 6)
        self.assertFalse(result['native_ready'])

    def test_legacy_prepared_unchanged_new_report_strict(self):
        from private_authority.protocol import validate_payload, AuthorityError
        old = dict(worker_nonce='1'*64, capability_hash='2'*64,
            physical_capture_verified=False, gpu_drain_verified=False)
        validate_payload('PREPARED', old)
        with self.assertRaisesRegex(AuthorityError, 'payload schema'):
            validate_payload('PREPARED', dict(old, live_v2_readiness=report()))
        validate_payload('PREPARED_LIVE', dict(old, live_v2_readiness=report()))
        for key, value in [('cuda_qualified', True), ('native_ready', False),
                           ('worker_attached', 1), ('unexpected', True), ('worker_receipt_sha256', 'invalid')]:
            altered = report(); altered[key] = value
            with self.assertRaises(AuthorityError):
                validate_payload('PREPARED_LIVE', dict(old, live_v2_readiness=altered))

    def test_four_policy_real_spawn_private_handshake_and_f1(self):
        from remote_ra_linux.native_authority import NativeManager
        for policy in ('full', 'random50', 'goal_query', 'observation_query'):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                ctx = mp.get_context('spawn')
                pipe, child = ctx.Pipe()
                result, result_child = ctx.Pipe(duplex=False)
                process = ctx.Process(target=pipe_engine, args=(child, result_child, policy, directory))
                manager = NativeManager.__new__(NativeManager)
                manager.policy, manager.pipe = policy, pipe
                manager.loaded = NS(manifest_hash='b'*64)
                manager.cleanup_path=Path(directory)/'engine-cleanup.json'
                manager.process = process
                manager.started_at, manager.deadline = time.monotonic(), time.monotonic() + 10
                manager.revoked, manager.done = threading.Event(), threading.Event()
                manager.b0, manager.f1 = threading.Event(), threading.Event()
                manager.lock, manager.commands = threading.RLock(), queue.Queue(maxsize=1)
                manager.error = manager.channel = manager.identity = None
                manager.live_v2_readiness = None
                process.start(); child.close(); result_child.close()
                thread = threading.Thread(target=manager._run)
                thread.start()
                try:
                    self.assertTrue(manager.b0.wait(8), manager.error)
                    self.assertEqual(manager.live_v2_readiness, report())
                    manager.commands.put(('BIND', '3'*64), timeout=2)
                    self.assertTrue(manager.f1.wait(5), manager.error)
                    manager.commands.put(('CLOSE', None), timeout=2)
                    self.assertTrue(manager.done.wait(5), manager.error)
                    thread.join(5)
                    self.assertTrue(result.poll(5))
                    self.assertEqual(result.recv(), dict(policy=policy, closed=True, phase='MONITOR'))
                    process.join(5)
                    self.assertEqual(process.exitcode, 0)
                    self.assertIsNone(manager.error)
                finally:
                    manager.revoked.set()
                    if process.is_alive(): process.terminate(); process.join(5)
                    thread.join(5); pipe.close(); result.close()

    def test_timeout_and_eof_reasons_distinct(self):
        from private_authority.protocol import receive_raw, AuthorityError
        ctx = mp.get_context('spawn')
        parent, child = ctx.Pipe()
        try:
            with self.assertRaisesRegex(AuthorityError, 'phase deadline expired'):
                receive_raw(parent, time.monotonic() + .01)
            child.close()
            with self.assertRaisesRegex(AuthorityError, 'peer EOF'):
                receive_raw(parent, time.monotonic() + 1)
        finally: parent.close(); child.close()

    def test_original_failure_persisted_before_cleanup(self):
        import startup_runtime
        import remote_ra_http.diagnostics as diagnostics
        from remote_ra_http import config
        with tempfile.TemporaryDirectory() as directory:
            owner = NS(identity=NS(epoch='a'*64), diagnostic_phase='engine-loaded-worker-compose',
                policy='full', started_at=time.monotonic()-150, deadline=time.monotonic()+450)
            observed = []
            def close():
                paths = list(Path(directory).glob('failure-engine-construction-failed-*.json'))
                self.assertEqual(len(paths), 1)
                row = json.loads(paths[0].read_bytes())
                self.assertIn('original exact refusal', row['traceback'])
                self.assertGreater(row['deadline_remaining_seconds'], 400)
                observed.append(row)
            owner.close = close
            core = NS(_remote_ra_authority=owner)
            try: raise ValueError('original exact refusal')
            except ValueError as original:
                with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'), \
                        patch.object(config, 'load', return_value={'output_directory': directory}), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        self.assertRaises(BaseExceptionGroup):
                    startup_runtime.failed_constructor(core, original, cleanup_process=())
            self.assertEqual(len(observed), 1)

    def test_failed_diagnostic_storage_cannot_mask_original(self):
        from remote_ra_http.diagnostics import failure
        from remote_ra_http import config
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'), \
                patch.object(config, 'load', side_effect=RuntimeError('invalid startup config')), \
                contextlib.redirect_stdout(io.StringIO()) as stream:
            result = failure('TEST_FAILURE', ValueError('actual fault'))
        self.assertEqual(result['reason'], 'actual fault')
        self.assertIn('DIAGNOSTIC_PERSIST_FAILED', stream.getvalue())

if __name__ == '__main__': unittest.main()
