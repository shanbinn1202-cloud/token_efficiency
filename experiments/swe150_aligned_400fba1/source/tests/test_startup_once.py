"""The startup config is read once (review 1110, runtime 1.1 / perf 2.1). CPU only.

Since async-c the topology gate of every engine frame (``EngineAssembly.require_active`` ->
``EngineReadiness`` -> ``bootstrap_contract.validate_topology`` -> ``async_sched.allowed``) read
``startup.json`` again: the file, its SHA-256, the JSON parse and the whole validation, on the engine thread,
for every RegisteredAdd, Cancel, CloseSession and refused retry. The rule is: integrity checks at startup
only. ``EngineReadiness`` now reads the --async-scheduling pin once, when the assembly is built, and the gate
compares the engine's config objects with it.

Here: the real EngineAssembly (test_gemma_assembly's objects) on a real startup config written by
serve.prepare (synchronous and --async-scheduling), with ``remote_ra_http.config.load``, its SHA-256 and the
file read counted.
* startup still reads, hashes and validates the config, and still refuses a drifted file;
* after startup the gate (``require_active``, ``readiness`` and a frame through the dispatcher) reads no
  file and hashes nothing, also when the file is gone;
* the gate keeps the pin: a scheduler config that differs from it is refused on every frame.
"""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import runtime_renderer as RR  # noqa: E402
import test_gemma_assembly as TGA  # noqa: E402
from remote_ra_http import config as http_config  # noqa: E402

TASKS = [dict(instance_id='repo__a-1', problem_statement='Fix a.'),
         dict(instance_id='repo__b-2', problem_statement='Fix b.')]


class Counts:
    """Calls of config.load, of its SHA-256 and reads of the startup config file."""
    def __init__(self, test, path):
        self.load = self.hash = self.read = 0
        load, sha256, read_bytes = http_config.load, http_config.sha256, Path.read_bytes
        target = Path(path)

        def counted_load():
            self.load += 1
            return load()

        def counted_hash(*args):
            self.hash += 1
            return sha256(*args)

        def counted_read(this):
            if Path(this) == target:
                self.read += 1
            return read_bytes(this)
        test.enterContext(patch.object(http_config, 'load', counted_load))
        test.enterContext(patch.object(http_config, 'sha256', counted_hash))
        test.enterContext(patch.object(Path, 'read_bytes', counted_read))

    def take(self):
        value = (self.load, self.hash, self.read)
        self.load = self.hash = self.read = 0
        return value


class StartupOnceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import install
        import serve
        import test_installer as INSTALL
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        with contextlib.redirect_stdout(io.StringIO()):
            package, site, model, release = INSTALL.fixture(root)
            install.install(site, model, release, package_root=package, apply=True)
            kw = dict(kv_cache_gib=24, page_tokens=16, cuda_graphs='full', max_model_len=131072,
                      decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True)
            configs = dict(sync=serve.prepare(release, [dict(t) for t in TASKS], 'full', root / 'sync', 8231, **kw),
                           on=serve.prepare(release, [dict(t) for t in TASKS], 'full', root / 'on', 8231,
                                            async_scheduling=True, **kw))
        cls.files = {}
        for name, prepared in configs.items():
            raw = serve.wire(prepared[1])
            cls.files[name] = (raw, serve.digest(raw))

    def start(self, name, *, sha=None):
        """The assembly of a server whose startup config is ``name``; the file is this test's own copy."""
        raw, pin = self.files[name]
        directory = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(directory) / 'startup.json'
        path.write_bytes(raw)
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                                     REMOTE_RA_HTTP_CONFIG_SHA256=sha or pin))
        asynchronous = name == 'on'
        topology = TGA.topology

        def pinned_topology():
            value = topology()
            value.scheduler_config.async_scheduling = asynchronous
            value.max_concurrent_batches = 2 if asynchronous else 1
            return value
        self.enterContext(patch.object(TGA, 'topology', pinned_topology))
        counts = Counts(self, path)
        assembly, core = TGA.assemble(self, TGA.bare(RR.EngineSourceRenderer))
        return assembly, core, counts, path

    def frame(self, assembly):
        """One frame through the dispatcher's gate (the frame itself is refused by the codec after it)."""
        steps = assembly.dispatcher.handle_steps(b'not a registered frame')
        with self.assertRaises(Exception) as caught:
            next(steps)
        self.assertNotIn('startup', str(caught.exception))
        self.assertNotIn('readiness', str(caught.exception))

    def check_startup(self, name):
        assembly, _core, counts, _path = self.start(name)
        load, hashed, read = counts.take()
        self.assertGreaterEqual(load, 1)
        self.assertGreaterEqual(hashed, load)
        self.assertEqual(read, load)
        self.assertIs(assembly.readiness.asynchronous, name == 'on')

    def test_startup_reads_and_validates_the_config_synchronous(self):
        self.check_startup('sync')

    def test_startup_reads_and_validates_the_config_async_scheduling(self):
        self.check_startup('on')

    def test_startup_still_refuses_a_drifted_config(self):
        with self.assertRaisesRegex(Exception, 'startup config drift'):
            self.start('sync', sha='0' * 64)

    def test_startup_still_refuses_another_scheduler(self):
        # The pinned config says asynchronous; the engine was started synchronous (and the reverse).
        for name, other in (('sync', 'on'), ('on', 'sync')):
            with self.subTest(config=name), contextlib.ExitStack() as stack:
                raw, pin = self.files[name]
                path = Path(stack.enter_context(tempfile.TemporaryDirectory())) / 'startup.json'
                path.write_bytes(raw)
                stack.enter_context(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1',
                                               REMOTE_RA_HTTP_CONFIG=str(path), REMOTE_RA_HTTP_CONFIG_SHA256=pin))
                topology = TGA.topology

                def wrong():
                    value = topology()
                    value.scheduler_config.async_scheduling = other == 'on'
                    value.max_concurrent_batches = 2 if other == 'on' else 1
                    return value
                stack.enter_context(patch.object(TGA, 'topology', wrong))
                with self.assertRaisesRegex(Exception, 'scheduler config required'):
                    TGA.assemble(self, TGA.bare(RR.EngineSourceRenderer))

    def check_gate(self, name):
        assembly, core, counts, path = self.start(name)
        self.assertGreaterEqual(counts.take()[0], 1)
        for _ in range(3):
            self.assertIs(assembly.require_active(), True)
            self.assertIs(assembly.readiness(), True)
            self.frame(assembly)
        self.assertEqual(counts.take(), (0, 0, 0), 'the gate read or hashed the startup config again')
        # A startup config that went away or changed mid-run no longer fails every frame.
        path.write_bytes(b'{"changed": true}')
        self.assertIs(assembly.require_active(), True)
        path.unlink()
        self.assertIs(assembly.require_active(), True)
        self.frame(assembly)
        self.assertEqual(counts.take(), (0, 0, 0))
        # The gate still holds the engine to the pin, from memory.
        pinned = name == 'on'
        core.vllm_config.scheduler_config.async_scheduling = not pinned
        with self.assertRaisesRegex(Exception, 'scheduler config required'):
            assembly.require_active()
        core.vllm_config.scheduler_config.async_scheduling = pinned
        core.vllm_config.max_concurrent_batches = 1 if pinned else 2
        with self.assertRaisesRegex(Exception, 'MRV1 required'):
            assembly.readiness()
        core.vllm_config.max_concurrent_batches = 2 if pinned else 1
        self.assertIs(assembly.require_active(), True)
        self.assertEqual(counts.take(), (0, 0, 0))

    def test_the_gate_reads_nothing_synchronous(self):
        self.check_gate('sync')

    def test_the_gate_reads_nothing_async_scheduling(self):
        self.check_gate('on')

    def test_startup_callers_still_read_the_config(self):
        """validate_topology without a pin (the startup callers: the assembly's input check, the native
        authority) reads the startup config as before."""
        import bootstrap_contract
        assembly, core, counts, path = self.start('on')
        counts.take()
        bootstrap_contract.validate_topology(core.vllm_config)
        load, hashed, read = counts.take()
        self.assertEqual((load, read), (1, 1))
        self.assertGreaterEqual(hashed, 1)
        path.write_bytes(b'{}')
        with self.assertRaisesRegex(Exception, 'startup config drift'):
            bootstrap_contract.validate_topology(core.vllm_config)
        for bad in (1, 'yes'):
            with self.subTest(pin=bad), self.assertRaisesRegex(Exception, 'pin must be a bool'):
                bootstrap_contract.validate_topology(core.vllm_config, bad)


if __name__ == '__main__':
    unittest.main()
