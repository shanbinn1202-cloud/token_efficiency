"""File-level identity for the bounded native smoke; not a native-code sandbox."""
import hashlib
import json
from pathlib import Path
import sys


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


class LiveIdentity:
    def __init__(self):
        self.source = Path(__file__).resolve().parents[1]
        self.path = self.source.parent / 'runtime-manifest.json'
        self.manifest_hash = digest(self.path)
        self.manifest = json.loads(self.path.read_text(encoding='utf8'))
        self.site = Path(self.manifest['site']).resolve()
        self.model_root = Path(self.manifest['model_root']).resolve()
        self.pins = dict(self.manifest['source_files'])
        self.verified = {}  # module name -> the module object whose file digest matched
        self.verify_files()

    def verify_files(self):
        require(digest(self.path) == self.manifest_hash, 'live manifest drift')
        for relative, sha in self.pins.items():
            path = self.source / relative
            require(path.resolve().is_relative_to(self.source) and digest(path) == sha,
                    'live release file drift: ' + relative)
        for relative, sha in self.manifest['model_pins'].items():
            require(digest(self.model_root / relative) == sha, 'model metadata drift: ' + relative)
        for relative, row in self.manifest['native_pins'].items():
            require(digest(self.site / relative) == row['sha256'], 'tokenizer native drift')
        return True

    def verify_module(self, name):
        module = sys.modules.get(name)
        require(module is not None and getattr(module, '__spec__', None) is not None,
                'required live module missing: ' + name)
        # A module's file is digested once; later calls (per request) only check that the loaded module is
        # still that object. Re-hashing the file on disk would not prove anything about already-loaded code.
        if self.verified.get(name) is module:
            return True
        path = Path(module.__file__).resolve()
        spec = module.__spec__
        require(spec.name == name and Path(spec.origin).resolve() == path,
                'live module owner/spec mismatch: ' + name)
        if path.is_relative_to(self.source):
            relative = path.relative_to(self.source).as_posix()
            sha = self.pins.get(relative, self.manifest['baseline_files'].get(relative))
        elif path.is_relative_to(self.site):
            relative = path.relative_to(self.site).as_posix()
            require(not name.startswith(('vllm.', 'remote_ra_linux.')),
                    'live core must come from the sealed release')
            sha = self.manifest['baseline_files'].get(relative)
        else:
            raise RuntimeError('live core origin outside release/site: ' + name)
        require(sha is not None and digest(path) == sha, 'live module SHA mismatch: ' + name)
        self.verified[name] = module
        return True

    def verify_loaded(self):
        # No transitive namespace/path-shape or per-function bytecode gate.
        for name in ('vllm.v1.engine.utils', 'vllm.v1.engine.core',
                     'vllm.v1.executor.uniproc_executor', 'vllm.v1.worker.worker_base',
                     'vllm.v1.worker.gpu_worker', 'vllm.v1.worker.gpu_model_runner',
                     'runtime_renderer', 'tokenizer_semantics', 'remote_ra_linux.native_authority'):
            if name in sys.modules:
                self.verify_module(name)
        return True
