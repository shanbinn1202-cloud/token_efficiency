"""Reviewed task-specific ARM64 compatibility, without changing benchmark files."""
import asyncio
import hashlib
import json
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path

from container_base import Arm64ApptainerEnvironment as Base
from kvm_environment import start_kvm

R = Path(__file__).resolve().parent
D = R.parents[1]
ASSETS = D / 'terminal-public-inputs'
KVM = set(json.loads((R / 'backend-map.json').read_text())['kvm'])
COMMANDS = json.loads((R / 'environment-commands.json').read_text())


class Arm64ApptainerEnvironment(Base):
    async def _checked(self, command, timeout_sec=240):
        result = await self.exec(command=command, timeout_sec=timeout_sec)
        if result.return_code:
            raise RuntimeError('ARM64 environment preparation failed: ' +
                               (result.stdout or '')[-1500:] + (result.stderr or '')[-1500:])
        return result

    async def _commands(self, name):
        for command in COMMANDS[name]:
            await self._checked(**command)

    async def _start_server(self):
        name = self._sif_path.stem
        self._public_cache = None
        self._cache_dir = None
        if name == 'caffe-cifar-10':
            self._cache_dir = Path(tempfile.mkdtemp(prefix='terminal-public-cache-'))
            self._public_cache = await asyncio.create_subprocess_exec(
                sys.executable, str(R / 'public_input_proxy.py'),
                str(ASSETS / 'cifar-10-binary.tar.gz'),
                str(self._cache_dir / 'ready.json'), str(self._cache_dir / 'hits.jsonl'),
                start_new_session=True)
            for _ in range(100):
                if (self._cache_dir / 'ready.json').exists():
                    break
                if self._public_cache.returncode is not None:
                    raise RuntimeError('Public input cache failed to start')
                await asyncio.sleep(.1)
            port = json.loads((self._cache_dir / 'ready.json').read_text())['port']
            # Instance-local arguments avoid mutating os.environ during concurrent trials.
            self._public_proxy = f'http://127.0.0.1:{port}'
        if name in KVM:
            await start_kvm(self)
        else:
            await super()._start_server()
        if name in {'qemu-startup', 'qemu-alpine-ssh'}:
            await self._commands('bullseye')
        if name in {'mteb-leaderboard', 'mteb-retrieve'}:
            await self._commands('mteb')
        if name == 'rstan-to-pystan':
            await self._commands('curl_pin')
        if name == 'mailman':
            await self._checked('postfix post-install upgrade-permissions', 60)
        if name == 'mcmc-sampling-stan':
            await self._checked('apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake && dpkg-query -W cmake cmake-data', 300)
        if name == 'caffe-cifar-10':
            await self._checked("printf '%s\\n' 'Acquire::http::Proxy \"DIRECT\";' > /etc/apt/apt.conf.d/99public-cache-direct", 30)

    async def start(self, force_build):
        await super().start(force_build)
        if self._sif_path.stem in {'make-doom-for-mips', 'make-mips-interpreter'}:
            artifact = ASSETS / 'doom1.wad'
            expected = '1d7d43be501e67d927e415e0b8f3e29c3bf33075e859721816f652a526cac771'
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Public WAD cache hash mismatch')
            await self.upload_file(artifact, '/app/doom.wad')
            result = await self._checked('sha256sum /app/doom.wad', 20)
            if not result.stdout.startswith(expected):
                raise RuntimeError('Public WAD copy hash mismatch')

    async def stop(self, delete):
        try:
            await super().stop(delete)
        finally:
            process = getattr(self, '_public_cache', None)
            if process is not None and process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
                await process.wait()
            cache = getattr(self, '_cache_dir', None)
            if cache is not None:
                shutil.rmtree(cache, ignore_errors=True)
