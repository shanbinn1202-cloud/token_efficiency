# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Singularity/Apptainer environment for running tasks on HPC clusters.

This environment converts Docker images to Singularity .sif format and
runs a FastAPI server inside the container to handle command execution.
"""

import asyncio
import asyncio.subprocess
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import override

if sys.platform != "win32":
    import fcntl

import httpx

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


class MemoryLimitExceededError(Exception):
    """Raised when a container exceeds its memory limit."""

    pass


class Arm64ApptainerEnvironment(BaseEnvironment):
    """
    Singularity-based environment for HPC clusters.

    This environment:
    1. Pulls Docker images and converts them to .sif format
    2. Runs a FastAPI server inside the container for command execution
    3. Uses bind mounts for file transfer

    Kwargs (via --ek / environment.kwargs in trial config):
        singularity_image_cache_dir: Path to cache .sif files.
        singularity_force_pull: Force re-pull even if cached (default False).
        singularity_no_mount: Comma-separated mount types to suppress
            (default "home,tmp,bind-paths"). Use "" to allow all Singularity mounts.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        singularity_image_cache_dir: Path | str | None = None,
        singularity_force_pull: bool = False,
        singularity_no_mount: str | None = None,
        **kwargs,
    ):
        if singularity_image_cache_dir:
            self._image_cache_dir = Path(singularity_image_cache_dir)
        else:
            self._image_cache_dir = Path(tempfile.mkdtemp(prefix="singularity_cache_"))

        self._force_pull = singularity_force_pull
        self._singularity_no_mount = singularity_no_mount

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            **kwargs,
        )

        self._server_process: asyncio.subprocess.Process | None = None
        self._server_port: int | None = None
        self._staging_dir: Path | None = None
        self._sif_path: Path | None = None
        self._stream_task: asyncio.Task[None] | None = None
        self._memory_watchdog_task: asyncio.Task[None] | None = None
        self._http_client: httpx.AsyncClient | None = None

        memory_mb = self._effective_memory_mb
        self._memory_limit_bytes = (
            memory_mb * 1024 * 1024 if memory_mb is not None else None
        )
        self._memory_limit_exceeded: str | None = None

        self._workdir = self._resolve_workdir()

    @staticmethod
    @override
    def type() -> EnvironmentType:
        return EnvironmentType.SINGULARITY

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities()

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=True)

    @property
    def _docker_image(self) -> str | None:
        return self.task_env_config.docker_image

    @property
    def _staging(self) -> Path:
        """Return staging dir, raising if not yet initialized."""
        if self._staging_dir is None:
            raise RuntimeError("Staging directory not initialized — call start() first")
        return self._staging_dir

    @property
    def _is_sif_image(self) -> bool:
        """True when docker_image points to a pre-built .sif file."""
        return bool(self._docker_image and self._docker_image.endswith(".sif"))

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @override
    def _validate_definition(self):
        """Validate that required files and configuration exist."""
        if not self._docker_image:
            raise ValueError(
                "Singularity environment requires 'docker_image' in task.toml [environment]. "
                "Set it to a Docker image name (e.g. 'ubuntu:22.04') or a .sif file path."
            )

        if self._is_sif_image:
            sif_path = Path(self._docker_image)
            if not sif_path.exists():
                raise FileNotFoundError(
                    f".sif file not found: {sif_path}. "
                    "Please convert Docker images to .sif format first."
                )
            self.logger.debug(f"Using pre-built .sif image: {sif_path}")

    def _resolve_workdir(self) -> str:
        """Resolve container workdir from Dockerfile WORKDIR or default to /app."""
        if self._dockerfile_path.exists():
            workdir = "/app"
            try:
                for line in self._dockerfile_path.read_text().splitlines():
                    line = line.strip()
                    if line.upper().startswith("WORKDIR "):
                        workdir = line.split(None, 1)[1].strip()
            except Exception:
                pass
            return workdir
        return "/app"

    def _reserve_port(self) -> tuple[socket.socket, int]:
        """Reserve a free port by keeping the socket bound.

        Returns a tuple of (socket, port). The caller must close the socket
        when ready to use the port, minimizing the race condition window.

        Uses SO_REUSEADDR so the port can be immediately reused after closing.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        return s, port

    async def _convert_docker_to_sif(
        self, docker_image: str, *, force_pull: bool = False
    ) -> Path:
        """Convert a Docker image to Singularity .sif format.

        Uses file locking to prevent race conditions when multiple concurrent
        tasks try to convert the same image simultaneously.
        """
        if ":" not in docker_image:
            docker_image = f"{docker_image}:latest"

        safe_name = docker_image.replace("/", "_").replace(":", "_")
        sif_path = self._image_cache_dir / f"{safe_name}.sif"
        lock_path = self._image_cache_dir / f"{safe_name}.sif.lock"

        self._image_cache_dir.mkdir(parents=True, exist_ok=True)

        if not force_pull and sif_path.exists():
            self.logger.debug(f"Using cached Singularity image: {sif_path}")
            return sif_path

        self.logger.debug(f"Acquiring lock for image conversion: {docker_image}")
        lock_file = open(lock_path, "w")
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            )
            self.logger.debug(f"Lock acquired for: {docker_image}")

            if force_pull and sif_path.exists():
                self.logger.debug(
                    f"Force pull enabled, removing cached image: {sif_path}"
                )
                sif_path.unlink()

            if sif_path.exists():
                self.logger.debug(
                    f"Using cached Singularity image (created by another process): "
                    f"{sif_path}"
                )
                return sif_path

            self.logger.debug(f"Converting Docker image to Singularity: {docker_image}")

            tmp_sif_path = (
                self._image_cache_dir / f"{safe_name}.sif.tmp.{self.session_id}"
            )

            cmd = [
                "singularity",
                "pull",
                str(tmp_sif_path),
                f"docker://{docker_image}",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                if tmp_sif_path.exists():
                    tmp_sif_path.unlink()
                error_msg = stderr.decode(errors="replace")
                raise RuntimeError(f"Failed to convert Docker image: {error_msg}")

            tmp_sif_path.rename(sif_path)

            self.logger.debug(f"Created Singularity image: {sif_path}")
            return sif_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    async def _cleanup_server_resources(self) -> None:
        """Clean up any existing server resources from a previous run."""
        if (
            hasattr(self, "_server_process")
            and self._server_process
            and self._server_process.returncode is None
        ):
            self._server_process.kill()
            await self._server_process.wait()
        if (
            hasattr(self, "_stream_task")
            and self._stream_task
            and not self._stream_task.done()
        ):
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        if hasattr(self, "_http_client") and self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        if (
            hasattr(self, "_memory_watchdog_task")
            and self._memory_watchdog_task
            and not self._memory_watchdog_task.done()
        ):
            self._memory_watchdog_task.cancel()
            try:
                await self._memory_watchdog_task
            except asyncio.CancelledError:
                pass
        if self._staging_dir and self._staging_dir.exists():
            shutil.rmtree(self._staging_dir, ignore_errors=True)
            self._staging_dir = None

    async def _start_server(self) -> None:
        """Start the FastAPI server inside the Singularity container.

        Uses port reservation with retry logic to handle race conditions where
        another process might grab the port between reservation and binding.
        """
        await self._cleanup_server_resources()

        self._staging_dir = Path(tempfile.mkdtemp(prefix="singularity_staging_"))
        self._staging_dir.chmod(0o755)

        server_script = Path(__file__).parent / "server.py"
        staging_server = self._staging_dir / "_hbexec.py"
        shutil.copy(server_script, staging_server)

        bootstrap_src = Path(__file__).parent / "bootstrap.sh"
        bootstrap_script = self._staging_dir / "bootstrap.sh"
        shutil.copy(bootstrap_src, bootstrap_script)
        bootstrap_script.chmod(0o755)

        # A private writable directory avoids overlay rename(EXDEV) failures in
        # ordinary apt/dpkg operations. The base SIF remains immutable.
        self._sandbox_parent = Path(tempfile.mkdtemp(prefix="terminal-arm64-root-"))
        sandbox = self._sandbox_parent / "rootfs"
        # Apptainer's implicit unsquashfs starts one worker per host CPU;
        # this exceeds the PID limit with concurrent containers. Extract the
        # same ARM64 system partition directly with four workers instead.
        listing = await asyncio.create_subprocess_exec(
            "apptainer", "sif", "list", str(self._sif_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        table, error = await listing.communicate()
        if listing.returncode:
            raise RuntimeError("Cannot inspect SIF system partition: " + error.decode())
        import re
        offsets = re.findall(r"\|\s*(\d+)-\d+\s*\|FS \(Squashfs/\*System/arm64\)", table.decode())
        if len(offsets) != 1:
            raise RuntimeError("Expected exactly one native ARM64 system partition")
        build = await asyncio.create_subprocess_exec(
            "unsquashfs", "-no-progress", "-processors", "4", "-offset", offsets[0],
            "-dest", str(sandbox), str(self._sif_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, umask=0)
        output, _ = await build.communicate()
        if build.returncode:
            raise RuntimeError("ARM64 sandbox extraction failed: " + output.decode(errors="replace")[-4000:])
        for directory in ("staging", "logs/agent", "logs/verifier", "logs/artifacts", "tmp", "var/tmp"):
            (sandbox / directory).mkdir(parents=True, exist_ok=True)
        # OCI image layers omit the runtime-generated hosts file. Docker
        # supplies localhost entries; provide the same loopback resolution in
        # each private Apptainer rootfs without copying host cluster mappings.
        hosts_path = sandbox / "etc/hosts"
        hosts_text = hosts_path.read_text() if hosts_path.exists() else ""
        host_rows = [line.split("#", 1)[0].split() for line in hosts_text.splitlines()]
        additions = []
        for address, aliases in (("127.0.0.1", "localhost"),
                                 ("::1", "localhost ip6-localhost ip6-loopback")):
            if not any(row and row[0] == address and "localhost" in row[1:]
                       for row in host_rows):
                additions.append(address + " " + aliases)
        if additions:
            hosts_path.write_text(hosts_text.rstrip("\n") + "\n" + "\n".join(additions) + "\n")
        # OCI WORKDIR can name an empty directory omitted from the SIF rootfs.
        (sandbox / self._workdir.lstrip("/")).mkdir(parents=True, exist_ok=True)
        probe = await asyncio.create_subprocess_exec(
            "apptainer", "exec", "--containall", "--no-mount", "home,tmp,bind-paths",
            "--pwd", "/", str(sandbox), "getconf", "GNU_LIBC_VERSION",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        version, _ = await probe.communicate()
        version_match = re.search(rb"glibc (\d+)\.(\d+)", version)
        modern = bool(version_match and tuple(map(int, version_match.groups())) >= (2, 34))
        tool_name = "build-tools" if modern else "runtime-tools-bullseye"
        compat = Path('/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023') / "terminal-images" / tool_name
        if not (compat / "usr/bin/fakeroot-sysv").is_file():
            raise RuntimeError("Verified ARM64 fakeroot runtime is missing")
        (sandbox / "opt/arm64-fakeroot").mkdir(parents=True, exist_ok=True)
        max_port_retries = 3
        last_error = None

        for port_attempt in range(max_port_retries):
            reserved_socket, port = self._reserve_port()
            self._server_port = port

            # Note: --memory and --cpus flags are NOT used because they require
            # cgroups support (systemd running as init), which is typically not
            # available on HPC clusters. Resource limits should be enforced at
            # the SLURM level instead (via --mem, --cpus-per-task).
            env_files_dir = self.environment_dir / "files"
            bind_mounts = [
                "-B",
                f"{self._staging_dir}:/staging",
                "-B", f"{compat}:/opt/arm64-fakeroot:ro",
            ]
            for mount in self._host_mounts:
                if mount.get("type") == "bind":
                    bind_mounts.extend(["-B", f"{mount['source']}:{mount['target']}"])
            if env_files_dir.exists():
                bind_mounts.extend(["-B", f"{env_files_dir}:/staging/env_files"])

            no_mount_args: list[str] = []
            singularity_no_mount = self._singularity_no_mount
            if singularity_no_mount is None:
                singularity_no_mount = "home,tmp,bind-paths"
            if singularity_no_mount:
                for part in singularity_no_mount.split(","):
                    part = part.strip()
                    if part:
                        no_mount_args.extend(["--no-mount", part])

            bootstrap_cmd = [
                "env", "FAKEROOTDONTTRYCHOWN=1", "TB_NATIVE_TASK="+self._sif_path.stem,
                "/opt/arm64-fakeroot/usr/bin/fakeroot-sysv",
                "-l", "/opt/arm64-fakeroot/usr/lib/aarch64-linux-gnu/libfakeroot/libfakeroot-sysv.so",
                "-f", "/opt/arm64-fakeroot/usr/bin/faked-sysv",
                "bash",
                "-c",
                'exec /staging/bootstrap.sh "$@"',
                "bash",
                self._workdir,
                "/staging/_hbexec.py",
                "--port",
                str(self._server_port),
                "--workdir",
                self._workdir,
            ]
            if self._sif_path.stem == 'qemu-alpine-ssh':
                # Native root-mapped namespace; do not emulate stat metadata.
                apt_config = sandbox / 'etc/apt/apt.conf.d'
                if apt_config.is_dir():
                    (apt_config/'99terminal-runner-root').write_text('APT::Sandbox::User "root";\n')
                    (apt_config/'99terminal-runner-root').chmod(0o644)
                bootstrap_cmd = ['env', 'TB_NATIVE_TASK=qemu-alpine-ssh', 'TAR_OPTIONS=--no-same-owner',
                                 '/bin/bash', '/staging/bootstrap.sh', self._workdir,
                                 '/staging/_hbexec.py', '--port', str(self._server_port),
                                 '--workdir', self._workdir]
            if getattr(self, '_public_proxy', None):
                bootstrap_cmd.insert(2, 'http_proxy='+self._public_proxy)
            cmd = [
                "singularity",
                "exec",
                *no_mount_args,
                "--pwd",
                self._workdir,
                "--writable",
                "--fakeroot",
                "--ignore-fakeroot-command",
                "--containall",
                "--pid",
                *bind_mounts,
                str(sandbox),
                *bootstrap_cmd,
            ]

            if self._sif_path.stem == 'qemu-alpine-ssh':
                import hashlib
                available = sorted(os.sched_getaffinity(0))
                cpu = available[int(hashlib.sha256(self.session_id.encode()).hexdigest(), 16) % len(available)]
                cmd = ['taskset', '-c', str(cpu), *cmd]

            self.logger.debug(
                f"Starting Singularity container with server on port "
                f"{self._server_port} (attempt {port_attempt + 1}/{max_port_retries})"
            )

            reserved_socket.close()

            self._server_process = await asyncio.create_subprocess_exec(
                *cmd,
                start_new_session=True,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            self._stream_task = asyncio.create_task(self._stream_server_output())

            self._http_client = httpx.AsyncClient(timeout=30.0)
            server_ready = False

            for i in range(600):
                try:
                    response = await self._http_client.get(
                        f"http://127.0.0.1:{self._server_port}/health", timeout=2.0
                    )
                    if response.status_code == 200:
                        # Verify OUR server is still alive before declaring ready.
                        # Another concurrent trial may have grabbed this port
                        # (Singularity shares host network namespace).
                        if self._server_process.returncode is not None:
                            self._stream_task.cancel()
                            last_error = RuntimeError(
                                f"Port collision on {self._server_port}: health "
                                f"check succeeded but our server process died."
                            )
                            self.logger.warning(
                                f"Health check succeeded but server process died "
                                f"- port {self._server_port} collision"
                            )
                            break
                        self.logger.debug("Singularity FastAPI server is ready")
                        if self._memory_limit_bytes is not None:
                            self._memory_watchdog_task = asyncio.create_task(
                                self._memory_watchdog()
                            )
                        server_ready = True
                        break
                except httpx.RequestError:
                    pass

                if self._server_process.returncode is not None:
                    self._stream_task.cancel()
                    last_error = RuntimeError(
                        f"Server process died on port {self._server_port}. "
                        f"Check trial.log for server output."
                    )
                    self.logger.warning(
                        f"Server failed to start on port {self._server_port}, "
                        f"will retry with new port"
                    )
                    break

                await asyncio.sleep(1)

            if server_ready:
                return

            # Clean up the failed server process before retrying
            if self._server_process and self._server_process.returncode is None:
                import signal
                os.killpg(self._server_process.pid, signal.SIGKILL)
                await self._server_process.wait()
            if self._stream_task and not self._stream_task.done():
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    pass

            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None

        raise last_error or RuntimeError(
            f"Failed to start Singularity FastAPI server after "
            f"{max_port_retries} port attempts"
        )

    async def _stream_server_output(self) -> None:
        """Stream server stdout/stderr to logger in real-time."""
        if not self._server_process or not self._server_process.stdout:
            return

        try:
            async for line in self._server_process.stdout:
                decoded = line.decode(errors="replace").rstrip()
                if decoded:
                    self.logger.debug(f"[server] {decoded}")
        except Exception as e:
            self.logger.debug(f"Server output stream ended: {e}")

    def _get_process_tree_memory(self, pid: int) -> int:
        """Get total PSS memory of a process and all its descendants.

        Uses PSS (Proportional Set Size) which properly accounts for shared
        memory by dividing it proportionally among sharing processes. Falls
        back to RSS if PSS is unavailable. Directly reads /proc for efficiency.
        Returns memory in bytes, or 0 if unable to read.
        """

        def get_all_descendants(root_pid: int) -> set[int]:
            pids = set()
            to_visit = [root_pid]

            while to_visit:
                current = to_visit.pop()
                if current in pids:
                    continue
                pids.add(current)

                try:
                    task_dir = Path(f"/proc/{current}/task")
                    for tid_dir in task_dir.iterdir():
                        children_file = tid_dir / "children"
                        try:
                            for child in children_file.read_text().split():
                                if child.isdigit():
                                    to_visit.append(int(child))
                        except (OSError, PermissionError):
                            pass
                except (OSError, PermissionError):
                    pass

            return pids

        def get_process_memory(p: int) -> int:
            # Try PSS from smaps_rollup (Linux 4.14+)
            try:
                for line in Path(f"/proc/{p}/smaps_rollup").read_text().splitlines():
                    if line.startswith("Pss:"):
                        return int(line.split()[1]) * 1024
            except (OSError, PermissionError, ValueError, IndexError):
                pass

            # Fallback to RSS from statm (second field, in pages)
            try:
                rss_pages = int(Path(f"/proc/{p}/statm").read_text().split()[1])
                return rss_pages * os.sysconf("SC_PAGE_SIZE")
            except (OSError, PermissionError, ValueError, IndexError):
                pass

            return 0

        try:
            return sum(get_process_memory(p) for p in get_all_descendants(pid))
        except Exception:
            return 0

    async def _memory_watchdog(self) -> None:
        """Monitor memory usage and kill container if it exceeds the limit.

        Features:
        - Adaptive intervals: checks every 1s when memory >50%, every 3s otherwise
        - Explosion detection: warns if growth rate would hit limit in <5s
        - Kill threshold at 95%: leaves headroom before actual OOM
        """
        if self._memory_limit_bytes is None:
            return

        base_interval = 3
        fast_interval = 1
        warning_threshold = 0.5
        kill_threshold = 0.95

        last_mem_usage = 0
        last_check_time = 0.0

        self.logger.debug(
            f"Memory watchdog started: "
            f"limit={self._memory_limit_bytes // 1024 // 1024}MB, "
            f"kill_at={kill_threshold * 100:.0f}%, "
            f"intervals={fast_interval}s/{base_interval}s"
        )

        while True:
            try:
                if (
                    not self._server_process
                    or self._server_process.returncode is not None
                ):
                    self.logger.debug("Memory watchdog: server process ended, stopping")
                    break

                current_time = asyncio.get_event_loop().time()
                mem_usage = self._get_process_tree_memory(self._server_process.pid)
                mem_mb = mem_usage / 1024 / 1024
                limit_mb = self._memory_limit_bytes / 1024 / 1024
                usage_pct = (
                    mem_usage / self._memory_limit_bytes
                    if self._memory_limit_bytes > 0
                    else 0
                )

                if last_check_time > 0 and last_mem_usage > 0:
                    time_delta = current_time - last_check_time
                    if time_delta > 0:
                        growth_rate = (mem_usage - last_mem_usage) / time_delta
                        if growth_rate > 0:
                            remaining_bytes = (
                                self._memory_limit_bytes * kill_threshold - mem_usage
                            )
                            time_to_limit = (
                                remaining_bytes / growth_rate
                                if growth_rate > 0
                                else float("inf")
                            )
                            if 0 < time_to_limit < 5:
                                self.logger.warning(
                                    f"Memory explosion detected: {mem_mb:.0f}MB, "
                                    f"growing {growth_rate / 1024 / 1024:.0f}MB/s, "
                                    f"~{time_to_limit:.1f}s until limit"
                                )

                last_mem_usage = mem_usage
                last_check_time = current_time

                if mem_usage > self._memory_limit_bytes * kill_threshold:
                    error_msg = (
                        f"Container exceeded memory limit "
                        f"({mem_mb:.0f}MB > "
                        f"{limit_mb * kill_threshold:.0f}MB)"
                    )
                    self.logger.error(
                        f"Memory limit exceeded: {mem_mb:.0f}MB > "
                        f"{limit_mb * kill_threshold:.0f}MB "
                        f"({usage_pct * 100:.0f}%). Killing container."
                    )
                    self._memory_limit_exceeded = error_msg
                    self._server_process.kill()
                    return

                interval = (
                    fast_interval if usage_pct > warning_threshold else base_interval
                )
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                self.logger.debug("Memory watchdog cancelled")
                raise
            except Exception as e:
                self.logger.debug(f"Memory watchdog error (continuing): {e}")
                await asyncio.sleep(base_interval)

    @override
    async def start(self, force_build: bool) -> None:
        """Start the Singularity environment."""
        if sys.platform == "win32":
            raise RuntimeError(
                "SingularityEnvironment is only supported on Linux/macOS. "
                "Singularity/Apptainer is not available on Windows."
            )
        if not self._docker_image:
            raise ValueError("docker_image must be set in task.toml [environment]")
        if self._is_sif_image:
            self._sif_path = Path(self._docker_image)
        else:
            force_pull = force_build or self._force_pull
            self._sif_path = await self._convert_docker_to_sif(
                self._docker_image, force_pull=force_pull
            )

        await self._start_server()
        await self.ensure_dirs(self._mount_targets(writable_only=True))

        await self._upload_environment_dir_after_start()

    @override
    async def stop(self, delete: bool) -> None:
        """Stop the Singularity environment and all child processes."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if self._memory_watchdog_task and not self._memory_watchdog_task.done():
            self._memory_watchdog_task.cancel()
            try:
                await self._memory_watchdog_task
            except asyncio.CancelledError:
                pass

        if self._server_process and self._server_process.returncode is None:
            pid = self._server_process.pid

            import signal
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.logger.debug(f"Sent SIGTERM to Singularity process group {pid}")

            try:
                await asyncio.wait_for(self._server_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.debug("Graceful shutdown timed out, force killing")
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self._server_process.wait()
            # The launcher can exit before its server; clean its isolated group.
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            try:
                subprocess.run(
                    ["pkill", "-9", "-P", str(pid)],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass

        if hasattr(self, "_stream_task") and self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._staging_dir and self._staging_dir.exists():
            shutil.rmtree(self._staging_dir, ignore_errors=True)

        if getattr(self, "_sandbox_parent", None):
            shutil.rmtree(self._sandbox_parent, ignore_errors=True)
        if delete:
            self.logger.debug(
                f"Singularity image preserved at {self._sif_path} for reuse"
            )

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute a command in the Singularity container via HTTP."""
        if not self._http_client or not self._server_port:
            raise RuntimeError("Singularity environment not started")

        if self._memory_limit_exceeded:
            raise MemoryLimitExceededError(self._memory_limit_exceeded)

        resolved_user = self._resolve_user(user)
        same_namespace_root = (self._sif_path.stem == 'qemu-alpine-ssh'
                               and resolved_user in (0, 'root'))
        # The server already runs as real namespace UID0. A redundant su tries
        # setgroups, which Linux deliberately denies in single-ID namespaces.
        if resolved_user is not None and not same_namespace_root:
            if isinstance(resolved_user, int):
                user_arg = f"$(getent passwd {resolved_user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(str(resolved_user))
            command = f"su {user_arg} -s /bin/bash -c {shlex.quote(command)}"

        merged_env = self._merge_env(env)

        try:
            # Harbor's outer trial timeout is authoritative when no command
            # timeout is specified. Do not add an unrelated 600s HTTP cutoff.
            http_timeout = (timeout_sec + 10) if timeout_sec is not None else None

            response = await self._http_client.post(
                f"http://localhost:{self._server_port}/exec",
                json={
                    "command": command,
                    "cwd": cwd,
                    "env": merged_env,
                    "timeout_sec": timeout_sec,
                },
                timeout=http_timeout,
            )
            response.raise_for_status()
            result = response.json()

            exec_result = ExecResult(
                stdout=result.get("stdout"),
                stderr=result.get("stderr"),
                return_code=result.get("return_code", 1),
            )

            if exec_result.return_code != 0:
                error_output = exec_result.stderr or exec_result.stdout or "<no output>"
                self.logger.debug(
                    f"Command exited with rc={exec_result.return_code}: {error_output}"
                )

            return exec_result

        except httpx.TimeoutException:
            if self._memory_limit_exceeded:
                raise MemoryLimitExceededError(self._memory_limit_exceeded)
            raise asyncio.TimeoutError(
                f"HTTP request timed out after {http_timeout} seconds"  # ty: ignore[possibly-unresolved-reference]
                if http_timeout  # ty: ignore[possibly-unresolved-reference]
                else "HTTP request timed out"
            )
        except (httpx.ConnectError, httpx.RemoteProtocolError):
            if self._memory_limit_exceeded:
                raise MemoryLimitExceededError(self._memory_limit_exceeded)
            raise

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Upload a file to the container via staging directory."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        staging_file = self._staging / source.name
        shutil.copy2(source, staging_file)

        result = await self.exec(
            f"cp {shlex.quote('/staging/' + source.name)} {shlex.quote(target_path)}"
        )
        staging_file.unlink(missing_ok=True)
        if result.return_code != 0:
            error_output = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"Failed to upload file: {error_output}")

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Upload a directory to the container via staging directory."""
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")

        staging_subdir = self._staging / source.name
        if staging_subdir.exists():
            shutil.rmtree(staging_subdir)
        shutil.copytree(source, staging_subdir)

        await self.exec(f"mkdir -p {shlex.quote(target_dir)}")
        result = await self.exec(
            f"cp -r /staging/{shlex.quote(source.name)}/. {shlex.quote(target_dir)}/"
        )
        if result.return_code != 0:
            result = await self.exec(
                f"cp -r {shlex.quote('/staging/' + source.name)} {shlex.quote(target_dir)}"
            )
            if result.return_code != 0:
                shutil.rmtree(staging_subdir, ignore_errors=True)
                error_output = result.stderr or result.stdout or "<no output>"
                raise RuntimeError(f"Failed to upload directory: {error_output}")
        shutil.rmtree(staging_subdir, ignore_errors=True)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download a file from the container via staging directory."""
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        filename = Path(source_path).name
        staging_path = f"/staging/download_{filename}"
        result = await self.exec(
            f"cp {shlex.quote(source_path)} {shlex.quote(staging_path)}"
        )
        if result.return_code != 0:
            error_output = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"Failed to download file: {error_output}")

        staging_file = self._staging / f"download_{filename}"
        if staging_file.exists():
            shutil.copy2(staging_file, target)
            staging_file.unlink()
        else:
            raise RuntimeError(f"File not found in staging: {staging_file}")

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a directory from the container via staging directory."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        dirname = Path(source_dir).name
        staging_path = f"/staging/download_{dirname}"
        result = await self.exec(
            f"cp -r {shlex.quote(source_dir)} {shlex.quote(staging_path)}"
        )
        if result.return_code != 0:
            error_output = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"Failed to download directory: {error_output}")

        staging_subdir = self._staging / f"download_{dirname}"
        if staging_subdir.exists():
            for item in staging_subdir.iterdir():
                dest = target / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            shutil.rmtree(staging_subdir)
        else:
            raise RuntimeError(f"Directory not found in staging: {staging_subdir}")
