"""Apptainer environments and architecture-aware instance image resolution."""

import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from minisweagent.environments.singularity import SingularityEnvironment


def check_apptainer(executable: str) -> None:
    if shutil.which(executable) is None:
        raise RuntimeError(
            f"Apptainer executable {executable!r} was not found. Load your cluster's "
            "apptainer/singularity module or set evaluation.apptainer_executable."
        )


def native_architecture() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    raise RuntimeError(f"Unsupported native architecture: {machine}")


def check_swebench_platform() -> None:
    native_architecture()


class SwebenchApptainerEnvironment(SingularityEnvironment):
    """Activate the image's Python environment for each independent bash call.

    Upstream Singularity executes bash -c, unlike Docker's bash -lc. Explicit
    activation is needed because --containall hides the image's /root/.bashrc.
    Each environment owns a separate writable sandbox, including /testbed.
    """

    def __init__(self, *, activate_testbed: bool = True, **kwargs):
        self.activate_testbed = activate_testbed
        super().__init__(**kwargs)
        # --containall otherwise gives each exec a fresh temporary home and
        # /tmp. Agents commonly save reproducers there between tool calls.
        # Bind only this sandbox's directories, never the user's host home.
        for relative in ("tmp", "var/tmp", "root"):
            directory = self.sandbox_dir / relative
            directory.mkdir(parents=True, exist_ok=True)
            self.config.exec_args.extend(["--bind", f"{directory}:/{relative}"])
        # OCI images may leave /etc/hosts empty for Docker to populate at run
        # time. This isolated writable Apptainer sandbox needs the same local
        # loopback names (e.g. pytest-rerunfailures binds a localhost socket).
        # Update only the private sandbox, preserving existing image entries.
        hosts = self.sandbox_dir / "etc/hosts"
        contents = hosts.read_text() if hosts.exists() else ""
        aliases = {
            alias
            for line in contents.splitlines()
            for alias in line.split("#", 1)[0].split()[1:]
        }
        if "localhost" not in aliases:
            hosts.write_text(
                contents.rstrip("\n")
                + "\n127.0.0.1 localhost\n::1 localhost ip6-localhost ip6-loopback\n"
            )

    def _build_sandbox(self) -> Path:
        image = Path(self.config.image)
        if not image.is_file():
            return super()._build_sandbox()
        # Apptainer's bundled unsquashfs spawns one thread per host CPU. Eight
        # workers on this 144-core machine exhaust the thread limit. Extract
        # the SIF system partition with two threads and bounded queues instead.
        listing = subprocess.check_output(
            [self.config.executable, "sif", "list", str(image)], text=True
        )
        partitions = [
            line.split("|")
            for line in listing.splitlines()
            if "FS (Squashfs/*System/" in line
        ]
        if len(partitions) != 1:
            raise ValueError(f"Expected one SquashFS system partition in {image}")
        partition = partitions[0]
        arch = "arm64" if native_architecture() == "arm64" else "amd64"
        if f"System/{arch})" not in partition[-1]:
            raise ValueError(f"SIF architecture does not match native {arch}: {image}")
        offset = int(partition[3].strip().split("-", 1)[0])
        sandbox = Path(tempfile.mkdtemp(prefix="token-efficiency-")) / "rootfs"
        try:
            subprocess.run(
                [
                    "unsquashfs",
                    "-processors",
                    "2",
                    "-data-queue",
                    "16",
                    "-frag-queue",
                    "16",
                    "-no-xattrs",
                    "-no-progress",
                    "-offset",
                    str(offset),
                    "-dest",
                    str(sandbox),
                    str(image),
                ],
                check=True,
                capture_output=True,
                timeout=1800,
            )
        except BaseException:
            shutil.rmtree(sandbox.parent, ignore_errors=True)
            raise
        return sandbox

    def execute(
        self, action: dict, cwd: str = "", *, timeout: int | None = None
    ) -> dict:
        if not self.activate_testbed:
            return super().execute(action, cwd, timeout=timeout)
        command = (
            "source /opt/miniconda3/bin/activate testbed || exit $?\n"
            + action.get("command", "")
        )
        return super().execute({**action, "command": command}, cwd, timeout=timeout)

    def cleanup(self):
        # Upstream __del__ also calls cleanup after a failed sandbox build.
        if getattr(self, "sandbox_dir", None) is not None:
            super().cleanup()
            if (
                self.sandbox_dir.name == "rootfs"
                and self.sandbox_dir.parent.name.startswith("token-efficiency-")
                and self.sandbox_dir.parent.exists()
            ):
                self.sandbox_dir.parent.rmdir()


def create_apptainer_environment(
    image_name: str,
    *,
    executable: str = "apptainer",
    timeout: int = 60,
    activate_testbed: bool = True,
) -> SwebenchApptainerEnvironment:
    check_apptainer(executable)
    # docker:// names an OCI registry transport; it does not invoke Docker.
    if Path(image_name).exists():
        image = str(Path(image_name).resolve())
    else:
        image = image_name if "://" in image_name else f"docker://{image_name}"
    return SwebenchApptainerEnvironment(
        image=image,
        executable=executable,
        cwd="/testbed",
        timeout=timeout,
        activate_testbed=activate_testbed,
        env={
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        # Avoid injecting the host's libfakeroot into older Ubuntu images (glibc
        # incompatibility). The root-mapped user namespace still supplies uid 0.
        exec_args=[
            "--containall",
            "--cleanenv",
            "--fakeroot",
            "--ignore-fakeroot-command",
            "--no-mount",
            "bind-paths",
        ],
    )


def resolve_instance_image(
    instance: dict, default_image: str, manifest_path: str = ""
) -> str:
    arch = native_architecture()
    if manifest_path:
        path = Path(manifest_path)
        if not path.is_file():
            raise RuntimeError(
                f"Missing native image manifest: {path}. Run "
                "python -m baselines.common.prepare first."
            )
        manifest = json.loads(path.read_text())
        if manifest["architecture"] != arch:
            raise ValueError(
                f"Image manifest architecture {manifest['architecture']} != host {arch}"
            )
        entry = manifest["images"].get(instance["instance_id"])
        if entry is None:
            raise ValueError(
                f"Instance {instance['instance_id']} is missing from {path}"
            )
        if entry["base_commit"] != instance["base_commit"]:
            raise ValueError(
                f"Image base commit mismatch for {instance['instance_id']}"
            )
        image = entry["image"]
        if "://" not in image and not Path(image).is_file():
            raise FileNotFoundError(f"Prepared instance image missing: {image}")
        return image
    if arch == "arm64":
        return f"docker://ghcr.io/epoch-research/swe-bench.eval.arm64.{instance['instance_id'].lower()}:latest"
    return default_image
