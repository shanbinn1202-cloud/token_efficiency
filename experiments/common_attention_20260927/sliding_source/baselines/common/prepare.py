"""Prepare native SWE-bench images without a Docker daemon.

Epoch AI publishes native ARM64 images for modern Astropy instances. The four
Astropy 1.3 instances in Verified's first 20 need a local Python 3.6 environment.
This builder preserves Python and pip-package versions from the pinned harness,
replacing x86-specific Conda build strings with native conda-forge packages.
It never introduces reference code/test patches into agent images.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from datasets import load_dataset
from swebench.harness.test_spec.test_spec import make_test_spec

from baselines.common.concurrency import iter_parallel_completed
from baselines.common.environment import (
    check_apptainer,
    create_apptainer_environment,
    native_architecture,
)
from baselines.common.settings import DEFAULT_CONFIG_PATH, Config

logger = logging.getLogger(__name__)
NATIVE_BASE = (
    "docker://ghcr.io/epoch-research/swe-bench.base.arm64@"
    "sha256:d640de244096fd78c554d0320f14c1d637f85878a13cbd8c2f1c196d12f4420c"
)


def epoch_image(instance_id: str, arch: str) -> str:
    """Verify registry access and architecture, then pin the immutable digest."""
    name = f"epoch-research/swe-bench.eval.{arch}.{instance_id.lower()}"
    with urllib.request.urlopen(
        f"https://ghcr.io/token?scope=repository:{name}:pull", timeout=60
    ) as r:
        token = json.load(r)["token"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.docker.distribution.manifest.v2+json",
    }
    request = urllib.request.Request(
        f"https://ghcr.io/v2/{name}/manifests/latest", headers=headers
    )
    with urllib.request.urlopen(request, timeout=60) as r:
        manifest = json.load(r)
        digest = r.headers["Docker-Content-Digest"]
    request = urllib.request.Request(
        f"https://ghcr.io/v2/{name}/blobs/{manifest['config']['digest']}",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=60) as r:
        config = json.load(r)
    expected = "arm64" if arch == "arm64" else "amd64"
    if config["architecture"] != expected:
        raise ValueError(
            f"Registry architecture mismatch: {instance_id}: {config['architecture']}"
        )
    return f"docker://ghcr.io/{name}@{digest}"


def build_sif(executable: str, source: str, target: Path, log: Path) -> None:
    """Publish a completed SIF atomically, retaining command output on failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".build-") as work:
        partial = Path(work) / "image.sif"
        with log.open("w") as output:
            subprocess.run(
                [
                    executable,
                    "build",
                    "--mksquashfs-args",
                    "-processors 2 -mem 256M",
                    str(partial),
                    source,
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=7200,
                # Concurrent OCI writes to one Apptainer cache can corrupt its
                # layout index. Keep a resumable cache per immutable image.
                env={
                    **os.environ,
                    "GOMAXPROCS": "2",
                    "APPTAINER_CACHEDIR": str(
                        Path(
                            os.environ.get(
                                "APPTAINER_CACHEDIR", str(target.parent / "cache")
                            )
                        )
                        / hashlib.sha256(source.encode()).hexdigest()[:16]
                    ),
                },
            )
        partial.replace(target)


def legacy_setup_script(instance: dict) -> str:
    if instance["repo"] != "astropy/astropy" or instance["version"] != "1.3":
        raise ValueError(f"No native local build recipe for {instance['instance_id']}")
    official = make_test_spec(instance, arch="arm64").setup_env_script
    environment_yaml = official.split(" > /root/environment.yml\n", 1)[1].split(
        "\nEOF_", 1
    )[0]
    dependencies = yaml.safe_load(environment_yaml)["dependencies"]
    pins = next(entry["pip"] for entry in dependencies if isinstance(entry, dict))
    requirements = "\n".join(pins)
    return f"""#!/bin/bash
set -euxo pipefail
source /opt/miniconda3/bin/activate
conda create -y --override-channels -c conda-forge -n testbed python=3.6.13 pip
conda activate testbed
python -m pip install pip==21.2.2 setuptools==38.2.4 wheel==0.37.1
# Old setup.py/easy_install otherwise fetches a current Python>=3.7-only SCM helper.
python -m pip install setuptools_scm==4.1.2
# These source-build prerequisites must precede pyerfa's metadata generation.
python -m pip install --no-build-isolation numpy==1.16.0 cython==0.27.3 jinja2==2.10 markupsafe==1.0
cat > /coact-requirements.txt <<'COACT_REQUIREMENTS'
{requirements}
COACT_REQUIREMENTS
python -m pip install --no-build-isolation -r /coact-requirements.txt
python -m pip freeze > /coact-native-pip-freeze.txt
conda list --explicit > /coact-native-conda-spec.txt
"""


def repo_setup_script(instance: dict) -> str:
    """Fetch only the task's base commit; no future commits or reference patches."""
    if instance["repo"] != "astropy/astropy" or not re.fullmatch(
        r"[0-9a-f]{40}", instance["base_commit"]
    ):
        raise ValueError("Unsupported native repository build")
    official = make_test_spec(instance, arch="arm64").install_repo_script
    install = (
        "source /opt/miniconda3/bin/activate"
        + official.split("source /opt/miniconda3/bin/activate", 1)[1]
    )
    return f"""#!/bin/bash
set -euxo pipefail
git init /testbed
cd /testbed
git remote add origin https://github.com/astropy/astropy
git fetch --depth 1 origin {instance["base_commit"]}
git checkout --detach FETCH_HEAD
git remote remove origin
{install}
"""


def execute_setup(env, script: str, log: Path) -> None:
    (env.sandbox_dir / "coact_setup.sh").write_text(script)
    env.config.cwd = "/"
    result = env.execute({"command": "bash /coact_setup.sh"}, timeout=7200)
    log.write_text(result["output"])
    if result["returncode"] != 0 or result.get("exception_info"):
        raise RuntimeError(
            f"Native image setup failed: {log}: {result.get('exception_info', '')}"
        )
    (env.sandbox_dir / "coact_setup.sh").unlink()


def prepare_legacy_env(instance: dict, root: Path, executable: str) -> Path:
    target = root / "astropy-1.3-env.sif"
    script = legacy_setup_script(instance)
    recipe_path = root / "astropy-1.3-env.sh"
    if target.is_file() and recipe_path.is_file() and recipe_path.read_text() == script:
        return target
    logger.info(
        "Building native Astropy 1.3 dependencies; initial source compilation takes several minutes"
    )
    env = create_apptainer_environment(
        NATIVE_BASE, executable=executable, activate_testbed=False
    )
    try:
        execute_setup(env, script, root / "astropy-1.3-env.log")
        build_sif(
            executable, str(env.sandbox_dir), target, root / "astropy-1.3-env-build.log"
        )
        recipe_path.write_text(script)
    finally:
        env.cleanup()
    return target


def prepare(config: Config, max_instances: int, workers: int) -> Path:
    arch = native_architecture()
    executable = config.evaluation.apptainer_executable
    check_apptainer(executable)
    manifest_path = Path(config.evaluation.apptainer_image_manifest).resolve()
    root = manifest_path.parent
    root.mkdir(parents=True, exist_ok=True)
    rows = list(load_dataset(config.data.eval_dataset, split=config.data.eval_split))[
        :max_instances
    ]
    if not rows:
        raise ValueError("No instances selected")
    old = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if old and old.get("architecture") != arch:
        raise ValueError("Use a separate manifest path for each CPU architecture")

    def resolve(row):
        instance_id = row["instance_id"]
        cached = old.get("images", {}).get(instance_id)
        if (
            cached
            and cached["base_commit"] == row["base_commit"]
            and Path(cached["image"]).is_file()
        ):
            return row, cached
        try:
            return row, epoch_image(instance_id, arch)
        except urllib.error.HTTPError as exc:
            if (
                exc.code in {403, 404}
                and arch == "arm64"
                and row["repo"] == "astropy/astropy"
                and row["version"] == "1.3"
            ):
                return row, None
            raise

    resolved = list(iter_parallel_completed(rows, resolve, workers))
    legacy_rows = [row for row, source in resolved if source is None]
    legacy_env = (
        prepare_legacy_env(legacy_rows[0], root, executable) if legacy_rows else None
    )

    def materialize(item):
        row, source = item
        if isinstance(source, dict):
            return row["instance_id"], source
        target = root / f"{row['instance_id']}.sif"
        if source is not None:
            logger.info("Pulling native %s image: %s", arch, row["instance_id"])
            build_sif(executable, source, target, target.with_suffix(".build.log"))
        else:
            logger.info("Building native repository: %s", row["instance_id"])
            env = create_apptainer_environment(
                str(legacy_env), executable=executable, activate_testbed=False
            )
            try:
                script = repo_setup_script(row)
                target.with_suffix(".setup.sh").write_text(script)
                execute_setup(env, script, target.with_suffix(".setup.log"))
                build_sif(
                    executable,
                    str(env.sandbox_dir),
                    target,
                    target.with_suffix(".build.log"),
                )
            finally:
                env.cleanup()
        return row["instance_id"], {
            "image": str(target),
            "base_commit": row["base_commit"],
            "source": source or "local-native-astropy-1.3",
        }

    manifest = {
        "architecture": arch,
        "dataset": config.data.eval_dataset,
        "split": config.data.eval_split,
        "instance_ids": [r["instance_id"] for r in rows],
        "images": {},
    }
    errors = []

    def attempt(item):
        try:
            return materialize(item)
        except Exception:
            logger.exception("Image preparation failed: %s", item[0]["instance_id"])
            return item[0]["instance_id"], None

    for instance_id, entry in iter_parallel_completed(resolved, attempt, workers):
        if entry is None:
            errors.append(instance_id)
            continue
        manifest["images"][instance_id] = entry
        # Partial progress is resumable; evaluation still checks every selected ID.
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(manifest_path)
    if errors:
        raise RuntimeError(
            f"Images failed to prepare: {errors}. Rerun to reuse completed images."
        )
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-instances", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Grade reference patches in fresh native images after preparation",
    )
    args = parser.parse_args()
    if args.max_instances <= 0 or args.workers <= 0:
        parser.error("--max-instances and --workers must be positive")
    config = Config.load(args.config)
    if not config.evaluation.apptainer_image_manifest:
        parser.error("Set evaluation.apptainer_image_manifest in the configuration")
    scratch = Path(tempfile.gettempdir()) / "token-efficiency-build"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("APPTAINER_TMPDIR", str(scratch))
    os.environ.setdefault("GOMAXPROCS", "2")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    print(prepare(config, args.max_instances, args.workers))
    if args.validate:
        from baselines.common.harness import run_apptainer_harness

        rows = list(
            load_dataset(config.data.eval_dataset, split=config.data.eval_split)
        )[: args.max_instances]
        output = (
            Path(config.evaluation.apptainer_image_manifest).parent / "gold-validation"
        )
        output.mkdir(parents=True, exist_ok=True)
        predictions = [
            {
                "instance_id": row["instance_id"],
                "model_name_or_path": "gold-environment-check",
                "model_patch": row["patch"],
            }
            for row in rows
        ]
        (output / "predictions.json").write_text(json.dumps(predictions, indent=2))
        path = run_apptainer_harness(
            predictions=predictions,
            dataset_name=config.data.eval_dataset,
            split=config.data.eval_split,
            instance_ids=[row["instance_id"] for row in rows],
            max_workers=args.workers,
            timeout=config.evaluation.instance_timeout,
            run_id="native-gold-validation",
            report_path=output / "report.json",
            executable=config.evaluation.apptainer_executable,
            image_manifest=config.evaluation.apptainer_image_manifest,
        )
        report = json.loads(path.read_text())
        print(json.dumps(report, indent=2))
        if report["resolved_instances"] != len(rows):
            raise RuntimeError(
                f"Reference patches did not all pass. Inspect {output} before model evaluation."
            )


if __name__ == "__main__":
    main()
