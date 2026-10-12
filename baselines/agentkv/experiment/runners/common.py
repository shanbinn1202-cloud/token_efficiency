from __future__ import annotations

import atexit
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import dataclasses
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "experiment" / "runs"
FLASHINFER_WORKSPACE_BASE = Path(
    os.environ.get("AGENTKV_FLASHINFER_WORKSPACE_BASE", "/tmp/agentkv_flashinfer_minisgl_profile")
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str  # HF repo id (e.g. "Qwen/Qwen3-8B"); resolved to a local snapshot if available.
    bfcl_model: str
    tau2_agent_llm: str


# Registry keyed by short name -> HF repo id. The engine accepts an HF repo id directly
# (auto-download). To reuse an existing local HF cache instead of downloading, set
# AGENTKV_MODEL_DIR to a HuggingFace cache dir that contains
# `hub/models--<Org>--<Name>/snapshots/<rev>` (see resolve_model_path below).
MODELS = {
    "qwen3-8b": ModelSpec(
        name="qwen3-8b",
        path="Qwen/Qwen3-8B",
        bfcl_model="qwen3-8b-minisglang-prompt",
        tau2_agent_llm="openai/qwen3-8b-minisglang",
    ),
    "qwen3-32b": ModelSpec(
        name="qwen3-32b",
        path="Qwen/Qwen3-32B",
        bfcl_model="qwen3-32b-minisglang-prompt",
        tau2_agent_llm="openai/qwen3-32b-minisglang",
    ),
}


def resolve_model_path(hf_repo: str) -> str:
    """Return a local snapshot dir for `hf_repo` if AGENTKV_MODEL_DIR points at an HF cache
    that already has it; otherwise return the HF repo id unchanged (engine will download)."""
    base = os.environ.get("AGENTKV_MODEL_DIR")
    if base:
        snaps = Path(base) / "hub" / ("models--" + hf_repo.replace("/", "--")) / "snapshots"
        if snaps.is_dir():
            revs = sorted(p for p in snaps.iterdir() if p.is_dir())
            if revs:
                return str(revs[-1])
    return hf_repo


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def slugify_name(value: str) -> str:
    chars = []
    previous_sep = False
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
            previous_sep = False
        elif not previous_sep:
            chars.append("_")
            previous_sep = True
    return "".join(chars).strip("_") or "unnamed"


def auto_job_name(*parts: str) -> str:
    return "_".join([timestamp(), *(slugify_name(part) for part in parts)])


def model_spec(name: str) -> ModelSpec:
    try:
        spec = MODELS[name]
    except KeyError:
        raise RuntimeError(f"unknown model name: {name}. expected one of {sorted(MODELS)}")
    return dataclasses.replace(spec, path=resolve_model_path(spec.path))


def write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def ensure_env_file() -> Path:
    env_file = ROOT / ".env"
    if not env_file.exists():
        raise RuntimeError(f"missing required env file: {env_file}")
    return env_file


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _set_pdeathsig() -> None:
    """Run in the child after fork: ask the kernel to SIGKILL us when the launching parent
    dies, covering the case where the parent is SIGKILL'd (so atexit/killpg never run).
    This only protects the direct child (the minisgl launcher); its own workers are
    grandchildren and rely on the launcher's teardown or the process-group kill."""
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    if os.getppid() == 1:  # parent already died before prctl took effect
        os._exit(1)


def _stop_process_group(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    print(f"[{name}] stopping pid={proc.pid}")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        print(f"[{name}] force stopping pid={proc.pid}")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def run_logged(
    cmd: list[str],
    log_path: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run `cmd`, streaming its output to stdout and `log_path`."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            return proc.wait()
        except BaseException:
            _stop_process_group(proc, "workload")
            raise


_ACTIVE_SERVERS: set["MiniSGLangServer"] = set()
_ACTIVE_LOCK = threading.RLock()
_STOPPING_ALL = False


class MiniSGLangServer:
    def __init__(
        self,
        *,
        model_name: str,
        log_path: Path,
        code_root: Path | None = None,
        cuda_visible_devices: str | None = None,
        memory_ratio: str = "0.75",
        max_running_requests: str = "16",
        cuda_graph_max_bs: str = "16",
        max_seq_len: str = "40960",
        max_extend_tokens: str | None = None,
        page_size: str | None = None,
        attention_backend: str = "fi",
        eviction_plan: str = "none",
        compact_interval: str = "128",
        compact_after_prefill: bool = False,
        compact_before_park: bool = False,
        profile_compact: bool = False,
        compact_debug_artifacts: bool = False,
        compact_debug_dir: Path | None = None,
    ):
        self.spec = model_spec(model_name)
        self.log_path = log_path
        self.code_root = code_root or ROOT
        self.cuda_visible_devices = cuda_visible_devices
        self.memory_ratio = memory_ratio
        self.max_running_requests = max_running_requests
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.max_seq_len = max_seq_len
        self.max_extend_tokens = max_extend_tokens or max_seq_len
        self.page_size = page_size
        self.attention_backend = attention_backend
        self.eviction_plan = eviction_plan
        self.compact_interval = compact_interval
        self.compact_after_prefill = compact_after_prefill
        self.compact_before_park = compact_before_park
        self.profile_compact = profile_compact
        self.compact_debug_artifacts = compact_debug_artifacts
        self.compact_debug_dir = compact_debug_dir or (log_path.parent / "compact_debug")
        self.port: int | None = None
        self.proc: subprocess.Popen[str] | None = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise RuntimeError("mini-sglang server has not started")
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> None:
        if self.proc is not None:
            return
        # Fixed, deterministic port (no auto-allocation). Offset by the first visible GPU so
        # runs on different GPUs don't collide; override the base with MINISGLANG_PORT.
        base = int(os.environ.get("MINISGLANG_PORT", "30000"))
        gpu = (self.cuda_visible_devices or "0").split(",")[0]
        self.port = base + (int(gpu) * 8 if gpu.isdigit() else 0)

        env = os.environ.copy()
        env.setdefault("FLASHINFER_WORKSPACE_BASE", str(FLASHINFER_WORKSPACE_BASE))
        python_bin_dir = str(Path(sys.executable).resolve().parent)
        env["PATH"] = python_bin_dir + (":" + env["PATH"] if env.get("PATH") else "")
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        minisglang_repo = self.code_root / "engine"
        py_path = str(minisglang_repo / "python")
        env["PYTHONPATH"] = py_path + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "minisgl",
            "--model-path",
            self.spec.path,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--max-running-requests",
            self.max_running_requests,
            "--cuda-graph-max-bs",
            self.cuda_graph_max_bs,
            "--max-seq-len-override",
            self.max_seq_len,
            "--max-extend-length",
            self.max_extend_tokens,
            "--memory-ratio",
            self.memory_ratio,
            "--attention-backend",
            self.attention_backend,
            "--eviction-plan",
            self.eviction_plan,
            "--compact-interval",
            self.compact_interval,
            "--compact-after-prefill" if self.compact_after_prefill else "--no-compact-after-prefill",
            "--compact-before-park" if self.compact_before_park else "--no-compact-before-park",
            "--compact-debug-dir",
            str(self.compact_debug_dir),
        ]
        if self.page_size is not None:
            cmd.extend(["--page-size", self.page_size])
        if self.profile_compact:
            cmd.append("--compact-debug-jsonl")
        if self.compact_debug_artifacts:
            cmd.append("--compact-debug")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("a", encoding="utf-8", errors="replace")
        self._log_file.write(f"$ {' '.join(cmd)}\n")
        self._log_file.flush()
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(minisglang_repo),
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                preexec_fn=_set_pdeathsig,
            )
        except BaseException:
            self._close_log()
            raise
        _register_server(self)
        try:
            self._wait_ready()
        except BaseException:
            self.stop()
            raise

    def _wait_ready(self, timeout_s: int = 900) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"mini-sglang exited during startup; see {self.log_path}")
            if self._server_matches_model():
                print(f"[minisglang] ready: {self.base_url} model={self.spec.name}")
                return
            time.sleep(2)
        raise RuntimeError(f"mini-sglang failed to start; see {self.log_path}")

    def _server_matches_model(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/models", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return False
        for model in payload.get("data", []):
            if model.get("id") == self.spec.path or model.get("root") == self.spec.path:
                return True
        return False

    def stop(self) -> None:
        proc = self.proc
        try:
            if proc is not None:
                _stop_process_group(proc, "minisglang")
        finally:
            self.proc = None
            _unregister_server(self)
            self._close_log()

    def _close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "MiniSGLangServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _register_server(server: MiniSGLangServer) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_SERVERS.add(server)


def _unregister_server(server: MiniSGLangServer) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_SERVERS.discard(server)


def stop_all_servers() -> None:
    global _STOPPING_ALL
    with _ACTIVE_LOCK:
        if _STOPPING_ALL:
            return
        _STOPPING_ALL = True
        servers = list(_ACTIVE_SERVERS)
    for server in reversed(servers):
        try:
            server.stop()
        except Exception as exc:
            print(f"[minisglang] failed to stop cleanly: {exc}", file=sys.stderr)
    with _ACTIVE_LOCK:
        _STOPPING_ALL = False


def _handle_signal(signum: int, _frame: object) -> None:
    stop_all_servers()
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


atexit.register(stop_all_servers)
if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
