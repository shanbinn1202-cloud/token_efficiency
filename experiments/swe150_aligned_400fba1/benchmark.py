"""Run one SWE-150 allocation. The model and private ARM64 testbeds run on the assigned node."""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import shutil
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
POLICIES = {"random": "random_fill", "sliding_window": "recent_fill"}
METHODS = [*POLICIES, "agentkv", "claude_compression"]
@contextmanager
def local_port():
    # Independent GPU allocations sometimes share a host and its loopback network.
    job = int(os.environ["SLURM_JOB_ID"])
    for offset in range(128):
        port = 20000 + (job + offset) % 10000
        with open(f"/tmp/ssun4-swe150-port-{port}.lock", "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            yield port
            return
    raise RuntimeError("Could not reserve a local inference port")


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def native_server(method, local, out, port):
    return [sys.executable, "-B", str(ROOT / "source/serve.py"),
            "--release-dir", str(local / "engines/goal"),
            "--task", str(ROOT / "assets/tasks/tasks-pool.json"),
            "--policy", POLICIES[method], "--protocol", "mswea-xml",
            "--output", str(out / "server"), "--port", str(port),
            "--kv-cache-gib", "25", "--max-model-len", "262144",
            "--max-running-sessions", "24", "--max-tokens", "4096",
            "--page-tokens", "16", "--kv-layout", "decoupled",
            "--dram-arena-gib", "68", "--connector-pool-gib", "1",
            "--active-token-budget", "8192", "--cuda-graphs", "full",
            "--decode-fastpath", "--decode-fastpath2", "--decode-fastpath3",
            "--deferred-completion-io", "--language-model-only", "--enable-thinking",
            "--async-scheduling", "--release-idle-pages", "--start"]


def allocation_receipt(out):
    job = os.environ["SLURM_JOB_ID"]
    details = subprocess.check_output(["scontrol", "-o", "show", "job", job], text=True)
    fields = dict(item.split("=", 1) for item in details.split() if "=" in item)
    allocated = dict(pair.split("=", 1) for pair in fields["AllocTRES"].split(","))
    match = re.fullmatch(r"(\d+)([KMGT]?)", allocated.get("mem", ""), re.IGNORECASE)
    memory = None if not match else int(match[1]) * {
        "": 2**20, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}[match[2].upper()]
    receipt = {"job_id": job, "requested": fields["ReqTRES"], "allocated": allocated,
               "node": fields["NodeList"], "allocated_memory_bytes": memory}
    write(out / "allocation.json", receipt)
    if (allocated.get("cpu") != "32" or allocated.get("node") != "1"
            or allocated.get("gres/gpu") != "1" or memory is None or memory < 145 * 2**30):
        raise RuntimeError(f"Expected a single-GPU 32-core, 145-GiB allocation: {allocated}")


def slurm_step(role, command, out, scratch):
    """Request separate CPU, memory and GPU cgroups for each side of the benchmark."""
    engine = role == "engine"
    resources = ["--mem=113560M", "--gpus-per-node=1"] if engine else ["--mem=30G", "--gres=none"]
    launcher = shutil.which("srun")
    if launcher is None:
        raise RuntimeError("Slurm srun is not available on this node")
    args = [launcher, "--exact", "--nodes=1", "--ntasks=1", "--cpus-per-task=16",
            "--cpu-bind=cores", "--mpi=none", "--export=ALL", *resources,
            sys.executable, "-B", str(ROOT / "placement.py"), "--role", role,
            "--receipt", str(out / (role + "-placement.json")), "--scratch", str(scratch)]
    if role in ("agent", "probe"):
        args.extend(["--peer", str(out / "engine-placement.json")])
    args.append("--")
    return args + list(map(str, command))


def step_environment():
    # The batch allocation requested one GPU, 32 cores and 145 GiB. srun must use
    # each step's own mutually exclusive resource options, including for CPU-only steps.
    exclude = {"SLURM_CPUS_PER_TASK", "SLURM_TRES_PER_TASK", "SLURM_TRES_BIND",
               "SLURM_GRES", "SLURM_GRES_FLAGS", "SLURM_GPU_BIND", "SLURM_GPUS",
               "SLURM_GPUS_PER_TASK", "SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_SOCKET",
               "SLURM_CPUS_PER_GPU", "SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU",
               "SLURM_MEM_PER_NODE", "SLURM_EXACT", "SLURM_OVERLAP",
               "SLURM_CPU_BIND", "SLURM_CPU_BIND_LIST", "SLURM_CPU_BIND_TYPE"}
    return {k: v for k, v in os.environ.items() if k not in exclude}


@contextmanager
def process(cmd, log, receipt):
    with log.open("xb") as stream:
        proc = subprocess.Popen(list(map(str,cmd)), stdout=stream, stderr=subprocess.STDOUT,
                                env=step_environment(), start_new_session=True)
        try:
            yield proc
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    if receipt.is_file():
                        step = json.loads(receipt.read_text())
                        if (step["role"] == "engine" and step["job_id"] == os.environ["SLURM_JOB_ID"]
                                and str(step["step_id"]).isdigit()):
                            subprocess.run(["scancel", "--signal=KILL", step["job_id"] + "." + str(step["step_id"])],
                                           check=False)
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def ready(proc, port, suffix="/v1/models"):
    url = f"http://127.0.0.1:{port}{suffix}"
    deadline = time.monotonic()+1200
    while time.monotonic()<deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with {proc.returncode} before {suffix} was ready")
        try:
            with urlopen(url, timeout=5) as reply:
                if reply.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server did not become ready at {url}")


def validate(method,out,port):
    with urlopen(f"http://127.0.0.1:{port}/server_info?config_format=json",timeout=60) as reply:
        cfg=json.load(reply)["vllm_config"]
    checks={"model_config":{"max_model_len":262144,"seed":42},
            "cache_config":{"kv_cache_memory_bytes":25*1024**3},
            "scheduler_config":{"max_num_seqs":24,"max_num_batched_tokens":16384,
                                "async_scheduling":True}}
    evidence={section:{k:cfg[section].get(k) for k in fields} for section,fields in checks.items()}
    write(out/"effective-config.json",evidence)
    if evidence!=checks:
        raise ValueError(f"Unexpected runtime settings: {evidence}")
    if method in POLICIES:
        startup=json.loads((out/"server/startup.json").read_text())
        expected_ids=[s.split()[0] for s in (ROOT/"source/examples/swebench/verified-150.txt").read_text().splitlines() if s.strip()]
        assert [x["instance_id"] for x in startup["tasks"]]==expected_ids
        assert startup["selection_contract"]["active_token_budget"]==8192
        assert startup["policy"]==POLICIES[method] and startup["release_idle_pages"] and startup["async_scheduling"]
        assert startup["effective_server_config"]["additional_config"]["remote_ra_kv_layout"]["mamba_slots"]==49
        assert startup["dram_arena_pages"]==68*1024


def require_interactive_check():
    pilot = json.loads((ROOT / "interactive-check/queue.json").read_text())
    job = str(pilot["job_id"])
    report = json.loads((ROOT / "interactive-check/runs" / job / "verification.json").read_text())
    expected_hash = hashlib.sha256((ROOT / "files.json").read_bytes()).hexdigest()
    if (report.get("passed") is not True or str(report.get("job_id")) != job
            or report.get("partition") != "ghx4-interactive"
            or report.get("manifest_sha256") != expected_hash):
        raise RuntimeError("The interactive GPU check for this benchmark revision has not passed")
    return {"interactive_job_id": job, "manifest_sha256": expected_hash}


def run(method,out,local,port, *, probe_count=None, probe_steps=None):
    pilot = probe_count is not None or probe_steps is not None
    if pilot and (os.environ.get("SLURM_JOB_PARTITION") != "ghx4-interactive"
                  or not isinstance(probe_count, int) or not 1 <= probe_count <= 150
                  or not isinstance(probe_steps, int) or not 1 <= probe_steps <= 120):
        raise ValueError("Short validation runs require a ghx4-interactive allocation and explicit task and step counts")
    if not pilot:
        write(out / "interactive-qualification.json", require_interactive_check())
    allocation_receipt(out)
    scratch = Path(os.environ["TMPDIR"])
    with ExitStack() as stack:
        if method in POLICIES:
            cmd=native_server(method,local,out,port)
        else:
            cmd=[sys.executable,"-B",ROOT/"engines_adapter/launch.py","--method",method,
                 "--output",out/"server","--port",str(port),"--local",local]
        cmd = slurm_step("engine", cmd, out, scratch)
        write(out/"server-command.json",list(map(str,cmd)))
        engine=stack.enter_context(process(cmd,out/"server.log",out/"engine-placement.json"))
        ready(engine,port)
        validate(method,out,port)
        if method in ("agentkv","claude_compression"):
            smoke = [sys.executable, "-B", str(ROOT/"engines_adapter/smoke.py"),
                     "--method", method, "--port", str(port), "--local", str(local),
                     "--output", str(out/"engine-smoke.json")]
            subprocess.run(slurm_step("probe", smoke, out, scratch), env=step_environment(), check=True)
        if pilot and method != "agentkv":
            probe_script = ROOT / "interactive-check/probes.py"
            args = [sys.executable, "-B", str(probe_script)]
            if method in POLICIES:
                args += ["native-live", "--method", method]
            else:
                args.append("compact")
            args += ["--port", str(port), "--out", str(out), "--model", str(local / "model")]
            with (out/"interactive-probe.log").open("x") as log:
                subprocess.run(slurm_step("probe", args, out, scratch), env=step_environment(),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest=ROOT/"local_images/manifest.json"
        output=out/"evaluation"
        write(out/"status.json",{"stage":"inference","time":time.time(),"method":method})
        args=[sys.executable,"-B",str(ROOT/"clients/local_run.py"),"--method",method,
              "--model",str(local/"model"),"--port",str(port),
              "--output",str(output),"--manifest",str(manifest)]
        if pilot:
            args.extend(["--count", str(probe_count), "--steps", str(probe_steps)])
        with (out/"agent.log").open("x") as log:
            subprocess.run(slurm_step("agent", args, out, scratch), stdout=log,
                           stderr=subprocess.STDOUT, env=step_environment(), check=True)
    deadline = time.monotonic() + 60
    while True:
        with socket.socket() as check:
            check.settimeout(1)
            if check.connect_ex(("127.0.0.1", port)) != 0:
                break
        if time.monotonic() >= deadline:
            raise RuntimeError("The inference port remained open after stopping the GPU service")
        time.sleep(1)
    write(out/"status.json",{"stage":"grading","time":time.time(),"method":method})
    with (out/"grading.log").open("x") as log:
        grading = [sys.executable, "-B", str(ROOT/"clients/local_grade.py"),
                   "--output", str(output), "--manifest", str(manifest), "--workers", "8"]
        subprocess.run(slurm_step("grade", grading, out, scratch), stdout=log,
                       stderr=subprocess.STDOUT, env=step_environment(), check=True)



def main():
    p=argparse.ArgumentParser()
    p.add_argument("method",choices=METHODS)
    p.add_argument("out",type=Path)
    p.add_argument("local",type=Path)
    a=p.parse_args()
    try:
        with local_port() as port:
            run(a.method,a.out,a.local,port)
    except BaseException as exc:
        write(a.out/"status.json",{"stage":"failed","error":repr(exc),"time":time.time()})
        raise
    write(a.out/"status.json",{"stage":"complete","time":time.time()})


if __name__=="__main__":
    main()
