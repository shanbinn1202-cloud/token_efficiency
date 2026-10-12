"""Run either backend in one allocation, sharing all server/client settings."""
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
from tebench.config import ROOT, load
from tebench.methods import check_support
from tebench.run import main, parser
from tebench.serve import command, environment


def run():
    args = parser().parse_args()
    c = load(args.config, args.set + ([f"backend.name={args.backend}"] if args.backend else []))
    check_support(args.method, c)
    if args.dry_run or args.list or args.eval_only or c["backend"]["name"] == "transformers":
        return main()
    log = ROOT / "logs" / f"vllm-{os.environ.get('SLURM_JOB_ID', os.getpid())}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = environment(args.method, c)
    if c["model"]["local_files_only"]:
        env["HF_HUB_OFFLINE"] = "1"
    with log.open("w") as handle:
        server = subprocess.Popen(command(c, args.method), stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + c["generation"]["request_timeout"]
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited with {server.returncode}; inspect {log}")
                try:
                    with urlopen(c["backend"]["endpoint"].rstrip("/") + "/models", timeout=2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"vLLM startup timed out; inspect {log}")
                    time.sleep(2)
            main()
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


if __name__ == "__main__":
    run()
