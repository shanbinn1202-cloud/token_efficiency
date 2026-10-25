"""Launch vLLM with the same resolved model/context/precision settings as run."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from urllib.parse import urlparse
from tebench.config import load
from tebench.engine import engine_root, has_policy_hooks


def command(c, method=None):
    clean_hooks = has_policy_hooks()
    m, b, g = c["model"], c["backend"], c["generation"]
    endpoint = urlparse(b["endpoint"])
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", m["path"],
        "--served-model-name", m["name"].split("/")[-1], "--host", endpoint.hostname or "127.0.0.1",
        "--port", str(endpoint.port or 8000), "--dtype", m["dtype"],
        "--max-model-len", str(g["max_window_size"]), "--tensor-parallel-size", str(b["tensor_parallel_size"]),
        "--max-num-seqs", str(c["evaluation"]["workers"]), "--seed", str(g["seed"]),
        "--gpu-memory-utilization", str(b["gpu_memory_utilization"]), "--language-model-only", "--generation-config", "vllm",
        "--enable-prefix-caching" if b["prefix_caching"] else "--no-enable-prefix-caching"]
    if b["kv_cache_memory_bytes"] is not None:
        cmd += ["--kv-cache-memory-bytes", str(b["kv_cache_memory_bytes"])]
    # Vanilla uses vLLM's native prefix cache, so identical history blocks are
    # reused across turns. Stateful policies own their physical KV pages and
    # send only the new suffix/restored pages; native prefix caching is unsafe
    # alongside their in-place compaction and is disabled for those servers.
    if method in {"agentkv", "dram_backed_kv", "repairkv", "intentkv", "sliding_window", "random_attention"}:
        cmd[cmd.index("--enable-prefix-caching") if "--enable-prefix-caching" in cmd else cmd.index("--no-enable-prefix-caching")] = "--no-enable-prefix-caching"
    cmd += ["--async-scheduling" if b["async_scheduling"] else "--no-async-scheduling"]
    cmd += ["--scheduling-policy", "fcfs"]
    if method in {"agentkv", "dram_backed_kv", "repairkv", "sliding_window", "random_attention", "intentkv"}:
        if clean_hooks and method == "dram_backed_kv":
            raise ValueError("The clean engine supports the requested six methods; DRAM is not included")
        cmd += ["--middleware", "vllm.tebench.server.middleware" if clean_hooks else "vllm.kv_session_server.middleware"]
    if method in {"agentkv", "dram_backed_kv", "repairkv", "intentkv", "sliding_window", "random_attention"}:
        # Include the extension in vLLM's compiled-artifact cache key. Its
        # environment flags alone are not part of the standard config hash.
        cmd += ["--additional-config", json.dumps({"kv_session_extension": {
            "source": "upstream-ee0da84-v024-triton-six-v1" if clean_hooks else "legacy",
            "method": method, "budget": c["cache"]["kv_budget"],
            **({"intentkv_protocol": "persistent_eviction_v1"} if method == "intentkv" else {}),
            "model_audit": method == "agentkv" and os.environ.get("VLLM_AGENTKV_MODEL_AUDIT") == "1",
            "options": c["methods"][method]["options"]}})]
    if method == "dram_backed_kv":
        # One serialized cold goal-query forward needs a scratch native slot
        # while all eight agent sessions may be parked holding recurrent KV.
        cmd[cmd.index("--max-num-seqs") + 1] = str(c["evaluation"]["workers"] + 1)
    if method == "sliding_window":
        if not clean_hooks:
            raise ValueError("Native sliding window requires the clean vLLM engine")
        cmd += ["--kv-transfer-config", json.dumps({
                    "kv_connector": "RandomAttentionConnector", "kv_role": "kv_both",
                    "kv_connector_module_path": "vllm.tebench.random_connector"})]
    if method == "random_attention":
        cmd += ["--kv-transfer-config", json.dumps({
            "kv_connector": "RandomAttentionConnector", "kv_role": "kv_both",
            "kv_connector_module_path": "vllm.tebench.random_connector"})]
    if method == "intentkv":
        if not clean_hooks:
            raise ValueError("Persistent IntentKV requires the v0.24 policy engine")
        cmd += ["--kv-transfer-config", json.dumps({
            "kv_connector": "IntentKVConnector", "kv_role": "kv_both",
            "kv_connector_module_path": "vllm.tebench.intentkv_connector"})]
    if method == "repairkv":
        cmd += ["--kv-transfer-config", json.dumps({
            "kv_connector": "RepairKVConnector", "kv_role": "kv_both",
            "kv_connector_module_path": "vllm.tebench.repairkv_connector" if clean_hooks else "vllm.repairkv_connector"})]
    if method == "agentkv":
        cmd += ["--kv-transfer-config", json.dumps({
            "kv_connector": "AgentKVConnector", "kv_role": "kv_both",
            "kv_connector_module_path": "vllm.tebench.agentkv_connector" if clean_hooks else "vllm.agentkv_connector"})]
    if clean_hooks:
        cmd += ["--worker-extension-cls", "vllm.tebench.worker.WorkerExtension"]
    compilation = {
        "mode": "VLLM_COMPILE" if b["torch_compile"] else "NONE",
        "cudagraph_mode": ("FULL_AND_PIECEWISE" if b["torch_compile"] else "FULL_DECODE_ONLY")
        if b["cuda_graph"] else "NONE",
    }
    # Use the previously approved common 16K batch profile. Backends are
    # explicit for every method, including Vanilla and AgentDiet.
    cmd += ["--attention-config", json.dumps({"backend": b["vllm_attention_backend"]}),
            "--gdn-prefill-backend", b["gdn_prefill_backend"],
            "--max-num-batched-tokens", str(b["max_num_batched_tokens"])]
    cmd += ["--compilation-config", json.dumps(compilation)]
    if not b["cuda_graph"] and not b["torch_compile"]:
        cmd.append("--enforce-eager")
    return cmd


def environment(method=None, config=None):
    from tebench.config import ROOT
    env = dict(os.environ)
    local = engine_root()
    # Exclude obsolete engine imports from both the API server and spawned workers.
    inherited = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                 if p and not (Path(p) / "vllm/v1/engine/core.py").is_file()]
    env["PYTHONPATH"] = os.pathsep.join([str(local), *inherited])
    env["TEBENCH_VLLM_ROOT"] = str(local)
    env["VLLM_REPAIRKV"] = "1" if method == "repairkv" else "0"
    env['VLLM_INTENTKV'] = '1' if method == 'intentkv' else '0'
    env['VLLM_INTENTKV_GQA'] = '1' if method == 'intentkv' else '0'
    if method == 'intentkv':
        c = config or load()
        env.pop('VLLM_KV_CACHE_LAYOUT', None)
        env['VLLM_INTENTKV_BUDGET'] = str(c['cache']['kv_budget'])
        env['VLLM_INTENTKV_OPTIONS'] = json.dumps(c['methods'][method]['options'])
    if method == "repairkv":
        c = config or load()
        env["PHASE18_SCORE_ON_GPU"] = "0"
        env["VLLM_REPAIRKV_BUDGET"] = str(c["cache"]["kv_budget"])
        env["VLLM_REPAIRKV_OPTIONS"] = json.dumps(c["methods"][method]["options"])
    env["VLLM_RANDOM_ATTENTION"] = "1" if method == "random_attention" else "0"
    if method == "random_attention":
        c = config or load()
        env["VLLM_RANDOM_ATTENTION_BUDGET"] = str(c["cache"]["kv_budget"])
        env["VLLM_RANDOM_ATTENTION_OPTIONS"] = json.dumps(c["methods"][method]["options"])
        env["TRIATTN_RUNTIME_RANDOM_SEED"] = str(c["methods"][method]["options"]["random_seed"])
    env["VLLM_KV_WINDOW"] = "0"
    env["VLLM_NATIVE_WINDOW"] = "0"
    env["VLLM_COMPACT_WINDOW"] = "1" if method == "sliding_window" else "0"
    if method == "sliding_window":
        c = config or load()
        env["VLLM_COMPACT_WINDOW_BUDGET"] = str(c["cache"]["kv_budget"])
        env["VLLM_COMPACT_WINDOW_OPTIONS"] = json.dumps(c["methods"][method]["options"])
    env["VLLM_AGENTKV"] = "1" if method == "agentkv" else "0"
    env["VLLM_DRAM_KV"] = "1" if method == "dram_backed_kv" else "0"
    if method == "dram_backed_kv":
        c = config or load()
        env["VLLM_DRAM_KV_BUDGET"] = str(c["cache"]["kv_budget"])
        env["VLLM_DRAM_KV_OPTIONS"] = json.dumps(c["methods"][method]["options"])
    if method == "agentkv":
        c = config or load()
        env["VLLM_AGENTKV_BUDGET"] = str(c["cache"]["kv_budget"])
        env["VLLM_AGENTKV_OPTIONS"] = json.dumps(c["methods"]["agentkv"]["options"])
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--method", default="vanilla")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    c = load(args.config, args.set)
    cmd = command(c, args.method)
    if args.dry_run:
        print(shlex.join(cmd))
    else:
        if c["model"]["local_files_only"]:
            os.environ["HF_HUB_OFFLINE"] = "1"
        os.execve(cmd[0], cmd, environment(args.method, c))


if __name__ == "__main__":
    main()
