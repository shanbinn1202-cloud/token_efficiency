"""Locked local source context. Snapshot identity is not loaded-server proof."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "evidence_sources"
PINS = {
    "decision": ("development/vllm-dynamic-authority-architecture-v1/DECISION.md",
        "f7ab814783f6e29a631b287136b654b357705061e6f87702fe18f164634ea760"),
    "process_report": ("development/vllm-dynamic-process-order-audit-v1/audit_report.json",
        "1c9129f360634e23debfcb4519f624e4d549f4882faae7e9af40d37782e35349"),
    "native_launch": ("verification/vllm-0.24.0-installed-source-audit/engine/utils.py",
        "bc3fbc0fa6d8feb776b43ef008919ae4f28d732ce69bcd3b93649cd38795fc26"),
    "candidate_core": ("development/vllm-dynamic-composed-candidate-v2/tree/source/vllm/v1/engine/core.py",
        "889c8619ecd57dcc13152e93f1172b38fb735c8df1d99ea80d0803b66c8d71d9"),
    "native_worker_gate": ("development/vllm-dynamic-native-worker-seam-v1/native_worker_registration.py",
        "fa22fa4f556cc13becade03e1d405867e3e778cb96acd48651f098f3f22ac50c"),
    "native_worker_manifest": ("development/vllm-dynamic-native-worker-seam-v1/candidate_manifest.json",
        "49fb7461f15ba9cc9796ae89d4b48bfcc94c1b3c670b4d0e2f244d18876af0c9"),
    "frontend_manifest": ("development/vllm-dynamic-frontend-lifespan-v4/manifest.json",
        "e99f9230213f5097dcc4c29b977df9022e01c5b3f9b53bd629d71e22a66c0edb"),
    "chat_router": ("development/vllm-dynamic-chat-route-integration-v1/inputs/api_router.py.txt",
        "118e4e7dfd74d09a296fe4aa8ad094ea829cee5e107c7ac78fed92249db12d1a"),
}


def locked(name):
    path, expected = PINS[name]
    raw = (ROOT / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError("source context drift: " + name)
    return raw


def verify_sources():
    for name in PINS:
        locked(name)
    tree = ast.parse(locked("native_launch"))
    launch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "launch_core_engines")
    yields = sorted(n.lineno for n in ast.walk(launch) if isinstance(n, ast.Yield))
    waits = sorted(n.lineno for n in ast.walk(launch) if isinstance(n, ast.Call)
                   and ast.unparse(n.func) == "wait_for_engine_startup")
    if not yields or len(waits) != 1 or waits[0] <= max(yields):
        raise RuntimeError("native HELLO/INIT wait no longer follows launch yield")
    report = json.loads(locked("process_report"))
    if report["status"] != "NOT_INSTALLABLE" or report["authority_transferred"] is not False:
        raise RuntimeError("unexpected predecessor proof status")
    locations = report["source_locations"]
    if not (locations["candidate_begin"] < locations["candidate_worker_construction"]
            < locations["candidate_activate"] < locations["candidate_input_thread_start"]):
        raise RuntimeError("candidate activation ordering differs")
    core = locked("candidate_core").decode("utf-8")
    if "EngineCoreReadyResponse(" not in core or '"status": "READY"' not in core:
        raise RuntimeError("both native READY exits are required")
    router = ast.parse(locked("chat_router"))
    handler = next(n for n in router.body if isinstance(n, ast.AsyncFunctionDef)
                   and n.name == "create_chat_completion")
    decorators = [ast.unparse(n) for n in handler.decorator_list]
    if "with_cancellation" not in decorators or "load_aware_call" not in decorators:
        raise RuntimeError("chat handler decorator provenance changed")
    return {"status": "LOCKED_SOURCE_CONTEXT_ONLY", "inputs": len(PINS),
            "loaded_server_proof": False, "installed": False, "gpu_used": False,
            "source_hashes_prelocked": False, "chat_source_provenance": "TOFU_FIRST_READ"}


if __name__ == "__main__":
    print(json.dumps(verify_sources(), sort_keys=True))
