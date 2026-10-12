"""Optional real-model transport/tool smoke test; writes no result files."""
import argparse
import json
from tebench.config import load
from tebench.backends import create
from tebench.methods import create_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="transformers")
    p.add_argument("--method", default="vanilla")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--turns", type=int, default=2)
    p.add_argument("--observation-padding", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()
    c = load(overrides=[f"backend.name={args.backend}", *args.set])
    backend = create(c, args.method)
    print(json.dumps(getattr(backend, "execution_info", {})), flush=True)
    try:
        # Exercise model query, original action parser and repeated local-cache use.
        model = create_model(args.method, c, backend, model_name="openai/Qwen3.6-27B", cost_tracking="ignore_errors",
            model_kwargs={"max_tokens": args.max_tokens, "temperature": 0})
        messages = [{"role": "system", "content": "Use the bash tool exactly once per reply. The command must be echo smoke. Do not explain."},
                    {"role": "user", "content": "Run the command now."}]
        for turn in range(args.turns):
            answer = model.query(messages)
            assert answer["extra"]["actions"], answer
            messages.append(answer)
            call = answer["tool_calls"][0]
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": "smoke\n" + "background " * args.observation_padding + "\nRun echo smoke again."})
            print(json.dumps({"turn": turn, "actions": answer["extra"]["actions"], "backend": backend.last_measurement}, default=str), flush=True)
        print("GPU_SMOKE_PASSED", args.method, args.backend, flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
