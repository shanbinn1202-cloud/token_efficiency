"""Real model turns, exact worker KV checks, and native capacity preemption."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier
import time
import uuid

import httpx

from tebench.config import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pressure", action="store_true")
    p.add_argument("--method", choices=("random_attention", "sliding_window"), default="random_attention")
    a = p.parse_args()
    c = load(a.config)
    budget = c["cache"]["kv_budget"]
    url = c["backend"]["endpoint"].rstrip("/")
    from scripts.run_clean_stage import early_eos_smoke, native_smoke

    early_eos_smoke(c, a.method, a.output.with_name("early-eos.json"))
    native_smoke(c, a.method, a.output.with_name("frontend.json"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(c["model"]["path"], local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    with httpx.Client(timeout=1200) as client:
        r = client.post(url.removesuffix("/v1") + "/collective_rpc",
                        json={"method": "kv_session_recompute_stats"})
        r.raise_for_status()
        pool = r.json()["results"][0]
    full = [g for g in pool["groups"] if g["type"] == "FullAttentionSpec"]
    assert len(full) == 1
    page = full[0]["block_size"]
    recurrent = sum(g["type"] == "MambaSpec" for g in pool["groups"])
    blocks = pool["kv_pool_blocks"] - 1
    clients = min(8, blocks//((budget+page-1)//page+recurrent)) if a.pressure else 1
    # A page edge divisible by 64 compacts before it needs another page.
    # Fill the native pool with edges strictly between the original cut points.
    if a.pressure:
        while clients >= 2:
            quotient, remainder = divmod(blocks, clients)
            lengths = [(quotient+(i<remainder)-recurrent)*page for i in range(clients)]
            if all(n % 64 != 0 for n in lengths):
                break
            clients -= 1
        assert clients >= 2
    quotient, remainder = divmod(blocks, clients)
    targets = [(quotient+(i<remainder)-recurrent)*page-8 for i in range(clients)]
    if a.pressure and a.method == "sliding_window":
        clients = 8
    gate = Barrier(clients)

    def case(i):
        sid = uuid.uuid4().hex
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Remember case {i}. Count integers one by one."}]
        initial = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prefix = len(encode(initial))
        rows, history, pending, physical = [], [], None, 0
        with httpx.Client(base_url=url+"/", timeout=1200) as client:
            try:
                for turn in range(3):
                    ids = encode(full) if not turn else [pending] + encode("<|im_end|>\n<|im_start|>user\nContinue counting integers from 1 to 1000 without skipping.<|im_end|>\n<|im_start|>assistant\n<think>\n")
                    # Small protected initial prompt; appended data is eligible history.
                    # Turn 0 crosses B, turn 1 forces allocation before the next cut.
                    if a.method == "sliding_window":
                        desired = budget + 900 + i * 31 if a.pressure and turn else budget + 200 if not turn else 350
                    else:
                        desired = budget+200 if not turn else (targets[i]-physical if a.pressure and turn == 1 else 350)
                    filler = encode(" data")
                    assert len(filler) == 1 and desired >= len(ids), (desired, len(ids), physical)
                    ids += filler * (desired-len(ids))
                    gate.wait(timeout=1200)
                    r = client.post("kv-session/complete", json=dict(method=a.method, session_id=sid,
                        input_ids=ids, prefix_tokens=prefix, header_tokens=5, budget=budget,
                        max_tokens=512 if a.pressure else 256, temperature=0, top_p=1,
                        top_k=-1, repetition_penalty=1, seed=0, ignore_eos=True))
                    r.raise_for_status()
                    row = r.json()
                    outputs = row.pop("output_ids")
                    row.pop("text")
                    assert row["cached_tokens"] == physical
                    assert row["prefill_tokens"] == len(ids)
                    assert row["prefix_tokens"] == prefix
                    assert row["checkpoint_restores"] == turn
                    assert row["native_request_finished"]
                    assert row["compactions"] > (rows[-1]["compactions"] if rows else 0)
                    history += ids+outputs[:-1]
                    pending = outputs[-1]
                    physical = row["retained_kv_tokens"]
                    assert row["logical_computed_tokens"] == len(history)
                    assert budget <= physical < budget+64, physical
                    rows.append(row)
                    print("PERSISTENT_KV_TURN_DONE", a.method, i, turn, len(history), row["recompute"], flush=True)
            except BaseException:
                gate.abort()
                raise
            finally:
                client.post("kv-session/close", json={"session_id": sid}).raise_for_status()
        return rows

    forced_case = None
    if a.pressure and a.method == "sliding_window":
        # Drain in-flight GPU writes before native cache reset: connectors use
        # deferred block freeing under async scheduling. Then test 8 clients.
        gate = Barrier(1)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(case, clients)
            events = a.output.with_name("events.jsonl")
            deadline = time.monotonic() + 1200
            while True:
                if events.exists() and events.read_text().count('"event": "compacted"') >= 2:
                    break
                assert not future.done(), "No compaction observed"
                if time.monotonic() > deadline:
                    raise TimeoutError("No compaction before forced native preemption")
                time.sleep(.05)
            endpoint = url.removesuffix("/v1")
            httpx.post(endpoint + "/pause?mode=keep&clear_cache=false", timeout=1200).raise_for_status()
            try:
                httpx.post(endpoint + "/reset_prefix_cache?reset_running_requests=true", timeout=1200).raise_for_status()
            finally:
                httpx.post(endpoint + "/resume", timeout=1200).raise_for_status()
            forced_case = future.result()
        gate = Barrier(clients)
    with ThreadPoolExecutor(max_workers=clients) as executor:
        cases = list(executor.map(case, range(clients)))
    measured = cases + ([forced_case] if forced_case is not None else [])
    totals = {k: sum(rows[-1]["recompute"][k] for rows in measured)
              for k in ("preemptions", "recomputed_tokens", "replayed_compactions")}
    report = dict(passed=True, method=a.method, clients=clients, cases=cases,
                  native_pool=pool, forced_case=forced_case, **totals)
    if a.pressure:
        report["passed"] = all(v > 0 for v in totals.values())
    a.output.write_text(json.dumps(report, indent=2))
    assert report["passed"], "No native preemption with historical selection replay observed"
    print("RANDOM_ATTENTION_VALIDATED", totals, flush=True)


if __name__ == "__main__":
    main()
