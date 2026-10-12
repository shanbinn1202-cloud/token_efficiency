"""Real attention-mask, persistent-history, and native preemption checks."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier
import uuid

import httpx

from tebench.config import load


def check_attention_mask():
    import torch
    from vllm.v1.attention.backends.flash_attn import _make_rswa_mask_mod
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.manual_seed(42)
    errors = []
    for prefix, budget, length, query_count in [
        (7, 20, 97, 1),
        (7, 20, 97, 33),
        (0, 20, 41, 41),
        (16, 20, 51, 13),
    ]:
        q = torch.randn(query_count, 4, 256, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(length, 2, 256, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        offsets_q = torch.tensor([0, query_count], device="cuda", dtype=torch.int32)
        offsets_k = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        actual = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=offsets_q,
            cu_seqlens_k=offsets_k,
            max_seqlen_q=query_count,
            max_seqlen_k=length,
            causal=False,
            softmax_scale=1 / 16,
            fa_version=4,
            mask_mod=_make_rswa_mask_mod(True),
            aux_tensors=[
                torch.tensor([prefix], device="cuda", dtype=torch.int32),
                torch.tensor([budget], device="cuda", dtype=torch.int32),
            ],
        )
        keys = k.repeat_interleave(2, dim=1).transpose(0, 1).float()
        values = v.repeat_interleave(2, dim=1).transpose(0, 1).float()
        queries = q.transpose(0, 1).float()
        qp = torch.arange(length - query_count, length, device="cuda")[:, None]
        kp = torch.arange(length, device="cuda")[None, :]
        mask = (kp <= qp) & ((kp < prefix) | (qp - kp < budget - prefix))
        logits = queries @ keys.transpose(-1, -2) / 16
        expected = (
            logits.masked_fill(~mask, -torch.inf).softmax(-1) @ values
        ).transpose(0, 1)
        error = float((actual.float() - expected).abs().max())
        assert error < 0.035, (prefix, budget, length, query_count, error)
        errors.append(error)
    return errors


def check_recycled_pages():
    """Masked lanes must not import NaNs from a page recycled as SSM state."""
    import torch
    from vllm.tebench.native_window import gap_blocks
    from vllm.v1.attention.backends.flash_attn import _make_rswa_mask_mod
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    prefix, budget, length, block = 29, 8192, 16791, 784
    pages = (length + block - 1) // block
    q = torch.randn(1, 24, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(pages, block, 4, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    positions = torch.arange(length, device="cuda")
    active = (positions < prefix) | (length - 1 - positions < budget - prefix)
    keys = k.flatten(0, 1)[:length][active].repeat_interleave(6, dim=1).float()
    values = v.flatten(0, 1)[:length][active].repeat_interleave(6, dim=1).float()
    logits = torch.einsum("qhd,khd->hqk", q.float(), keys) / 16
    expected = torch.einsum("hqk,khd->qhd", logits.softmax(-1), values)
    first, last = gap_blocks(prefix, length - 1, budget, block)
    k[first:last] = float("nan")
    v[first:last] = float("nan")
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :]

    def attention():
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            seqused_k=torch.tensor([length], device="cuda", dtype=torch.int32),
            max_seqlen_q=1,
            max_seqlen_k=length,
            block_table=table,
            causal=False,
            softmax_scale=1 / 16,
            fa_version=4,
            mask_mod=_make_rswa_mask_mod(True),
            aux_tensors=[
                torch.tensor([prefix], device="cuda", dtype=torch.int32),
                torch.tensor([budget], device="cuda", dtype=torch.int32),
            ],
        )

    poisoned = attention()
    table[:, first:last] = 0  # This test's finite, owned prefix page.
    actual = attention()
    error = float((actual.float() - expected).abs().max())
    assert actual.isfinite().all() and error < 0.035, error
    return dict(
        unprotected_nonfinite=int((~poisoned.isfinite()).sum()),
        protected_max_error=error,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pressure", action="store_true")
    args = parser.parse_args()
    config = load(args.config)
    budget = config["cache"]["kv_budget"]
    url = config["backend"]["endpoint"].rstrip("/")
    errors = check_attention_mask()
    recycled_pages = check_recycled_pages()
    from scripts.run_clean_stage import early_eos_smoke, native_smoke

    early_eos_smoke(config, "sliding_window", args.output.with_name("early-eos.json"))
    native_smoke(config, "sliding_window", args.output.with_name("frontend.json"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["path"], local_files_only=True
    )

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    clients = 8 if args.pressure else 1
    gate = Barrier(clients)
    targets = None
    pool = None
    if args.pressure:
        with httpx.Client(timeout=1200) as client:
            response = client.post(
                url.removesuffix("/v1") + "/collective_rpc",
                json={"method": "kv_session_recompute_stats"},
            )
            response.raise_for_status()
            pool = response.json()["results"][0]
        groups = pool["groups"]
        attention = [g for g in groups if g["type"] == "RSWASpec"]
        assert len(attention) == 1, groups
        page = attention[0]["block_size"]
        recurrent = sum(g["type"] == "MambaSpec" for g in groups)
        quotient, remainder = divmod(pool["kv_pool_blocks"] - 1, clients)
        targets = [
            (quotient + (i < remainder) - recurrent) * page - 8 for i in range(clients)
        ]
        assert max(targets) < budget

    def case(index):
        sid = uuid.uuid4().hex
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Remember case {index}. Count positive integers, one by one.",
            },
        ]
        initial = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=True
        )
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        prefix = len(encode(initial))
        history = []
        rows = []
        pending = None
        with httpx.Client(base_url=url + "/", timeout=1200) as client:
            for turn in range(3):
                ids = (
                    encode(full)
                    if not turn
                    else (
                        ([pending] if pending is not None else [])
                        + encode(
                            "<|im_end|>\n<|im_start|>user\n"
                            + " data" * (budget + 900 + index * 31)
                            + "\nContinue counting positive integers.<|im_end|>\n<|im_start|>assistant\n<think>\n"
                        )
                    )
                )
                if not turn and targets:
                    # Fill native pages near their next allocation boundary;
                    # decode must request more blocks before reaching B.
                    filler = encode(" data")
                    assert len(filler) == 1 and len(ids) < targets[index]
                    ids += filler * (targets[index] - len(ids))
                body = dict(
                    method="sliding_window",
                    session_id=sid,
                    input_ids=ids,
                    prefix_tokens=prefix,
                    header_tokens=5,
                    budget=budget,
                    max_tokens=512 if args.pressure else 32,
                    temperature=0,
                    top_p=1,
                    top_k=-1,
                    repetition_penalty=1,
                    seed=0,
                    ignore_eos=True,
                )
                gate.wait(timeout=1200)
                result = client.post("kv-session/complete", json=body)
                result.raise_for_status()
                data = result.json()
                outputs = data.pop("output_ids")
                data.pop("text")
                assert data["cached_tokens"] == len(history)
                assert data["prefill_tokens"] == len(ids)
                assert data["prefix_tokens"] == prefix
                assert data["native_request_finished"]
                assert data["checkpoint_restores"] == turn
                if turn and not args.pressure:
                    reference_id = uuid.uuid4().hex
                    reference = client.post(
                        "kv-session/complete",
                        json={
                            **body,
                            "session_id": reference_id,
                            "input_ids": history + ids,
                        },
                    )
                    reference.raise_for_status()
                    reference_tokens = reference.json()["output_ids"]
                    data["recompute_reference_tokens_match"] = (
                        outputs == reference_tokens
                    )
                    assert outputs == reference_tokens, (outputs, reference_tokens)
                    client.post(
                        "kv-session/close", json={"session_id": reference_id}
                    ).raise_for_status()
                history += ids + outputs[:-1]
                pending = outputs[-1]
                assert data["logical_computed_tokens"] == len(history)
                assert data["retained_kv_tokens"] == min(len(history), budget - 1)
                assert data["total_evicted_tokens"] == max(
                    0, len(history) - (budget - 1)
                )
                rows.append(data)
                print(
                    "WINDOW_TURN_DONE",
                    index,
                    turn,
                    len(history),
                    data["recompute"],
                    flush=True,
                )
            client.post("kv-session/close", json={"session_id": sid}).raise_for_status()
        return rows

    with ThreadPoolExecutor(max_workers=clients) as executor:
        cases = list(executor.map(case, range(clients)))
    preemptions = sum(rows[-1]["recompute"]["preemptions"] for rows in cases)
    recomputed = sum(rows[-1]["recompute"]["recomputed_tokens"] for rows in cases)
    report = dict(
        passed=True,
        clients=clients,
        cases=cases,
        mask_max_errors=errors,
        preemptions=preemptions,
        recomputed_tokens=recomputed,
        native_pool=pool,
        pressure_prompt_lengths=targets,
        recycled_page_validation=recycled_pages,
    )
    if args.pressure:
        report["passed"] = preemptions > 0 and recomputed > 0
    args.output.write_text(json.dumps(report, indent=2))
    assert report["passed"], "No native capacity preemption was observed"
    print("NATIVE_PREFIX_WINDOW_VALIDATED", flush=True)


if __name__ == "__main__":
    main()
