"""Compare real-model eager/graph decode before and after a KV compaction."""
import copy

import torch


@torch.inference_mode()
def validate_graph(runtime):
    runtime.reset()
    runtime.graph_enabled = False
    ids = runtime.tokenizer.encode("This is background information for a short arithmetic question. " * 200)[:1536]
    for offset in range(0, len(ids), 1024):
        logits = runtime.forward(ids[offset:offset + 1024], [0] * len(ids[offset:offset + 1024]), offset)
    cache = copy.deepcopy(runtime.cache)
    rings = {key: value.clone() for key, value in runtime.rings.items()}
    sizes, positions = runtime.ring_sizes.copy(), runtime.ring_positions.copy()
    initial_sizes, initial_positions = tuple(sizes), tuple(positions)
    token = int(logits.argmax().item())
    tokens, expected = [], []
    for step in range(4):
        if step == 2:
            runtime.compact()
        tokens.append(token)
        logits = runtime.forward([token], [1], len(ids) + step)
        expected.append(logits.clone())
        if step == 0:
            first_key = runtime.cache.layers[runtime.layers[0]].keys[:, :, -1].clone()
            first_state = runtime.cache.layers[0].recurrent_states[0].clone()
        token = int(logits.argmax().item())
    expected_length = runtime.cache.layers[runtime.layers[0]].keys.shape[-2]
    runtime.cache, runtime.rings = cache, rings
    runtime.ring_sizes, runtime.ring_positions = sizes, positions
    runtime.graph_enabled = True
    differences = []
    for step, token in enumerate(tokens):
        if step == 2:
            runtime.compact()
        actual = runtime.forward([token], [1], len(ids) + step)
        logp = expected[step].float().log_softmax(-1)
        logq = actual.float().log_softmax(-1)
        differences.append({"step": step,
            "max_logit_difference": float((actual.float() - expected[step].float()).abs().max().item()),
            "kl": float((logp.exp() * (logp - logq)).sum().item()),
            "expected_argmax": int(expected[step].argmax().item()),
            "actual_argmax": int(actual.argmax().item())})
        print(differences[-1], flush=True)
        if step == 0:
            first_key_difference = float((first_key - runtime.cache.layers[runtime.layers[0]].keys[:, :, len(ids)]).abs().max().item())
            first_state_difference = float((first_state - runtime.cache.layers[0].recurrent_states[0]).abs().max().item())
            print({"first_attention_key_difference": first_key_difference, "first_linear_state_difference": first_state_difference}, flush=True)
    assert runtime.graph_context.length == expected_length == min(len(ids) + 2, runtime.plan.budget) + 2
    runtime.graph_context.deactivate()
    assert all(runtime.cache.layers[i].keys.shape[-2] == expected_length for i in runtime.layers)
    # Isolate graph capture itself from variable-length versus padded SDPA kernels.
    static_results = []
    for replay in (False, True):
        runtime.cache = copy.deepcopy(cache)
        runtime.rings = {key: value.clone() for key, value in rings.items()}
        runtime.ring_sizes, runtime.ring_positions = list(initial_sizes), list(initial_positions)
        runtime.graph_decoder.replay_enabled = replay
        outputs = []
        for step, token in enumerate(tokens):
            if step == 2:
                runtime.compact()
            outputs.append(runtime.forward([token], [1], len(ids) + step).clone())
        runtime.graph_context.deactivate()
        static_results.append(outputs)
    for step, (eager, captured) in enumerate(zip(*static_results)):
        print({"static_step": step, "max_logit_difference": float((eager.float() - captured.float()).abs().max().item())}, flush=True)
        torch.testing.assert_close(eager, captured, rtol=0, atol=0)
    assert all(d["expected_argmax"] == d["actual_argmax"] and d["kl"] < 0.001 for d in differences)
    assert first_key_difference == 0 and first_state_difference == 0
    runtime.reset()
    return {"eager_graph_argmax_equal": all(d["expected_argmax"] == d["actual_argmax"] for d in differences), "differences": differences,
            "first_attention_key_difference": first_key_difference, "first_linear_state_difference": first_state_difference,
            "static_eager_vs_graph_logits_bit_exact": True,
            "decode_after_compaction_passed": True, "physical_length_after_compaction_and_two_steps": expected_length}
