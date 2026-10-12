import argparse
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import repeat_kv

from intentkv import IntentKVCache
from utils import bench_func

class DynamicCacheWithRepeat(DynamicCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp_key_cache = []
        self.temp_value_cache = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        update_global_past_kv = cache_kwargs.get("update_global_past_kv", True)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if update_global_past_kv:  # add KVs to global past_kv
            assert len(self.temp_key_cache) == 0 and len(self.temp_value_cache) == 0, (
                "when you updating global past_kv, make sure the temp_kv_cache is empty. "
                "User past_key_values.clear_temp_kv_cache() to empty the temp_kv_cache"
            )

            # prefilling
            if len(self.key_cache) == layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:  # decoding
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
        else:  # add KVs to temp_kv_cache, this is used when you have a common context but different query, the KVs of the query will be added to temp_kv_cache, and will be cleaned in the next query
            if len(self.temp_key_cache) == layer_idx:
                self.temp_key_cache.append(key_states)
                self.temp_value_cache.append(value_states)
            else:  # decoding
                self.temp_key_cache[layer_idx] = torch.cat(
                    [self.temp_key_cache[layer_idx], key_states], dim=-2
                )
                self.temp_value_cache[layer_idx] = torch.cat(
                    [self.temp_value_cache[layer_idx], value_states], dim=-2
                )

        if self.temp_key_cache:  # concat global past_kv and temp_kv_cache
            key_states, value_states = torch.cat(
                [self.key_cache[layer_idx], self.temp_key_cache[layer_idx]], dim=-2
            ), torch.cat(
                [self.value_cache[layer_idx], self.temp_value_cache[layer_idx]], dim=-2
            )
        else:
            key_states, value_states = (
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
            )

        # repeat kv if needed
        query_states = cache_kwargs.get("query_states", None)
        if query_states is not None:
            key_states = repeat_kv(
                key_states, query_states.size(1) // key_states.size(1)
            )
            value_states = repeat_kv(
                value_states, query_states.size(1) // value_states.size(1)
            )
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens

    def clear_temp_kv_cache(self):
        if self.temp_key_cache:
            self._seen_tokens -= self.temp_key_cache[-1].shape[
                -2
            ]  # seq_len of temp_kv_cache
        self.temp_key_cache = []
        self.temp_value_cache = []

method_to_cache_obj = {
    "dense": DynamicCacheWithRepeat,
    "intentkv": IntentKVCache,
}
def run_target_length(m: int, model, kv_type: str, intentkv_config: dict, prefill=False):
    # wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt
    prompt_complex = open("./prompt_hardest.txt", encoding="gbk").read()
    input_ids = tokenizer(prompt_complex)["input_ids"]
    n = len(input_ids)
    b = m // n + 1

    new_input_ids = (input_ids * b)[:m]
    prompt = tokenizer.decode(new_input_ids)
    data = tokenizer(prompt, return_tensors="pt")
    input_ids = data["input_ids"].cuda()
    
    if kv_type == "dense":
        past_key_values = DynamicCacheWithRepeat()
    else:
        past_key_values = method_to_cache_obj[kv_type](config=intentkv_config)
    
    if prefill:
        def func_prefill():
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            del outputs
            del past_key_values
            torch.cuda.empty_cache()
        gen_latency, gen_memory = bench_func(func_prefill, num_steps=20, num_warmup_steps=5)
    else:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        # Calculate memory overhead
        if kv_type == "dense":
            total_memory = 0
            for layer in past_key_values:
                for tensor in layer:
                    total_memory += tensor.numel() * tensor.element_size()
            total_memory_GB = total_memory / (1024 ** 3)
        else:
            total_memory = 0
            for keys, values in zip(past_key_values.key_cache, past_key_values.value_cache):
                total_memory += keys.numel() * keys.element_size()
                total_memory += values.numel() * values.element_size()
            total_memory_GB = total_memory / (1024 ** 3)
        
        print(f"Total memory overhead of past_key_values: {total_memory_GB:.2f} GB")
        def func():
            with torch.no_grad():
                _ = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            model.generate(input_ids=input_ids, use_cache=True, max_new_tokens=1)
        gen_latency, gen_memory = bench_func(func, num_steps=100, num_warmup_steps=50)

    print(kv_type, m, gen_latency, gen_memory)
    return gen_latency, gen_memory


def run_benchmark(model_name: str, intentkv_config: dict, prefill=False):
    TARGET_LENS = [l * 1000 for l in [32, 64, 96, 128]]
    KV_TYPES = ["dense", "intentkv"]
    KV_TYPES2NAME = {
        "dense": "FullKV",
        "intentkv": "IntentKV",
    }
    latency = defaultdict(list)

    for kv_type in KV_TYPES:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        for l in TARGET_LENS:
            t, m = run_target_length(l, model, kv_type, intentkv_config, prefill)
            latency[KV_TYPES2NAME[kv_type]].append([l, f"{t:.5f}|{m:.5f}"])
            
            print(kv_type, t, l)
            torch.cuda.empty_cache()

    res = [[""] + [KV_TYPES2NAME[kv_type] for kv_type in KV_TYPES]]
    for idx in range(len(TARGET_LENS)):
        l = TARGET_LENS[idx]
        res.append(
            [f"{l//1000}K"]
            + [latency[KV_TYPES2NAME[kv_type]][idx][-1] for kv_type in KV_TYPES]
        )
    print("\n".join(["\t".join(ii) for ii in res]))
    with open("res.csv", "w") as f:
        f.write("\n".join(["\t".join(ii) for ii in res]))
    return res


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    args.add_argument(
        "--kv_type",
        type=str,
        choices=[
            "dense",
            "intentkv",
        ],
    )
    args.add_argument("--context_window", type=int, default=100000)
    args.add_argument("--run_benchmark", action="store_true")
    args.add_argument("--trust_remote_code", action="store_true")
    args.add_argument("--init_q_window", type=int, default=64)
    args.add_argument("--max_capacity_prompt", type=int, default=4096)
    args.add_argument("--block_size", type=int, default=16)
    args.add_argument("--prefill", action="store_true")
    args = args.parse_args()

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=args.trust_remote_code
    )
    intentkv_config = {
        "init_q_window": args.init_q_window,
        "max_capacity_prompt": args.max_capacity_prompt,
        "block_size": args.block_size,
    }
    if args.run_benchmark:
        run_benchmark(model_name, intentkv_config, args.prefill)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=args.trust_remote_code,
        )
        run_target_length(args.context_window, model, args.kv_type, intentkv_config, args.prefill)
