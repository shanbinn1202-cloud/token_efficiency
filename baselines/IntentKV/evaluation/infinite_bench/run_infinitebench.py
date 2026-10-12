# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
from args import parse_args
from compute_scores import compute_scores
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    check_benchmark_availability,
    create_prompt,
    dump_jsonl,
    get_answer,
    load_data,
)
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from intentkv import patch_model



# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_input_length: int,
    verbose: bool = False,
    generation_config: GenerationConfig = None,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    input_tokens = truncate_by_tokens(input_text, tok, max_input_length)
    # import pdb; breakpoint() # use to investigate the input
    if verbose:
        print("# tokens:", len(input_tokens))
        print("=============== Input ===============")
        print(tok.decode(input_tokens[:200]))
        print("...")
        print(tok.decode(input_tokens[-200:]))
        print("=====================================")
    
    input_tensors = {
        "input_ids": torch.tensor(input_tokens).unsqueeze(0).to(model.device)
    }
    outputs = model.generate(**input_tensors, generation_config=generation_config)
    output = outputs[0, len(input_tokens) :]
    output = tok.decode(output, skip_special_tokens=True)
    output = output.strip()
    # print(input_text[:5000], input_text[-5000:])
    print("Chunked generation:", output)
    return output


def load_model(
    model_name: str,
    max_seq_length: int = None,
    trust_remote_code: bool = False,
    tensor_parallel_size: int = 1,
    enable_intentkv: bool = False,
    init_q_window: int = 64,
    max_capacity_prompt: int = 2048,
    block_size: int = 32,
):
    tok = AutoTokenizer.from_pretrained(
        model_name, resume_download=None, trust_remote_code=trust_remote_code
    )
    tok.pad_token = tok.eos_token
    
    config = AutoConfig.from_pretrained(
        model_name, resume_download=None, trust_remote_code=trust_remote_code
    )
    if "LWM" in model_name:
        c = {
            "theta": 10000000,
            "max_sequence_length": 131072,
            "scan_attention": True,
            "scan_query_chunk_size": 1024,
            "scan_key_chunk_size": 1024,
            "scan_mlp": True,
            "scan_mlp_chunk_size": 1024,
            "scan_layers": True,
        }
        config.update(c)

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype="auto",
        device_map="auto",
        resume_download=None,
        trust_remote_code=trust_remote_code,
    )

    if enable_intentkv:
        patch_model(llm, config={"init_q_window": init_q_window, "max_capacity_prompt": max_capacity_prompt, "block_size": block_size})
    print("Model and tokenizer loaded.")
    return llm, tok


if __name__ == "__main__":
    args = parse_args()

    check_benchmark_availability(args.data_dir)
    model_name = args.model_name_or_path
    max_seq_length = args.max_seq_length
    real_model_name = model_name.split("/")[-1]
    data_name = args.task

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]

    # Model
    model, tok = load_model(
        model_name,
        max_seq_length=max_seq_length,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_intentkv=args.enable_intentkv,
        init_q_window=args.init_q_window,
        max_capacity_prompt=args.max_capacity_prompt,
        block_size=args.block_size,
    )
    results = {}

    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if max_new_tokens >= max_seq_length:
            max_new_tokens = 500

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False,
            # temperature=0,
            # top_p=0.95,
            pad_token_id=tok.pad_token_id,
        )

        # Data
        result_dir = Path(args.output_dir, f"{real_model_name}")
        result_dir.mkdir(exist_ok=True, parents=True)
        output_path = result_dir / f"prediction_{data_name}_{'intentkv' if args.enable_intentkv else 'vanilla'}.jsonl"
        examples = load_data(data_name, data_dir=args.data_dir)

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            examples = examples[:num_eval_examples]

        preds = []
        print("==== Evaluation ====")
        print(f"# examples: {len(examples)}")
        print(f"Num eval examples: {args.num_eval_examples}")
        print(f"Verbose: {args.verbose}")
        print(f"Max new tokens: {max_new_tokens}")

        if os.path.exists(output_path) and not args.rewrite:
            print(f"Output file {output_path} exists. Loading from file.")
            compute_scores(output_path, data_name, real_model_name, max_seq_length)
            with open(output_path) as f:
                preds = [json.loads(ii) for ii in f.readlines()]

        for i, eg in tqdm(enumerate(examples)):
            if i < args.start_example_id or i < len(preds):
                continue
            input_text = create_prompt(eg, data_name, real_model_name, args.data_dir)
            ground_truth = get_answer(eg, data_name)
            # print(input_text.index(ground_truth), len(input_text), input_text.index(ground_truth) / len(input_text))
            # print(f"====== Example {i} ======")
            if "Yi" not in model_name:
                msgs = [dict(role="system", content=input_text)]
                input_text = tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False
                )
            pred = get_pred(
                model,
                tok,
                input_text,
                max_input_length=max_seq_length - max_new_tokens,
                verbose=args.verbose,
                generation_config=generation_config,
            )
            print("Ground Truth", get_answer(eg, data_name))
            if args.verbose:
                print(pred)
            preds.append(
                {
                    "id": i,
                    "prediction": pred,
                    "ground_truth": get_answer(eg, data_name),
                }
            )
            dump_jsonl(preds, output_path)
            torch.cuda.empty_cache()

        result_file_path = f"{real_model_name}_{'intentkv' if args.enable_intentkv else 'vanilla'}"
        score = compute_scores(output_path, data_name, result_file_path, 'intentkv' if args.enable_intentkv else 'vanilla')
        results[data_name] = score

    print("==== Results ====")
    print(json.dumps(results, indent=2))
