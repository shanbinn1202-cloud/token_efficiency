# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from argparse import ArgumentParser, Namespace

def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        # choices=list(DATA_NAME_TO_MAX_NEW_TOKENS.keys()) + ["all"],
        required=True,
        help='Which task to use. Note that "all" can only be used in `compute_scores.py`.',  # noqa
    )
    p.add_argument(
        "--data_dir", type=str, default="../data", help="The directory of data."
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="../results",
        help="Where to dump the prediction results.",
    )  # noqa
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-350m",
        help="For `compute_scores.py` only, specify which model you want to compute the score for.",  # noqa
    )
    p.add_argument(
        "--num_eval_examples",
        type=int,
        default=-1,
        help="The number of test examples to use, use all examples in default.",
    )  # noqa
    p.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data.",
    )  # noqa
    p.add_argument(
        "--stop_idx",
        type=int,
        help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset.",
    )  # noqa
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_seq_length", type=int, default=100000)
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--start_example_id", type=int, default=0)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--enable_intentkv", action="store_true")
    p.add_argument("--init_q_window", type=int, default=64)
    p.add_argument("--max_capacity_prompt", type=int, default=2048)
    p.add_argument("--block_size", type=int, default=16)
    return p.parse_args()
