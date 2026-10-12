# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Load data
# wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt
# /home/share/models/Meta-Llama-3.1-8B-Instruct \
# /home/share/models/Llama-2-7b-chat-hf/
python benchmark_e2e_kv.py \
    --model_name /home/share/models/Meta-Llama-3.1-8B-Instruct \
    --kv_type intentkv \
    --run_benchmark
