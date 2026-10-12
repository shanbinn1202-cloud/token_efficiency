# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

TASKS=("kv_retrieval" "longbook_choice_eng" "math_find" "longbook_qa_chn" "longbook_qa_eng" "longdialogue_qa_eng" "code_debug" "longbook_sum_eng" "number_string" "passkey")
# TASKS=("longbook_qa_eng")
export TOKENIZERS_PARALLELISM=false


for task in ${TASKS[@]}; do
echo $task
if [ "$task" == "kv_retrieval" ]; then
    block_size=128
else
    block_size=16
fi
python run_infinitebench.py \
    --task $task \
    --model_name_or_path /home/share/models/Meta-Llama-3.1-8B-Instruct \
    --data_dir /home/lzhao/Work/benchmarks/InfiniteBench \
    --output_dir ./results \
    --max_seq_length 131072 \
    --rewrite \
    --enable_intentkv \
    --block_size $block_size \
    --num_eval_examples -1
done
