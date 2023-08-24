#!/bin/bash
for((i=1;i<=$1;i++))
do
    python sync_benchmark_serving.py --backend=vllm --host=localhost --port=8000 --dataset=./ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer="hf-internal-testing/llama-tokenizer"  --num-prompts=100 &
done
