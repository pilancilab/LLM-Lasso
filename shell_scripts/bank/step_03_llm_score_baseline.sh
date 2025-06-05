#!/bin/bash

python scripts/llm_score.py \
        --prompt-filename prompts/llm-select/bank_prompt.txt \
        --feature_names_path small_scale/data/Bank_feature_names.pkl \
        --category Bank \
        --wipe \
        --save_dir data/llm-score/bank \
        --n-trials 1 \
        --step 1 \
        --model-type gpt-4o \
        --temp 0