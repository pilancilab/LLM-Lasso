#!/bin/bash

python scripts/llm_lasso_scores.py \
        --prompt-filename prompts/small_scale_prompts/bank_prompt.txt \
        --feature_names_path small_scale/data/Bank_feature_names.pkl \
        --category Bank \
        --wipe \
        --save_dir data/llm-lasso/bank \
        --n-trials 1 \
        --model-type gpt-4o \
        --temp 0