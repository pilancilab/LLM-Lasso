#!/bin/bash

python scripts/llm_lasso_scores.py \
        --prompt-filename prompts/small_scale_prompts/diabetes_prompt.txt \
        --feature_names_path small_scale/data/Diabetes_feature_names.pkl \
        --category Diabetes \
        --wipe \
        --save_dir data/llm-lasso/diabetes \
        --n-trials 1 \
        --model-type gpt-4o \
        --temp 0