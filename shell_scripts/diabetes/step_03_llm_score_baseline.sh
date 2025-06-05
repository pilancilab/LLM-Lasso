#!/bin/bash

python scripts/llm_score.py \
        --prompt-filename prompts/llm-select/diabetes_prompt.txt \
        --feature_names_path small_scale/data/Diabetes_feature_names.pkl \
        --category Diabetes \
        --wipe \
        --save_dir data/llm-score/diabetes \
        --n-trials 1 \
        --step 1 \
        --model-type gpt-4o \
        --temp 0