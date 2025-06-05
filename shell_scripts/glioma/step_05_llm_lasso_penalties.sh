#!/bin/bash

python scripts/llm_lasso_scores.py \
        --prompt-filename prompts/small_scale_prompts/glioma_prompt.txt \
        --feature_names_path small_scale/data/Glioma_feature_names.pkl \
        --category Glioma \
        --wipe \
        --save_dir data/llm-lasso/glioma \
        --n-trials 1 \
        --model-type gpt-4o \
        --temp 0