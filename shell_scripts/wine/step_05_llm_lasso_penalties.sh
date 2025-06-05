#!/bin/bash

python scripts/llm_lasso_scores.py \
        --prompt-filename prompts/small_scale_prompts/wine_prompt.txt \
        --feature_names_path small_scale/data/Wine_feature_names.pkl \
        --category Wine \
        --wipe \
        --save_dir data/llm-lasso/wine \
        --n-trials 1 \
        --model-type gpt-4o \
        --temp 0