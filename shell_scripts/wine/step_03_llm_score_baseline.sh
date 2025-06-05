#!/bin/bash

python scripts/llm_score.py \
        --prompt-filename prompts/llm-select/wine_prompt.txt \
        --feature_names_path small_scale/data/Wine_feature_names.pkl \
        --category Wine \
        --wipe \
        --save_dir data/llm-score/wine \
        --n-trials 1 \
        --step 1 \
        --model-type gpt-4o \
        --temp 0