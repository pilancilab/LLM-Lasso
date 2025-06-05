#!/bin/bash

python scripts/llm_score.py \
        --prompt-filename prompts/llm-select/glioma_prompt.txt \
        --feature_names_path small_scale/data/Glioma_feature_names.pkl \
        --category Glioma \
        --wipe \
        --save_dir data/llm-score/glioma \
        --n-trials 1 \
        --step 1 \
        --model-type gpt-4o \
        --temp 0