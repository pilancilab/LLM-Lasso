#!/bin/bash

python scripts/llm_score.py \
        --prompt-filename prompts/llm-select/spotify_prompt.txt \
        --feature_names_path small_scale/data/Spotify_feature_names.pkl \
        --category Spotify \
        --wipe \
        --save_dir data/llm-score/spotify \
        --n-trials 1 \
        --step 1 \
        --model-type gpt-4o \
        --temp 0