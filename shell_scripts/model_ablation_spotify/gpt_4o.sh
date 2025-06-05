#!/bin/bash

python scripts/llm_lasso_scores.py \
        --prompt-filename prompts/small_scale_prompts/spotify_prompt.txt \
        --feature_names_path small_scale/data/Spotify_feature_names.pkl \
        --category Spotify \
        --wipe \
        --save_dir data/llm-lasso/spotify/gpt-4o \
        --n-trials 1 \
        --model-type gpt-4o \
        --temp 0