#!/bin/bash

python scripts/llm_lasso_scores.py \
        --prompt-filename prompts/small_scale_prompts/spotify_prompt.txt \
        --feature_names_path small_scale/data/Spotify_feature_names.pkl \
        --category Spotify \
        --wipe \
        --save_dir data/llm-lasso/spotify/llama-3-8b-instruct \
        --n-trials 1 \
        --model-type openrouter \
        --model-name "meta-llama/llama-3-8b-instruct" \
        --temp 0