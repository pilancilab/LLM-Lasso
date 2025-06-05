#!/bin/bash

python scripts/llm_lasso_scores.py \
        --prompt-filename prompts/small_scale_prompts/spotify_prompt.txt \
        --feature_names_path small_scale/data/Spotify_feature_names.pkl \
        --category Spotify \
        --wipe \
        --save_dir data/llm-lasso/spotify/gpt-3.5-turbo-0613 \
        --n-trials 1 \
        --model-type openrouter \
        --model-name "openai/gpt-3.5-turbo-0613" \
        --temp 0