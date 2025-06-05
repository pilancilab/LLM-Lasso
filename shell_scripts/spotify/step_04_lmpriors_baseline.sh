#!/bin/bash

python scripts/run_lmpriors.py \
        --prompt-filename prompts/lmpriors/spotify_prompt.txt \
        --feature-description-path prompts/lmpriors/spotify_feature_description.json \
        --category Spotify \
        --data_name Spotify \
        --wipe \
        --save_dir data/lmpriors/spotify \
        --model-type gpt-4o \
        --temp 0