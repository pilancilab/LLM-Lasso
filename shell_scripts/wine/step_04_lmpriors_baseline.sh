#!/bin/bash

python scripts/run_lmpriors.py \
        --prompt-filename prompts/lmpriors/wine_prompt.txt \
        --feature-description-path prompts/lmpriors/wine_feature_description.json \
        --category Wine \
        --data_name Wine \
        --wipe \
        --save_dir data/lmpriors/wine \
        --model-type gpt-4o \
        --temp 0