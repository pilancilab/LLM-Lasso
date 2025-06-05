#!/bin/bash

python scripts/run_lmpriors.py \
        --prompt-filename prompts/lmpriors/bank_prompt.txt \
        --feature-description-path prompts/lmpriors/bank_feature_description.json \
        --category Bank \
        --data_name Bank \
        --wipe \
        --save_dir data/lmpriors/bank \
        --model-type gpt-4o \
        --temp 0