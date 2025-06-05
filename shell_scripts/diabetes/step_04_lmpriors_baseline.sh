#!/bin/bash

python scripts/run_lmpriors.py \
        --prompt-filename prompts/lmpriors/diabetes_prompt.txt \
        --feature-description-path prompts/lmpriors/diabetes_feature_description.json \
        --category Diabetes \
        --data_name Diabetes \
        --wipe \
        --save_dir data/lmpriors/diabetes \
        --model-type gpt-4o \
        --temp 0