#!/bin/bash

python scripts/run_lmpriors.py \
        --prompt-filename prompts/lmpriors/glioma_prompt.txt \
        --feature-description-path prompts/lmpriors/glioma_feature_description.json \
        --category Glioma \
        --data_name Glioma \
        --wipe \
        --save_dir data/lmpriors/glioma \
        --model-type gpt-4o \
        --temp 0