#!/bin/bash

python scripts/llm_score.py \
        --prompt-filename prompts/llm-select/glioma_prompt.txt \
        --feature_names_path data/Lung_TCGA/genenames.txt \
        --category "LUAD vs. LUSC" \
        --wipe \
        --save_dir data/llm-score/Lung_TCGA \
        --n-trials 3 \
        --step 1 \
        --num-threads 5 \
        --model-type o1 \
        --temp 0