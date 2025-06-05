#!/bin/bash

python scripts/llm_score.py \
        --prompt-filename prompts/llm-select/glioma_prompt.txt \
        --feature_names_path data/adversarial/Lung_TCGA/new_genenames.pkl \
        --category "LUAD vs. LUSC" \
        --wipe \
        --save_dir data/adversarial/llm-score/Lung_TCGA \
        --n-trials 3 \
        --step 1 \
        --num-threads 5 \
        --model-type o1 \
        --temp 0