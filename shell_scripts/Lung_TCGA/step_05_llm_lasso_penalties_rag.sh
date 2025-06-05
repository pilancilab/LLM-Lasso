#!/bin/bash

python scripts/llm_lasso_scores.py \
        --prompt-filename prompts/Lung_TCGA_prompt.txt  \
        --feature_names_path data/Lung_TCGA/genenames.txt \
        --category "LUAD vs. LUSC" \
        --wipe \
        --save_dir data/llm-lasso/Lung_TCGA \
        --n-trials 3 \
        --model-type o1 \
        --omim-rag \
        --omim-rag-num-docs 3