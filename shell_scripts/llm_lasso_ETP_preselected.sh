#!/bin/bash

python scripts/llm_lasso_scores.py \
        --prompt-filename prompts/simple_etp_prompt.txt \
        --feature_names_path data/preselected/ETP/genes_per_split.txt \
        --category "ETP-All and non-ETP-All" \
        --wipe \
        --preselection_done \
        --save_dir data/llm-lasso/etp_experimental \
        --n-trials 5 \
        --num_threads 5 \
        --batch_size 25 \
        --model-type o1 \
        --temp 0