#!/bin/bash

python scripts/run_baselines.py \
        --split-dir data/splits/Lung_TCGA \
        --n-splits 10 \
        --save-dir data/baselines/Lung_TCGA \
        --max 50