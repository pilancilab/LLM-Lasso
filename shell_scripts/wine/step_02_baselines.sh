#!/bin/bash

python scripts/run_baselines.py \
        --split-dir data/splits/wine \
        --n-splits 10 \
        --save-dir data/baselines/wine