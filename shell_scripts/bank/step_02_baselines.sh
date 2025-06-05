#!/bin/bash

python scripts/run_baselines.py \
        --split-dir data/splits/bank \
        --n-splits 10 \
        --save-dir data/baselines/bank