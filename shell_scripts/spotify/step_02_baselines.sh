#!/bin/bash

python scripts/run_baselines.py \
        --split-dir data/splits/spotify \
        --n-splits 10 \
        --save-dir data/baselines/spotify