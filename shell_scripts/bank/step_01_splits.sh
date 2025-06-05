#!/bin/bash
python scripts/small_scale_splits.py \
        --dataset Bank \
        --save_dir data/splits/bank \
        --n-splits 10