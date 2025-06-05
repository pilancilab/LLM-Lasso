#!/bin/bash
python scripts/small_scale_splits.py \
        --dataset Diabetes \
        --save_dir data/splits/diabetes \
        --n-splits 10