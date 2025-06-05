#!/bin/bash
python scripts/small_scale_splits.py \
        --dataset Glioma \
        --save_dir data/splits/glioma \
        --n-splits 10