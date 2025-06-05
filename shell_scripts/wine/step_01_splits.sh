#!/bin/bash
python scripts/small_scale_splits.py \
        --dataset Wine \
        --save_dir data/splits/wine \
        --n-splits 10