#!/bin/bash
python scripts/small_scale_splits.py \
        --dataset Spotify \
        --save_dir data/splits/spotify \
        --n-splits 10