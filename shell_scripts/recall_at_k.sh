#!/bin/bash

python scripts/omim_recall.py \
    --testfile prompts/recall_at_k.txt \
    --k 1 3 5 8 10 20 50 100