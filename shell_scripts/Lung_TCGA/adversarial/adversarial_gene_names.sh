#!/bin/bash

python scripts/adversarial_feature_names.py \
    --feature-names-path data/Lung_TCGA/genenames.txt \
    --fake-names-dir data/adversarial/Lung_TCGA \
    --replace-top False \
    --max-replace 800