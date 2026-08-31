#!/bin/bash
cd /root/projects/sera_models/extraction
python3 -u train_lightweight.py \
  --output-dir checkpoints/local_e6b \
  --augmented-data data/processed/e6_augmented_20pct.jsonl \
  --batch-size 32 \
  --gradient-accumulation 2 \
  --epochs 2 \
  --max-val 300 \
  2>&1 | tee logs/local_e6b_final.log
