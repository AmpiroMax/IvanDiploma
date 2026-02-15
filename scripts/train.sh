#!/bin/bash
# Run batch training with validation
# Usage: ./scripts/train.sh [epochs] [batch_size]

cd "$(dirname "$0")/.."
EPOCHS=${1:-50}
BATCH=${2:-4}

python -m iternet.scripts.train_batch \
  --data_dir data/processed \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --device cuda
