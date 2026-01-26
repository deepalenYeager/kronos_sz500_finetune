#!/usr/bin/env bash
set -euo pipefail

GPUS=${GPUS:-1}
SKIP_TOKENIZER=${SKIP_TOKENIZER:-1}

export KRONOS_CONFIG_MODULE="finetune.config_sz50"
export KRONOS_PRETRAINED_TOKENIZER="./pretrained/Kronos-Tokenizer-base"
export KRONOS_PRETRAINED_PREDICTOR="./pretrained/Kronos-small"

python verify_sz50_setup.py

if [[ "$SKIP_TOKENIZER" -eq 1 ]]; then
  python run_sz50_finetune.py --gpus "$GPUS" --skip-tokenizer
else
  python run_sz50_finetune.py --gpus "$GPUS"
fi
