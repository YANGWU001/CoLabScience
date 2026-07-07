#!/bin/bash
# Quick Presenter SFT Training Script
echo "Starting Presenter SFT Training with LLaMA Factory"
echo "Model: meta-llama/Llama-3.1-8B-Instruct"
echo "GPUs: 0,1"

# Run LLaMA Factory training
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train \
    --config_path llamafactory_configs/presenter_sft_config.yaml \
    --dataset_dir ./data \
    --dataset_info llamafactory_configs/dataset_info.json

echo "Training completed!"
