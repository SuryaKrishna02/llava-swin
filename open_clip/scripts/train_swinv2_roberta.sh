#!/bin/bash

# Script for training OpenCLIP model with Swin-v2-B and RoBERTa
# Usage: ./scripts/train_swinv2_roberta.sh <num_gpus>

# Check if number of GPUs is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <num_gpus>"
  exit 1
fi

NUM_GPUS=$1
MASTER_PORT=$(shuf -i 10000-65535 -n 1)

# Directory setup
DATA_DIR="./data"  # Replace with your actual data path
OUTPUT_DIR="./checkpoints/swinv2b_roberta"
LOG_DIR="./logs/swinv2b_roberta"

# Create directories if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Training parameters
BATCH_SIZE=64           # Per GPU batch size
EPOCHS=90               # As per the SwinV2-B* configuration
LR=1.25e-4              # Learning rate from SwinV2-B* configuration
WARMUP_EPOCHS=5         # Warmup epochs from SwinV2-B* configuration
WARMUP=5000             # Calculated from warmup epochs
PRECISION="amp"         # Use AMP for mixed precision training
WINDOW_SIZE=16          # Swin Transformer window size for 256x256 input
WEIGHT_DECAY=0.1        # From SwinV2-B* configuration

# Use timm's pre-trained model
PRETRAINED_IMAGE="timm/swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"

# Dataset configuration
DATASET_TYPE="data"
TRAIN_DATA="$DATA_DIR/dummy_dataset.tar"
TRAIN_NUM_SAMPLES=1000  # Adjust to your dataset size

# Launch distributed training using torchrun
cd $(dirname $0)/..

torchrun --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m src.open_clip_train.main \
  --train-data="$TRAIN_DATA" \
  --train-num-samples=$TRAIN_NUM_SAMPLES \
  --dataset-type=$DATASET_TYPE \
  --batch-size=$BATCH_SIZE \
  --precision=$PRECISION \
  --workers=4 \
  --epochs=$EPOCHS \
  --lr=$LR \
  --warmup=$WARMUP \
  --output-dir=$OUTPUT_DIR \
  --log-dir=$LOG_DIR \
  --name="swinv2b_roberta" \
  --window-size=$WINDOW_SIZE \
  --save-frequency=1 \
  --save-most-recent \
  --report-to="tensorboard" \
  --local-loss \
  --gather-with-grad \
  --weight-decay=$WEIGHT_DECAY \
  --patch-dropout=0.0 \
  --model="swinv2_roberta_base" \
  --hf-tokenizer-name="roberta-base" \
  ${PRETRAINED_IMAGE:+--pretrained-image="$PRETRAINED_IMAGE"}