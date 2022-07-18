#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29508}

CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT tools/trainSize.py $CONFIG ${@:3}
