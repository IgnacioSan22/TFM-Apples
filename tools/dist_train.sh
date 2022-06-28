#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29507}

CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py $CONFIG ${@:3}
