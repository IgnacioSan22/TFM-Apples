#!/usr/bin/env bash
CONFIG=$1
CHECK_POINT=$2
OUT=$3
PORT=${PORT:-29505}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --master_port=$PORT  $(dirname "$0")/test.py --eval --out $OUT $CONFIG $CHECK_POINT
