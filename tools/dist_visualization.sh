#!/usr/bin/env bash
PORT=${PORT:-29506}
OUT=$1

OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --master_port=$PORT  $(dirname "$0")/visualization.py --out $OUT --save_instance_pc
