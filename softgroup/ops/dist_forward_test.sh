#!/usr/bin/env bash
PORT=${PORT:-29530}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --master_port=$PORT  softgroup/ops/forward_group_test.py