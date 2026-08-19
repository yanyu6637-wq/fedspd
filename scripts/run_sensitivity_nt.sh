#!/usr/bin/env bash
set -e
CONFIG=${1:-configs/fedspd.yaml}
for NT in 4 8 10 12 16; do
  python federated/train_FedSPD_bs4.py --config ${CONFIG} --algorithm fedspd --num_points 40 --num_task_tokens ${NT} --out_dir work_dirs/sens_nt_${NT}
done
