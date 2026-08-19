#!/usr/bin/env bash
set -e
CONFIG=${1:-configs/fedspd.yaml}
for NP in 20 30 40 50 60; do
  python federated/train_FedSPD_bs4.py --config ${CONFIG} --algorithm fedspd --num_points ${NP} --num_task_tokens 10 --out_dir work_dirs/sens_np_${NP}
done
