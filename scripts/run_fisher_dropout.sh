#!/usr/bin/env bash
set -e
CONFIG=${1:-configs/fedspd.yaml}
for DROP in 0 1 2; do
  python federated/train_FedSPD_bs4.py --config ${CONFIG} --algorithm fedspd --drop_clients ${DROP} --out_dir work_dirs/fedspd_drop${DROP}
  python federated/train_FedSPD_bs4.py --config ${CONFIG} --algorithm fedspd --drop_clients ${DROP} --no_fisher --out_dir work_dirs/fedspd_no_fisher_drop${DROP}
done
