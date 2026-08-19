#!/usr/bin/env bash
set -e
CONFIG=${1:-configs/fedspd.yaml}
for ALG in local fedavg fedprox fedper fedmix fedbabu fedfomo fedproto fedala fedspd; do
  python federated/train_FedSPD_bs4.py --config ${CONFIG} --algorithm ${ALG} --out_dir work_dirs/${ALG}
done
