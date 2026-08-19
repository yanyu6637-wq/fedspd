# FedSPD

> Federated Few-Shot Medical Image Segmentation with Structured Prompts, LoRA Adaptation, and Fisher-Weighted Aggregation.

FedSPD is a federated learning framework for few-shot medical image segmentation built on a promptable SAM (Segment Anything Model) backbone. Each client owns a small support set and query set; a central server coordinates multi-round communication while only **LoRA** adapters are uploaded for aggregation, and **personalized components** (prompt encoders, mask decoder) remain local. The repository additionally ships a full set of federated baselines (FedAvg, FedProx, FedPer, FedMix, FedBABU, FedFomo, FedProto, FedALA) and the experimental scripts used to produce the tables in the paper.

---

## 1. Overview

- **Federated training**: `federated/train_FedSPD_bs4.py` is the main entry point. It builds one `FedClient` per patient/client, runs local training, and aggregates on the `FedServer`.
- **Structured prompt generation (SP)**: prompts are sampled from the query mask via distance-transform refinement (`DSRefineSampler`), splitting `N_p` points into center / boundary / negative points, and are processed by multi-scale prompt encoders with learnable **task tokens** (`N_t`).
- **LoRA adaptation**: only `q/v` projections of the SAM ViT are injected with low-rank adapters (`QVLoRAQKVLinear`); all other backbone weights are frozen.
- **Fisher-weighted aggregation**: client updates are weighted by an empirical Fisher score of the aggregated parameters, improving robustness under client heterogeneity and client dropout.
- **Personalized components**: `prompt_encoder` and `mask_decoder` are always kept local (`LOCAL_PERSONAL_KEYWORDS`), so each client retains a personal head.
- **Optional meta warm-start**: `mamlpp_warm_start` performs a first-order MAML++-style initialization over client support/query pairs before the communication rounds.

## 2. Key Features

| Feature | Where |
| --- | --- |
| Federated orchestration | `federated/fed_client.py`, `federated/fed_server.py`, `federated/train_FedSPD_bs4.py` |
| Parameter upload policy / freezing | `federated/param_filter.py` |
| Structured prompt (DS-refine) sampler | `models/mmseg/models/sam/ds_refine.py` |
| LoRA for SAM q/v projections | `models/mmseg/models/sam/lora.py` |
| Task-token prompt encoder | `models/mmseg/models/sam/prompt_encoder.py` |
| Fisher score estimation | `FedClient.estimate_fisher_score` in `federated/fed_client.py` |
| Federated baselines | `federated/baselines/` |
| Meta warm-start | `federated/mamlpp_init.py` |
| Data loading (paired image/mask folders) | `datasets/` |
| Evaluation metrics (Dice/IoU/BER, COD metrics) | `utils.py`, `sod_metric.py` |

## 3. Requirements

- Python **>= 3.10** (code uses `X | None` type annotations and `torch.func`)
- PyTorch with CUDA support (tested on PyTorch 2.x)
- torchvision, mmcv (`mmcv.runner` is used by `test.py`)

Install base dependencies:

```bash
pip install -r requirements.txt
pip install torch torchvision           # per your CUDA version
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html   # or the build matching your environment
```

`requirements.txt` includes: `imageio, ipython, matplotlib, opencv-python, PyYAML, scikit-learn, scipy, tqdm, numpy, typing, terminaltables, Pillow, onnxruntime, setuptools, timm, easydict, attr, thop, torchsummary`.

## 4. Data Preparation

The pipeline consumes PNG images that have been preprocessed with **SimpleITK** (or equivalent) and resized to the network input size (default `256 x 256`; the legacy single-client configs use `1024 x 1024`). Images are RGB, masks are single-channel (0/1 or 0/255).

### 4.1 Federated setting (required directory layout)

Point `data_path` in the config to a directory containing one subdirectory per client, each with `support_train` (support set used to build prompts) and `query_train` (query set used for the training loss):

```
<data_path>/
└── client1/
    ├── support_train/
    │   ├── img/      # *.png images
    │   └── mask/     # *.png masks
    └── query_train/
        ├── img/
        └── mask/
```

Each client must have exactly the same filenames in `img` and `mask` (paired). Alternatively, provide an explicit `clients:` list in the config (see `build_client_specs` in `federated/train_FedSPD_bs4.py`).

### 4.2 Legacy / single-client setting

Existing configs (`configs/cod-sam-vit-b.yaml`, `configs/test.yaml`, ...) expect the classic layout `<data_path>/<task>/train/{img,mask}` and `<data_path>/<task>/eval/{img,mask}`.

## 5. Project Structure

```
.
├── README.md
├── requirements.txt
├── configs/                     # YAML configurations
├── datasets/                    # image-folder / paired-image-folders datasets + wrappers
├── models/                      # SAM-based segmentation model
│   ├── sam.py                   #   top-level SAM module (model registry: "sam")
│   ├── iou_loss.py, block.py, bn_helper.py
│   └── mmseg/                   #   mmseg-style ViT, prompt encoder, LoRA, DS-refine, decoder
├── federated/
│   ├── train_FedSPD_bs4.py      #   main federated training entry
│   ├── fed_client.py            #   FedClient (local training, fisher, prototypes, ALA, ...)
│   ├── fed_server.py            #   FedServer (aggregation)
│   ├── fed_utils.py             #   seeding, losses, state dict ops, csv logging
│   ├── param_filter.py          #   trainable / upload / personal parameter keywords
│   ├── mamlpp_init.py           #   optional MAML++ meta warm-start
│   ├── visualization.py
│   ├── baselines/               #   FedAvg, FedProx, FedPer, FedMix, FedBABU, FedFomo, FedProto, FedALA
│   └── configs/fedspd_bs3.yaml  #   (empty placeholder)
├── scripts/
│   ├── run_comparison.sh        #   all methods on one config
│   ├── run_fisher_dropout.sh    #   FedSPD +/- fisher under client dropout
│   ├── run_sensitivity_np.sh    #   sensitivity to N_p (num_points)
│   └── run_sensitivity_nt.sh    #   sensitivity to N_t (num_task_tokens)
├── pretrained/                  #   place SAM weights here (sam_vit_b_01ec64.pth)
├── train.py / train_MAML*.py / train_FMAM.py   # legacy single-client / meta-learning variants
├── test.py / test_LARGE*.py     # legacy evaluation variants
├── sod_metric.py                # COD metrics (SM, EM, WFM, MAE)
└── utils.py                     # metrics + helpers
```

## 6. Configuration

All settings are YAML. The canonical FedSPD config is `configs/fedspd.yaml`.

### 6.1 Global hyper-parameters

| Key | Default | Meaning |
| --- | --- | --- |
| `seed` | `2026` | global RNG seed |
| `communication_rounds` | `25` | number of federated rounds |
| `local_epochs` | `1` | local epochs per round per client |
| `max_steps_per_client` | `null` | optional cap on local steps per client |
| `drop_clients_per_round` | `0` | number of clients randomly dropped per round |
| `out_dir` | `./work_dirs/fedspd` | output directory |
| `data_path` | - | root of client data (Section 4.1) |
| `client_names` | `[client1..5]` | client sub-directories under `data_path` |
| `sam_checkpoint` | - | path to `sam_vit_b_01ec64.pth` |

### 6.2 Method-specific hyper-parameters

| Key | Default | Used by |
| --- | --- | --- |
| `feature_align_lambda` | `0.05` | FedSPD feature-alignment regularizer |
| `fedprox_mu` | `0.01` | FedProx proximal term |
| `fedmix_alpha` / `fedmix_weak_weight` | `0.4` / `0.1` | FedMix mixup / weak-supervision weight |
| `fedproto_lambda` | `0.05` | FedProto prototype alignment |
| `fedfomo_M` | `3` | FedFomo number of model candidates |
| `fedbabu_head_epochs` | `1` | FedBABU post-training head epochs |
| `fedala_steps/lr/batches/layer_idx/eta` | `5 / 0.05 / 2 / 0 / 1.0` | FedALA adaptive layer aggregation |
| `mamlpp_steps/inner_lr/outer_lr` | `0 / 1e-3 / 1e-1` | optional MAML++ warm-start |

### 6.3 Model (`model.args.encoder_mode`)

| Key | Default | Meaning |
| --- | --- | --- |
| `inp_size` | `256` | network input resolution |
| `use_lora` / `lora_rank` / `lora_alpha` / `lora_dropout` | `true / 4 / 8 / 0.0` | LoRA on ViT q/v projections |
| `num_points` | `40` | `N_p`, number of DS-refine sampled prompt points |
| `num_task_tokens` | `10` | `N_t`, number of learnable task tokens |
| `input_type` / `freq_nums` / `prompt_type` | `fft / 0.25 / highpass` | handcrafted high-pass prompt generator |

## 7. Training

### 7.1 FedSPD (main method)

```bash
python federated/train_FedSPD_bs4.py \
  --config configs/fedspd.yaml \
  --algorithm fedspd \
  --out_dir work_dirs/fedspd
```

### 7.2 Federated baselines

Any `--algorithm` among: `local, fedavg, fedprox, fedper, fedmix, fedbabu, fedfomo, fedproto, fedala`.

```bash
python federated/train_FedSPD_bs4.py --config configs/fedspd.yaml --algorithm fedavg --out_dir work_dirs/fedavg
```

Or run all of them sequentially:

```bash
bash scripts/run_comparison.sh configs/fedspd.yaml
```

### 7.3 CLI options of the federated entry

| Option | Meaning |
| --- | --- |
| `--config` | path to YAML config |
| `--algorithm` | method to run (see above) |
| `--rounds` | override `communication_rounds` |
| `--num_points` | override `N_p` (sensitivity analysis) |
| `--num_task_tokens` | override `N_t` (sensitivity analysis) |
| `--out_dir` | override output directory |
| `--drop_clients` | override client dropout per round |
| `--no_fisher` | disable Fisher-weighted aggregation (FedSPD ablated variant) |
| `--mamlpp_steps` | override meta warm-start steps |

### 7.4 Paper experiment scripts

```bash
# Table: all baselines vs. FedSPD
bash scripts/run_comparison.sh

# Ablation: Fisher weighting under client dropout (0/1/2 dropped clients)
bash scripts/run_fisher_dropout.sh

# Sensitivity analysis of N_p
bash scripts/run_sensitivity_np.sh

# Sensitivity analysis of N_t
bash scripts/run_sensitivity_nt.sh
```

### 7.5 Legacy single-client / meta-learning training (pre-training)

```bash
# OSAM pre-training on a single dataset
python train.py --config configs/cod-sam-vit-b.yaml

# MAML / FMAM variants (support-query meta-learning)
python train_MAML.py --config configs/cod-sam-vit-b.yaml
python train_FMAM.py  --config configs/FMAM-no-meta.yaml
```

## 8. Evaluation

The federated entry logs per-client and global-average **Dice** after every round. For the legacy pipeline, `test.py` evaluates the trained checkpoint:

```bash
python test.py --config configs/test.yaml --model /path/to/model_epoch_best_dice.pth
```

Metrics supported by `utils.py`:
- `ber`: Dice, IoU, BER (balanced error rate)
- `f1`: F1, AUC
- `fmeasure`: F-measure, MAE
- `cod`: S-measure (SM), E-measure (EM), weighted F-measure (WFM), MAE (via `sod_metric.py`)

## 9. Outputs

Federated runs write into `<out_dir>`:
- `{algorithm}_round_{r}.pth` — global model checkpoint per round
- `{algorithm}_history.csv` — per-client loss / Dice / fisher score and global-average Dice per round
- `{algorithm}_params.csv` — parameter summary (total/trainable params, communication cost in MB, number of aggregated keys)

## 10. Method Notes

1. **Parameter split** (`federated/param_filter.py`): `lora_*` weights are uploaded and aggregated (`FEDSPD_UPLOAD_KEYWORDS`); `prompt_encoder` and `mask_decoder` are personal and never leave the client (`LOCAL_PERSONAL_KEYWORDS`).
2. **Aggregation** (`federated/fed_server.py`): `weighted_delta_update` applies a weighted delta to the current global state. For FedSPD the weights are Fisher scores (gradient-norm squared of the aggregated parameters on a local batch).
3. **Local loss** (`federated/fed_client.py`): BCE + IoU (`bce_iou_loss`), optionally plus feature alignment to the frozen global reference (`feature_align_lambda`), plus the FedProx proximal term when selected.
4. **Prompt source**: when `support_inp` is provided, point embeddings are sampled from support features instead of query features, enabling prompt transfer from support to query.

## 11. License

This project is released for research purposes only. The SAM components are derived from Meta's SAM under its original license (see file headers in `models/mmseg/models/sam/`).


*README prepared to accompany the FedSPD release. Replace the placeholder paths, pretrained checkpoint, and citation fields with your final values before public release.*
