
# FedSPD

FedSPD is a research implementation for personalized federated few-shot
medical image segmentation. The framework combines a SAM-based
segmentation backbone with parameter-efficient LoRA adaptation,
structured prompt construction, client-specific components, and adaptive
federated aggregation.

The repository provides the main training pipeline together with
implementations of the principal components used in FedSPD and several
federated learning baselines for comparison and ablation studies.

## Overview

FedSPD is designed for few-shot segmentation under a federated setting,
where each client performs local optimization and the server coordinates
model aggregation without exchanging local image data.

The current implementation includes:

-   a SAM-based segmentation backbone with LoRA adaptation;
-   structured prompt construction from support annotations;
-   task-token based prompt representation;
-   personalized prompt and mask-decoding components;
-   feature-alignment regularization during local training;
-   Fisher-score-based adaptive aggregation;
-   configurable client participation and dropout;
-   several commonly used federated learning baselines;
-   optional meta-learning initialization;
-   scripts for representative comparison, ablation, and sensitivity
    experiments.

## Method

### Structured Prompt Segmentation

The local segmentation model uses support images and annotations to
construct structured prompts. The prompt sampler extracts complementary
foreground, boundary, and background information and maps the sampled
locations to support features. Learnable task tokens are then used to
obtain a compact task representation for query-image segmentation.

The relevant implementation is mainly located in:

``` text
models/mmseg/models/sam/ds_refine.py
models/mmseg/models/sam/prompt_encoder.py
models/mmseg/models/sam/
```

### LoRA Adaptation

LoRA modules are inserted into the query and value projections of the
SAM image encoder. The pretrained backbone can therefore retain its
general representation while a relatively small set of parameters is
adapted to the target task.

Implementation:

``` text
models/mmseg/models/sam/lora.py
federated/param_filter.py
```

### Personalized Federated Learning

During federated training, LoRA parameters form the main shared
parameter subset used for server aggregation. Prompt-related and
mask-decoding components are maintained locally to preserve
client-specific representations.

The parameter policy is defined in:

``` text
federated/param_filter.py
```

### Feature Alignment

A feature-alignment regularizer can be applied during local optimization
to reduce excessive deviation between the locally adapted representation
and the global reference representation.

Implementation:

``` text
federated/fed_client.py
```

### Adaptive Aggregation

FedSPD supports Fisher-score-based weighting of client updates. The
score is estimated from gradients of the shared parameters on local data
and is used to adjust the relative contribution of participating clients
during aggregation.

Implementation:

``` text
federated/fed_client.py
federated/fed_server.py
```

### Meta-learning Initialization

The repository also contains an optional first-order meta-learning
warm-start routine that can be enabled when required by the experimental
configuration.

Implementation:

``` text
federated/mamlpp_init.py
```

## Repository Structure

``` text
.
├── configs/                     # experiment configurations
├── datasets/                    # dataset loaders and wrappers
├── federated/
│   ├── train_FedSPD_bs4.py      # main federated training entry
│   ├── fed_client.py            # local client optimization
│   ├── fed_server.py            # server aggregation
│   ├── fed_utils.py             # federated utilities
│   ├── param_filter.py          # parameter sharing policy
│   ├── mamlpp_init.py           # optional meta-learning initialization
│   └── baselines/               # federated baseline implementations
├── models/                      # SAM-based segmentation model
│   └── mmseg/models/sam/        # LoRA, prompt modules and SAM components
├── scripts/                     # experiment helper scripts
├── pretrained/                  # pretrained model weights
├── train.py                     # single-client training entry
├── test.py                      # evaluation entry
├── requirements.txt
└── README.md
```

## Requirements

A CUDA-enabled PyTorch environment is recommended.

Main dependencies include:

-   Python 3.10 or later
-   PyTorch
-   torchvision
-   NumPy
-   SciPy
-   OpenCV
-   PyYAML
-   scikit-learn
-   timm
-   mmcv

Install the repository dependencies with:

``` bash
pip install -r requirements.txt
```

PyTorch, torchvision, and mmcv should be installed according to the CUDA
and PyTorch versions available in the local environment.

## Data

The federated training pipeline expects preprocessed image-mask pairs
organized for each client. Images and masks should use matching
filenames.

A minimal client directory can be organized as:

``` text
<data_path>/
└── client1/
    ├── support_train/
    │   ├── img/
    │   └── mask/
    └── query_train/
        ├── img/
        └── mask/
```

Additional clients can follow the same structure. Dataset paths and
client names can be specified in the experiment configuration.

## Configuration

The main federated configuration is:

``` text
configs/fedspd.yaml
```

Commonly adjusted settings include:

``` yaml
communication_rounds: 25
local_epochs: 1
drop_clients_per_round: 0

feature_align_lambda: 0.05

model:
  args:
    encoder_mode:
      inp_size: 256
      use_lora: true
      lora_rank: 4
      lora_alpha: 8
      num_points: 40
      num_task_tokens: 10
```

These values can be modified according to the experimental setting and
available computing resources.

## Training

The main FedSPD training entry is:

``` bash
python federated/train_FedSPD_bs4.py \
  --config configs/fedspd.yaml \
  --algorithm fedspd \
  --out_dir work_dirs/fedspd
```

The training program supports command-line overrides for commonly used
settings, including the number of communication rounds, structured
prompt points, task tokens, client dropout, output directory, and
optional meta-learning initialization.

For example:

``` bash
python federated/train_FedSPD_bs4.py \
  --config configs/fedspd.yaml \
  --algorithm fedspd \
  --rounds 25 \
  --out_dir work_dirs/fedspd
```

## Federated Baselines

The same training framework contains implementations of several
federated learning strategies, including:

``` text
local
fedavg
fedprox
fedper
fedmix
fedbabu
fedfomo
fedproto
fedala
```

A baseline can be selected through the `--algorithm` option:

``` bash
python federated/train_FedSPD_bs4.py \
  --config configs/fedspd.yaml \
  --algorithm fedavg \
  --out_dir work_dirs/fedavg
```

The baseline implementations use the common training framework in this
repository so that different aggregation strategies can be evaluated
under a consistent model and data interface.

## Experiment Scripts

Several helper scripts are included for representative federated
comparisons and analysis:

``` text
scripts/run_comparison.sh
scripts/run_fisher_dropout.sh
scripts/run_sensitivity_np.sh
scripts/run_sensitivity_nt.sh
```

They provide convenient entry points for running baseline comparisons,
client-dropout experiments, and sensitivity studies of the structured
prompt and task-token settings.

The scripts are intended as experiment helpers. Exact results can depend
on the dataset preprocessing, environment, pretrained weights, random
seed, and training configuration.

## Evaluation and Outputs

During federated training, the program records client-level training
statistics and global-average segmentation performance. Checkpoints and
training histories are written to the selected output directory.

Typical outputs include:

``` text
{algorithm}_round_{r}.pth
{algorithm}_history.csv
{algorithm}_params.csv
```

The repository also contains evaluation utilities for segmentation
metrics and additional single-client evaluation scripts.

## Pretrained Weights

The SAM checkpoint can be placed under:

``` text
pretrained/
```

and specified through the corresponding configuration field.

The exact checkpoint path should match the local environment.

## Notes

-   Raw medical images are not exchanged between clients during
    federated training.
-   The shared and personalized parameter subsets are controlled by
    `federated/param_filter.py`.
-   FedSPD primarily aggregates LoRA parameters while retaining
    designated prompt and decoding components locally.
-   Fisher weighting can be disabled for ablation experiments.
-   Client dropout can be configured to study training behavior under
    partial participation.
-   Meta-learning initialization is optional and can be enabled through
    the corresponding configuration.
-   Experimental results may vary with preprocessing, software versions,
    hardware, initialization, and random seeds.

## License

This repository is intended for research use. Third-party components and
pretrained models remain subject to their respective licenses.
