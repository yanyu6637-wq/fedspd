from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import torch


def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str | os.PathLike):
    Path(path).mkdir(parents=True, exist_ok=True)


def sigmoid_dice(pred_logits: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> float:
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    gt = (gt > 0.5).float()
    dims = tuple(range(1, pred.dim()))
    inter = (pred * gt).sum(dim=dims)
    union = pred.sum(dim=dims) + gt.sum(dim=dims)
    dice = (2.0 * inter + eps) / (union + eps)
    return float(dice.mean().detach().cpu())


def bce_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
    prob = torch.sigmoid(pred)
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter
    iou = 1.0 - (inter + 1e-7) / (union + 1e-7)
    return bce + iou.mean()


def state_dict_cpu(model: torch.nn.Module, keys: Iterable[str] | None = None) -> Dict[str, torch.Tensor]:
    sd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    if keys is None:
        return {k: v.detach().cpu().clone() for k, v in sd.items()}
    key_set = set(keys)
    return {k: v.detach().cpu().clone() for k, v in sd.items() if k in key_set}


def load_partial_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor], strict: bool = False):
    target = model.module if hasattr(model, 'module') else model
    current = target.state_dict()
    current.update({k: v.to(current[k].device) for k, v in state.items() if k in current})
    target.load_state_dict(current, strict=strict)



def normalize_weights(weights):
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return []
    s = float(w.sum())
    if abs(s) < 1e-12:
        w = np.ones_like(w) / max(len(w), 1)
    else:
        w = w / s
    return [float(x) for x in w]


def weighted_state_average(states: list[Dict[str, torch.Tensor]], weights: list[float]) -> Dict[str, torch.Tensor]:
    if len(states) == 0:
        raise ValueError('states cannot be empty.')
    w = normalize_weights(weights)
    out = {}
    keys = list(states[0].keys())
    for key in keys:
        acc = None
        for i, st in enumerate(states):
            if key not in st:
                continue
            val = st[key].detach().cpu().float() * w[i]
            acc = val if acc is None else acc + val
        if acc is not None:
            out[key] = acc
    return out


def blend_states(left: Dict[str, torch.Tensor], right: Dict[str, torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
    a = float(alpha)
    out = {}
    for k, v in left.items():
        if k in right:
            out[k] = a * v.detach().cpu().float() + (1.0 - a) * right[k].detach().cpu().float()
        else:
            out[k] = v.detach().cpu().clone()
    return out


def weighted_average(states: list[Dict[str, torch.Tensor]], weights: list[float]) -> Dict[str, torch.Tensor]:
    if len(states) == 0:
        raise ValueError('states cannot be empty.')
    w = normalize_weights(weights)
    out = {}
    for key in states[0].keys():
        acc = None
        for i, st in enumerate(states):
            val = st[key].float() * w[i]
            acc = val if acc is None else acc + val
        out[key] = acc
    return out




def weighted_delta_update(base_state: Dict[str, torch.Tensor], local_states: list[Dict[str, torch.Tensor]], weights: list[float]) -> Dict[str, torch.Tensor]:
    if len(local_states) == 0:
        raise ValueError('local_states cannot be empty.')
    w = normalize_weights(weights)
    out = {}
    for key in local_states[0].keys():
        base = base_state[key].detach().cpu().float()
        delta = torch.zeros_like(base)
        for i, st in enumerate(local_states):
            delta = delta + w[i] * (st[key].float() - base)
        out[key] = base + delta
    return out


def save_csv(path: str | os.PathLike, rows: list[dict]):
    ensure_dir(Path(path).parent)
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
