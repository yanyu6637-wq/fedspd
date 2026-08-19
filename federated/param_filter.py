from __future__ import annotations

from typing import Iterable, List
import torch

# FedSPD only sends LoRA; fisher is just a scalar
FEDSPD_UPLOAD_KEYWORDS = ('lora_',)
LOCAL_PERSONAL_KEYWORDS = ('prompt_encoder', 'mask_decoder')
TRAINABLE_KEYWORDS = FEDSPD_UPLOAD_KEYWORDS + LOCAL_PERSONAL_KEYWORDS

# baselines usually average the tuned parts
BASELINE_UPLOAD_KEYWORDS = TRAINABLE_KEYWORDS


def _state_dict(model: torch.nn.Module):
    return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()


def get_aggregated_param_names(model: torch.nn.Module, keywords: Iterable[str] = FEDSPD_UPLOAD_KEYWORDS) -> List[str]:
    kws = tuple(keywords)
    sd = _state_dict(model)
    return [k for k in sd.keys() if any(kw in k for kw in kws)]


def get_trainable_param_names(model: torch.nn.Module, keywords: Iterable[str] = TRAINABLE_KEYWORDS) -> List[str]:
    kws = tuple(keywords)
    return [name for name, p in model.named_parameters() if any(kw in name for kw in kws)]


def freeze_for_fedspd(model: torch.nn.Module, keywords: Iterable[str] = TRAINABLE_KEYWORDS):
    kws = tuple(keywords)
    for name, p in model.named_parameters():
        p.requires_grad = any(kw in name for kw in kws)
    return model


def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_total_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def communication_cost_mb(model: torch.nn.Module, keywords: Iterable[str] = FEDSPD_UPLOAD_KEYWORDS, bytes_per_param: int = 4) -> float:
    kws = tuple(keywords)
    n = 0
    for name, p in model.named_parameters():
        if any(kw in name for kw in kws):
            n += p.numel()
    # fisher scalar is tiny
    return n * bytes_per_param / (1024 ** 2)
