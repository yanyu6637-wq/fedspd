from __future__ import annotations

import copy
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader

from .fed_utils import bce_iou_loss, load_partial_state, state_dict_cpu


def _to_device(batch: dict, device: torch.device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def mamlpp_warm_start(model: torch.nn.Module,
                      client_loaders: List[tuple[DataLoader, DataLoader]],
                      aggregate_keys: Iterable[str],
                      device: torch.device,
                      steps: int = 0,
                      inner_lr: float = 1e-3,
                      outer_lr: float = 1e-4,
                      num_points: int = 40):
    if steps <= 0:
        return model
    keys = list(aggregate_keys)
    model = model.to(device)

    for _ in range(int(steps)):
        base = state_dict_cpu(model, keys=keys)
        delta = {k: torch.zeros_like(v.float()) for k, v in base.items()}
        used = 0

        for support_loader, query_loader in client_loaders:
            try:
                support = _to_device(next(iter(support_loader)), device)
                query = _to_device(next(iter(query_loader)), device)
            except StopIteration:
                continue

            fast_model = copy.deepcopy(model).to(device)
            fast_model.train()
            opt = torch.optim.SGD([p for p in fast_model.parameters() if p.requires_grad], lr=inner_lr)
            opt.zero_grad(set_to_none=True)
            pred_s = fast_model(support['inp'], support['gt'], support['gt'], num_points=num_points, support_inp=support['inp'])
            inner_loss = bce_iou_loss(pred_s, support['gt'])
            inner_loss.backward()
            opt.step()

            fast_model.zero_grad(set_to_none=True)
            pred_q = fast_model(query['inp'], query['gt'], support['gt'], num_points=num_points, support_inp=support['inp'])
            outer_loss = bce_iou_loss(pred_q, query['gt'])
            outer_loss.backward()
            # first-order MAML++ style update: use adapted weights as meta direction
            adapted = state_dict_cpu(fast_model, keys=keys)
            for k in keys:
                delta[k] += adapted[k].float() - base[k].float()
            used += 1

        if used > 0:
            new_state = {k: base[k].float() + outer_lr * delta[k] / float(used) for k in keys}
            load_partial_state(model, new_state, strict=False)
    return model
