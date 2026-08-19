
from __future__ import annotations

import copy
from typing import Dict, Iterable, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.func import functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call

from .fed_utils import bce_iou_loss, load_partial_state, sigmoid_dice, state_dict_cpu
from .param_filter import get_trainable_param_names, LOCAL_PERSONAL_KEYWORDS


class FedClient:
    def __init__(self, client_id: str, support_loader: DataLoader, query_loader: DataLoader,
                 device: torch.device, lr: float = 2e-4, local_epochs: int = 1,
                 feature_align_lambda: float = 0.05, fedprox_mu: float = 0.0,
                 num_points: int = 40, max_steps: int | None = None,
                 supervision: str = 'pixel'):
        self.client_id = str(client_id)
        self.support_loader = support_loader
        self.query_loader = query_loader
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.feature_align_lambda = feature_align_lambda
        self.fedprox_mu = fedprox_mu
        self.num_points = num_points
        self.max_steps = max_steps
        self.supervision = str(supervision or 'pixel').lower()
        self.personal_state: Dict[str, torch.Tensor] = {}
        self._last_full_state: Dict[str, torch.Tensor] = {}

    @property
    def num_samples(self) -> int:
        try:
            return max(len(self.query_loader.dataset), 1)
        except Exception:
            return 1

    @staticmethod
    def _batch_to_device(batch: dict, device: torch.device):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    @staticmethod
    def _get_feature(model):
        target = model.module if hasattr(model, 'module') else model
        return getattr(target, 'features', None)

    def _load_personal_part(self, model: torch.nn.Module):
        if self.personal_state:
            load_partial_state(model, self.personal_state, strict=False)

    def _save_personal_part(self, model: torch.nn.Module, aggregate_keys: Iterable[str]):
        agg = set(aggregate_keys)
        names = [n for n in get_trainable_param_names(model) if n not in agg]
        self.personal_state = state_dict_cpu(model, keys=names)

    def _proto_loss(self, feature: torch.Tensor | None, gt: torch.Tensor, global_protos: Dict[str, torch.Tensor] | None) -> torch.Tensor:
        if feature is None or not global_protos:
            return gt.new_tensor(0.0)
        gt_small = F.interpolate(gt.float(), size=feature.shape[-2:], mode='nearest')
        fg = (gt_small > 0.5).float()
        bg = 1.0 - fg
        losses = []
        for key, mask in [('fg', fg), ('bg', bg)]:
            denom = mask.sum(dim=(0, 2, 3)).clamp_min(1.0)
            proto = (feature * mask).sum(dim=(0, 2, 3)) / denom
            if key in global_protos and global_protos[key] is not None:
                losses.append(F.mse_loss(proto, global_protos[key].to(feature.device).detach()))
        if not losses:
            return gt.new_tensor(0.0)
        return sum(losses) / len(losses)

    def _run_forward_loss(self, model, inp, gt, support_mask, support_inp, global_protos=None, proto_lambda: float = 0.0):
        pred = model(inp, gt, support_mask, num_points=self.num_points, support_inp=support_inp)
        loss = bce_iou_loss(pred, gt)
        if proto_lambda > 0:
            loss = loss + float(proto_lambda) * self._proto_loss(self._get_feature(model), gt, global_protos)
        return pred, loss

    def train_one_round(self, global_model: torch.nn.Module, aggregate_keys: Iterable[str],
                        init_state: Dict[str, torch.Tensor] | None = None,
                        feature_align_lambda: float | None = None,
                        fedprox_mu: float | None = None,
                        mixup_alpha: float = 0.0,
                        global_protos: Dict[str, torch.Tensor] | None = None,
                        proto_lambda: float = 0.0,
                        save_personal: bool = True,
                        freeze_keywords: Iterable[str] = ()) -> Tuple[Dict, Dict]:
        aggregate_keys = list(aggregate_keys)
        model = copy.deepcopy(global_model).to(self.device)
        if init_state:
            load_partial_state(model, init_state, strict=False)
        else:
            self._load_personal_part(model)

        freeze_keywords = tuple(freeze_keywords)
        if freeze_keywords:
            for n, p in model.named_parameters():
                if any(k in n for k in freeze_keywords):
                    p.requires_grad = False

        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=5e-4) if params else None
        fa_lambda = self.feature_align_lambda if feature_align_lambda is None else float(feature_align_lambda)
        prox_mu = self.fedprox_mu if fedprox_mu is None else float(fedprox_mu)

        global_state = {k: v.detach().clone().to(self.device) for k, v in model.state_dict().items() if k in aggregate_keys}
        metrics = {'client_id': self.client_id, 'loss': 0.0, 'dice': 0.0, 'num_steps': 0, 'num_samples': self.num_samples}

        ref_model = global_model.to(self.device)
        ref_model.eval()

        for _ in range(self.local_epochs):
            iterator = zip(self.support_loader, self.query_loader)
            for step, (support, query) in enumerate(tqdm(iterator, desc=f'client {self.client_id}', leave=False)):
                if self.max_steps is not None and step >= self.max_steps:
                    break
                support = self._batch_to_device(support, self.device)
                query = self._batch_to_device(query, self.device)
                inp, gt = query['inp'], query['gt']
                support_inp, support_mask = support['inp'], support['gt']

                if opt is None:
                    continue
                opt.zero_grad(set_to_none=True)

                with torch.no_grad():
                    ref_feat = None
                    if fa_lambda > 0:
                        _ = ref_model(inp, gt, support_mask, num_points=self.num_points, support_inp=support_inp)
                        ref_feat = self._get_feature(ref_model)
                        ref_feat = None if ref_feat is None else ref_feat.detach()

                if mixup_alpha > 0 and inp.shape[0] > 1:
                    beta = torch.distributions.Beta(float(mixup_alpha), float(mixup_alpha))
                    lam = float(beta.sample().item())
                    perm = torch.randperm(inp.shape[0], device=inp.device)
                    train_inp = lam * inp + (1.0 - lam) * inp[perm]
                    train_gt = lam * gt + (1.0 - lam) * gt[perm]
                    pred, loss = self._run_forward_loss(model, train_inp, train_gt, support_mask, support_inp, global_protos, proto_lambda)
                else:
                    pred, loss = self._run_forward_loss(model, inp, gt, support_mask, support_inp, global_protos, proto_lambda)

                cur_feat = self._get_feature(model)
                if ref_feat is not None and cur_feat is not None and fa_lambda > 0:
                    loss = loss + fa_lambda * F.mse_loss(cur_feat, ref_feat)

                if prox_mu > 0:
                    prox = 0.0
                    named_params = dict(model.named_parameters())
                    for name in aggregate_keys:
                        p = named_params.get(name, None)
                        if p is not None and p.requires_grad and name in global_state:
                            prox = prox + torch.sum((p - global_state[name]) ** 2)
                    loss = loss + 0.5 * prox_mu * prox

                loss.backward()
                opt.step()

                with torch.no_grad():
                    pred_eval = model(inp, gt, support_mask, num_points=self.num_points, support_inp=support_inp)
                metrics['loss'] += float(loss.detach().cpu())
                metrics['dice'] += sigmoid_dice(pred_eval.detach(), gt.detach())
                metrics['num_steps'] += 1

        if metrics['num_steps'] > 0:
            metrics['loss'] /= metrics['num_steps']
            metrics['dice'] /= metrics['num_steps']

        fisher_score = self.estimate_fisher_score(model, aggregate_keys)
        local_state = state_dict_cpu(model, keys=aggregate_keys)
        self._last_full_state = state_dict_cpu(model, keys=get_trainable_param_names(model))
        if save_personal:
            self._save_personal_part(model, aggregate_keys)
        return local_state, {'metrics': metrics, 'fisher_score': fisher_score, 'num_samples': self.num_samples}

    @staticmethod
    def _mask_to_box(mask: torch.Tensor) -> torch.Tensor:
        # produce a rectangular foreground mask from the available mask. Used when a client is configured as box-supervised.
        box = torch.zeros_like(mask)
        for b in range(mask.shape[0]):
            pos = torch.nonzero(mask[b, 0] > 0.5, as_tuple=False)
            if pos.numel() == 0:
                continue
            y0, x0 = pos.min(dim=0).values.tolist()
            y1, x1 = pos.max(dim=0).values.tolist()
            box[b, :, y0:y1 + 1, x0:x1 + 1] = 1.0
        return box

    @staticmethod
    def _tv_loss(prob: torch.Tensor) -> torch.Tensor:
        dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :]).mean()
        dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1]).mean()
        return dx + dy

    def _mixed_label_loss(self, pred: torch.Tensor, gt: torch.Tensor, mode: str, weak_weight: float = 0.1) -> torch.Tensor:
        mode = (mode or 'pixel').lower()
        if mode in ('pixel', 'mask', 'strong'):
            return bce_iou_loss(pred, gt)
        prob = torch.sigmoid(pred)
        if mode in ('box', 'bbox', 'weak'):
            box = self._mask_to_box(gt)
            # Outside the box should be background; inside the box must contain foreground somewhere.
            outside = 1.0 - box
            loss_map = F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred), reduction='none')
            loss_out = (loss_map * outside).sum() / outside.sum().clamp_min(1.0)
            inside_prob = (prob * box).flatten(1).amax(dim=1).clamp(1e-6, 1.0 - 1e-6)
            has_fg = (gt.flatten(1).sum(dim=1) > 0).float()
            loss_in = F.binary_cross_entropy(inside_prob, has_fg)
            return loss_out + loss_in + float(weak_weight) * self._tv_loss(prob)
        if mode in ('image', 'tag', 'image_level'):
            has_fg = (gt.flatten(1).sum(dim=1) > 0).float()
            img_score = prob.flatten(1).amax(dim=1).clamp(1e-6, 1.0 - 1e-6)
            return F.binary_cross_entropy(img_score, has_fg) + float(weak_weight) * self._tv_loss(prob)
        return bce_iou_loss(pred, gt)

    def train_fedmix_round(self, global_model: torch.nn.Module, aggregate_keys: Iterable[str],
                           init_state: Dict[str, torch.Tensor] | None = None,
                           supervision: str | None = None,
                           mixup_alpha: float = 0.2,
                           weak_weight: float = 0.1,
                           save_personal: bool = True) -> Tuple[Dict, Dict]:
        # fedmix branch
        mode = (supervision or self.supervision or 'pixel').lower()
        aggregate_keys = list(aggregate_keys)
        model = copy.deepcopy(global_model).to(self.device)
        if init_state:
            load_partial_state(model, init_state, strict=False)
        else:
            self._load_personal_part(model)
        model.train()
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=self.lr, momentum=0.9, weight_decay=5e-4)
        metrics = {'client_id': self.client_id, 'loss': 0.0, 'dice': 0.0, 'num_steps': 0, 'num_samples': self.num_samples}

        for _ in range(self.local_epochs):
            for step, (support, query) in enumerate(tqdm(zip(self.support_loader, self.query_loader), desc=f'fedmix {self.client_id}', leave=False)):
                if self.max_steps is not None and step >= self.max_steps:
                    break
                support = self._batch_to_device(support, self.device)
                query = self._batch_to_device(query, self.device)
                inp, gt = query['inp'], query['gt']
                support_inp, support_mask = support['inp'], support['gt']
                opt.zero_grad(set_to_none=True)
                if mode in ('pixel', 'mask', 'strong') and mixup_alpha > 0 and inp.shape[0] > 1:
                    lam = float(torch.distributions.Beta(float(mixup_alpha), float(mixup_alpha)).sample().item())
                    perm = torch.randperm(inp.shape[0], device=inp.device)
                    train_inp = lam * inp + (1.0 - lam) * inp[perm]
                    train_gt = lam * gt + (1.0 - lam) * gt[perm]
                    pred = model(train_inp, train_gt, support_mask, num_points=self.num_points, support_inp=support_inp)
                    loss = self._mixed_label_loss(pred, train_gt, mode, weak_weight=weak_weight)
                else:
                    pred = model(inp, gt, support_mask, num_points=self.num_points, support_inp=support_inp)
                    loss = self._mixed_label_loss(pred, gt, mode, weak_weight=weak_weight)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    pred_eval = model(inp, gt, support_mask, num_points=self.num_points, support_inp=support_inp)
                metrics['loss'] += float(loss.detach().cpu())
                metrics['dice'] += sigmoid_dice(pred_eval.detach(), gt.detach())
                metrics['num_steps'] += 1
        if metrics['num_steps'] > 0:
            metrics['loss'] /= metrics['num_steps']
            metrics['dice'] /= metrics['num_steps']
        # FedMix uses adaptive client weights; reliable supervision and smaller local loss get larger weight.
        rel = {'pixel': 1.0, 'mask': 1.0, 'strong': 1.0, 'box': 0.75, 'bbox': 0.75, 'weak': 0.75, 'image': 0.45, 'tag': 0.45, 'image_level': 0.45}.get(mode, 1.0)
        fedmix_score = rel * float(self.num_samples) / max(float(metrics['loss']), 1e-6)
        local_state = state_dict_cpu(model, keys=aggregate_keys)
        if save_personal:
            self._save_personal_part(model, aggregate_keys)
        return local_state, {'metrics': metrics, 'fedmix_score': fedmix_score, 'supervision': mode, 'num_samples': self.num_samples}

    def estimate_fisher_score(self, model: torch.nn.Module, aggregate_keys: Iterable[str]) -> float:
        model.train()
        agg = set(aggregate_keys)
        if not agg:
            return 0.0
        try:
            support, query = next(iter(zip(self.support_loader, self.query_loader)))
        except StopIteration:
            return 0.0
        support = self._batch_to_device(support, self.device)
        query = self._batch_to_device(query, self.device)
        model.zero_grad(set_to_none=True)
        pred = model(query['inp'], query['gt'], support['gt'], num_points=self.num_points, support_inp=support['inp'])
        loss = bce_iou_loss(pred, query['gt'])
        loss.backward()
        score = 0.0
        for name, p in model.named_parameters():
            if name in agg and p.grad is not None:
                score += float(torch.sum(p.grad.detach() ** 2).cpu())
        return score

    @torch.no_grad()
    def collect_prototypes(self, base_model: torch.nn.Module, init_state: Dict[str, torch.Tensor] | None = None):
        model = copy.deepcopy(base_model).to(self.device)
        if init_state:
            load_partial_state(model, init_state, strict=False)
        else:
            self._load_personal_part(model)
        model.eval()
        sums = {'fg': None, 'bg': None}
        counts = {'fg': 0.0, 'bg': 0.0}
        for step, (support, query) in enumerate(zip(self.support_loader, self.query_loader)):
            if self.max_steps is not None and step >= self.max_steps:
                break
            support = self._batch_to_device(support, self.device)
            query = self._batch_to_device(query, self.device)
            _ = model(query['inp'], query['gt'], support['gt'], num_points=self.num_points, support_inp=support['inp'])
            feat = self._get_feature(model)
            if feat is None:
                continue
            gt_small = F.interpolate(query['gt'].float(), size=feat.shape[-2:], mode='nearest')
            masks = {'fg': (gt_small > 0.5).float(), 'bg': (gt_small <= 0.5).float()}
            for key, mask in masks.items():
                cnt = float(mask.sum().item())
                if cnt <= 0:
                    continue
                val = (feat * mask).sum(dim=(0, 2, 3)).detach().cpu()
                sums[key] = val if sums[key] is None else sums[key] + val
                counts[key] += cnt
        protos = {}
        for key in sums:
            if sums[key] is not None and counts[key] > 0:
                protos[key] = sums[key] / counts[key]
        return protos, counts

    def _param_distance(self, left: Dict[str, torch.Tensor] | None, right: Dict[str, torch.Tensor] | None) -> float:
        if not left or not right:
            return 1.0
        s = 0.0
        for k, v in left.items():
            if k in right:
                d = v.detach().float() - right[k].detach().float()
                s += float(torch.sum(d * d))
        return max(s ** 0.5, 1e-12)

    @torch.no_grad()
    def evaluate_loss_and_dice(self, model: torch.nn.Module) -> Dict[str, float]:
        model = copy.deepcopy(model).to(self.device)
        self._load_personal_part(model)
        model.eval()
        losses, dices = [], []
        for step, (support, query) in enumerate(zip(self.support_loader, self.query_loader)):
            if self.max_steps is not None and step >= self.max_steps:
                break
            support = self._batch_to_device(support, self.device)
            query = self._batch_to_device(query, self.device)
            pred = model(query['inp'], query['gt'], support['gt'], num_points=self.num_points, support_inp=support['inp'])
            losses.append(float(bce_iou_loss(pred, query['gt']).detach().cpu()))
            dices.append(sigmoid_dice(pred.detach(), query['gt']))
        return {'loss': float(sum(losses) / max(len(losses), 1)), 'dice': float(sum(dices) / max(len(dices), 1))}

    @torch.no_grad()
    def evaluate_state(self, base_model: torch.nn.Module, state: Dict[str, torch.Tensor] | None = None):
        model = copy.deepcopy(base_model).to(self.device)
        if state:
            load_partial_state(model, state, strict=False)
        return self.evaluate_loss_and_dice(model)

    def fine_tune_personal_head(self, global_model: torch.nn.Module, epochs: int = 1):
        model = copy.deepcopy(global_model).to(self.device)
        self._load_personal_part(model)
        for name, p in model.named_parameters():
            p.requires_grad = any(k in name for k in LOCAL_PERSONAL_KEYWORDS)
        old_epochs = self.local_epochs
        self.local_epochs = int(epochs)
        self.train_one_round(model, [], init_state=None, feature_align_lambda=0.0, fedprox_mu=0.0, save_personal=True)
        self.local_epochs = old_epochs

    def adapt_ala_state(self, global_model: torch.nn.Module, aggregate_keys: Iterable[str],
                        local_state: Dict[str, torch.Tensor] | None = None,
                        steps: int = 5, lr: float = 0.05, batch_limit: int = 2,
                        layer_idx: int = 0, eta: float = 1.0, rand_percent: float = 0.8) -> Dict[str, torch.Tensor]:
        # local/global blend before update
        aggregate_keys = list(aggregate_keys)
        if len(aggregate_keys) == 0:
            return {}
        model = copy.deepcopy(global_model).to(self.device)
        self._load_personal_part(model)
        if local_state:
            load_partial_state(model, local_state, strict=False)
        model.train()
        named_params = dict(model.named_parameters())
        keys = [k for k in aggregate_keys if k in named_params]
        if not keys:
            return local_state or state_dict_cpu(global_model, keys=aggregate_keys)
        start = min(max(int(layer_idx), 0), len(keys))
        preserved = keys[:start]
        learned = keys[start:]
        global_state = state_dict_cpu(global_model, keys=keys)
        local_base = state_dict_cpu(model, keys=keys)
        if not learned:
            return local_base
        # FedALA samples a small local subset to learn aggregation weights.
        ala_data = []
        for support, query in zip(self.support_loader, self.query_loader):
            ala_data.append((self._batch_to_device(support, self.device), self._batch_to_device(query, self.device)))
            if len(ala_data) >= max(int(batch_limit), 1):
                break
        if not ala_data:
            return local_base
        alpha_logits = torch.nn.ParameterDict()
        for key in learned:
            g = global_state[key]
            alpha_logits[key.replace('.', '#')] = torch.nn.Parameter(torch.zeros_like(g, dtype=torch.float32, device=self.device))
        opt = torch.optim.SGD(alpha_logits.parameters(), lr=float(lr))
        for _ in range(max(int(steps), 1)):
            for support, query in ala_data:
                base_params = {n: p.detach() for n, p in model.named_parameters()}
                buffers = {n: b.detach() for n, b in model.named_buffers()}
                for key in preserved:
                    base_params[key] = local_base[key].to(self.device).detach()
                for key in learned:
                    g = global_state[key].to(self.device).detach().float()
                    l = local_base[key].to(self.device).detach().float()
                    alpha = torch.sigmoid(alpha_logits[key.replace('.', '#')])
                    base_params[key] = l + float(eta) * alpha * (g - l)
                pred = functional_call(
                    model,
                    {**base_params, **buffers},
                    (query['inp'], query['gt'], support['gt']),
                    {'num_points': self.num_points, 'support_inp': support['inp']},
                )
                loss = bce_iou_loss(pred, query['gt'])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        out = {}
        for key in preserved:
            out[key] = local_base[key].detach().cpu().clone()
        for key in learned:
            g = global_state[key].detach().cpu().float()
            l = local_base[key].detach().cpu().float()
            alpha = torch.sigmoid(alpha_logits[key.replace('.', '#')]).detach().cpu()
            out[key] = l + float(eta) * alpha * (g - l)
        return out

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module) -> Dict[str, float]:
        res = self.evaluate_loss_and_dice(model)
        return {'client_id': self.client_id, 'dice': res['dice']}
