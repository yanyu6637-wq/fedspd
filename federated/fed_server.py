from __future__ import annotations

from typing import Dict, List
import torch

from .fed_utils import load_partial_state, state_dict_cpu, weighted_delta_update, normalize_weights


class FedServer:
    def __init__(self, model: torch.nn.Module, aggregate_keys: List[str], algorithm: str = 'fedspd', use_fisher: bool = True):
        self.model = model
        self.aggregate_keys = list(aggregate_keys)
        self.algorithm = algorithm.lower()
        self.use_fisher = bool(use_fisher)

    def _client_weights(self, client_infos: List[Dict]) -> List[float]:
        if self.algorithm == 'fedspd' and self.use_fisher:
            vals = [max(float(info.get('fisher_score', 0.0)), 1e-12) for info in client_infos]
            return normalize_weights(vals)
        vals = [max(float(info.get('num_samples', info.get('metrics', {}).get('num_samples', 1))), 1.0) for info in client_infos]
        return normalize_weights(vals)

    def aggregate(self, local_states: List[Dict[str, torch.Tensor]], client_infos: List[Dict]):
        if self.algorithm == 'local' or len(local_states) == 0:
            return
        weights = self._client_weights(client_infos)
        base_state = state_dict_cpu(self.model, keys=self.aggregate_keys)
        new_state = weighted_delta_update(base_state, local_states, weights)
        load_partial_state(self.model, new_state, strict=False)
