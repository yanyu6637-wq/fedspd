
from __future__ import annotations
from typing import Dict
import torch
from federated.fed_utils import weighted_state_average, normalize_weights
from federated.param_filter import BASELINE_UPLOAD_KEYWORDS


class FedFomoStrategy:
    # keep a small client preference table
    name = 'fedfomo'
    aggregate_keywords = BASELINE_UPLOAD_KEYWORDS
    use_fisher = False

    def __init__(self, config=None):
        self.config = config or {}
        self.M = int(self.config.get('fedfomo_M', 3))
        self.client_states: Dict[str, Dict[str, torch.Tensor]] = {}
        self.P: Dict[str, Dict[str, float]] = {}

    def _ensure_clients(self, clients):
        ids = [c.client_id for c in clients]
        for cid in ids:
            self.P.setdefault(cid, {})
            for j in ids:
                self.P[cid].setdefault(j, 1.0 if j == cid else 0.0)

    def _select_models_for(self, client_id: str):
        prefs = self.P.get(client_id, {})
        candidates = [(cid, float(prefs.get(cid, 0.0))) for cid in self.client_states]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in candidates[:max(self.M, 1)]]

    def _init_state_for(self, client_id: str):
        ids = self._select_models_for(client_id)
        if not ids:
            return self.client_states.get(client_id, None)
        weights = [max(float(self.P.get(client_id, {}).get(cid, 0.0)), 0.0) for cid in ids]
        if sum(weights) <= 1e-12:
            weights = [1.0 for _ in ids]
        return weighted_state_average([self.client_states[cid] for cid in ids], weights)

    @staticmethod
    def _state_distance(a, b):
        if not a or not b:
            return 1.0
        s = 0.0
        for k, v in a.items():
            if k in b:
                d = v.float() - b[k].float()
                s += float(torch.sum(d * d))
        return max(s ** 0.5, 1e-12)

    def train_client(self, client, server_model, aggregate_keys, clients=None):
        if clients is not None:
            self._ensure_clients(clients)
        init_state = self._init_state_for(client.client_id)
        local_state, info = client.train_one_round(
            server_model, aggregate_keys, init_state=init_state,
            feature_align_lambda=0.0, fedprox_mu=0.0, save_personal=False,
        )
        self.client_states[client.client_id] = local_state
        return local_state, info

    def aggregate(self, server, local_states, infos, clients=None):
        if clients is None:
            return
        self._ensure_clients(clients)
        uploaded = [(info['metrics']['client_id'], state) for state, info in zip(local_states, infos)]
        for client in clients:
            cur = self.client_states.get(client.client_id, None)
            if cur is None:
                continue
            base_loss = client.evaluate_state(server.model, cur)['loss']
            for src_id, src_state in uploaded:
                loss = client.evaluate_state(server.model, src_state)['loss']
                gain = base_loss - loss
                dist = self._state_distance(cur, src_state)
                # update preference, negative gains are ignored
                self.P.setdefault(client.client_id, {})[src_id] = max(float(gain) / dist, 0.0)
        for cid in list(self.P.keys()):
            vals = normalize_weights([max(v, 0.0) for v in self.P[cid].values()])
            for k, val in zip(list(self.P[cid].keys()), vals):
                self.P[cid][k] = val

    def evaluate(self, server, clients):
        rows = []
        for client in clients:
            state = self._init_state_for(client.client_id)
            res = client.evaluate_state(server.model, state)
            rows.append({'client_id': client.client_id, 'dice': res['dice']})
        return sum(r['dice'] for r in rows) / max(len(rows), 1), rows
