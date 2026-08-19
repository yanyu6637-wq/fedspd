
from __future__ import annotations
from typing import Dict
from federated.param_filter import BASELINE_UPLOAD_KEYWORDS


class FedALAStrategy:
    # local mix of global and previous local weights
    name = 'fedala'
    aggregate_keywords = BASELINE_UPLOAD_KEYWORDS
    use_fisher = False

    def __init__(self, config=None):
        self.config = config or {}
        self.client_states: Dict[str, dict] = {}
        self.ala_steps = int(self.config.get('fedala_steps', 5))
        self.ala_lr = float(self.config.get('fedala_lr', 0.05))
        self.ala_batches = int(self.config.get('fedala_batches', 2))
        self.ala_layer_idx = int(self.config.get('fedala_layer_idx', 0))
        self.ala_eta = float(self.config.get('fedala_eta', 1.0))

    def train_client(self, client, server_model, aggregate_keys):
        local_state = self.client_states.get(client.client_id, None)
        init_state = client.adapt_ala_state(
            server_model, aggregate_keys, local_state=local_state,
            steps=self.ala_steps, lr=self.ala_lr, batch_limit=self.ala_batches,
            layer_idx=self.ala_layer_idx, eta=self.ala_eta,
        )
        local_state, info = client.train_one_round(
            server_model, aggregate_keys, init_state=init_state,
            feature_align_lambda=0.0, fedprox_mu=0.0, save_personal=True,
        )
        self.client_states[client.client_id] = local_state
        return local_state, info

    def aggregate(self, server, local_states, infos, clients=None):
        server.aggregate(local_states, infos)

    def evaluate(self, server, clients):
        rows = []
        for client in clients:
            state = self.client_states.get(client.client_id, None)
            res = client.evaluate_state(server.model, state)
            rows.append({'client_id': client.client_id, 'dice': res['dice']})
        return sum(r['dice'] for r in rows) / max(len(rows), 1), rows
