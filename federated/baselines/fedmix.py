
from __future__ import annotations
from federated.fed_utils import normalize_weights, state_dict_cpu, weighted_delta_update, load_partial_state
from federated.param_filter import BASELINE_UPLOAD_KEYWORDS


class FedMixStrategy:
    # mixed labels on client side
    name = 'fedmix'
    aggregate_keywords = BASELINE_UPLOAD_KEYWORDS
    use_fisher = False

    def __init__(self, config=None):
        self.config = config or {}
        self.mixup_alpha = float(self.config.get('fedmix_alpha', 0.2))
        self.weak_weight = float(self.config.get('fedmix_weak_weight', 0.1))
        self.client_supervision = self.config.get('client_supervision', {}) or {}

    def train_client(self, client, server_model, aggregate_keys):
        mode = self.client_supervision.get(client.client_id, getattr(client, 'supervision', 'pixel'))
        return client.train_fedmix_round(
            server_model, aggregate_keys, supervision=mode,
            mixup_alpha=self.mixup_alpha, weak_weight=self.weak_weight,
            save_personal=False,
        )

    def aggregate(self, server, local_states, infos, clients=None):
        if not local_states:
            return
        scores = [max(float(info.get('fedmix_score', 0.0)), 1e-12) for info in infos]
        weights = normalize_weights(scores)
        base_state = state_dict_cpu(server.model, keys=server.aggregate_keys)
        new_state = weighted_delta_update(base_state, local_states, weights)
        load_partial_state(server.model, new_state, strict=False)

    def evaluate(self, server, clients):
        rows = [c.evaluate(server.model) for c in clients]
        return sum(r['dice'] for r in rows) / max(len(rows), 1), rows
