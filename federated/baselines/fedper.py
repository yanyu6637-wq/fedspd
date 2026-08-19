
from __future__ import annotations
from federated.param_filter import FEDSPD_UPLOAD_KEYWORDS


class FedPerStrategy:
    # base shared, head stays local
    name = 'fedper'
    aggregate_keywords = FEDSPD_UPLOAD_KEYWORDS
    use_fisher = False

    def __init__(self, config=None):
        self.config = config or {}

    def train_client(self, client, server_model, aggregate_keys):
        return client.train_one_round(server_model, aggregate_keys, feature_align_lambda=0.0, fedprox_mu=0.0, save_personal=True)

    def aggregate(self, server, local_states, infos, clients=None):
        server.aggregate(local_states, infos)

    def evaluate(self, server, clients):
        rows = [c.evaluate(server.model) for c in clients]
        return sum(r['dice'] for r in rows) / max(len(rows), 1), rows
