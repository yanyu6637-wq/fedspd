
from __future__ import annotations
from federated.param_filter import FEDSPD_UPLOAD_KEYWORDS, LOCAL_PERSONAL_KEYWORDS


class FedBABUStrategy:
    # train body first, tune head later
    name = 'fedbabu'
    aggregate_keywords = FEDSPD_UPLOAD_KEYWORDS
    use_fisher = False

    def __init__(self, config=None):
        self.config = config or {}
        self.head_epochs = int(self.config.get('fedbabu_head_epochs', 1))

    def train_client(self, client, server_model, aggregate_keys):
        return client.train_one_round(
            server_model, aggregate_keys, feature_align_lambda=0.0, fedprox_mu=0.0,
            save_personal=False, freeze_keywords=LOCAL_PERSONAL_KEYWORDS,
        )

    def aggregate(self, server, local_states, infos, clients=None):
        server.aggregate(local_states, infos)

    def post_training(self, server, clients):
        for client in clients:
            client.fine_tune_personal_head(server.model, epochs=self.head_epochs)

    def evaluate(self, server, clients):
        rows = [c.evaluate(server.model) for c in clients]
        return sum(r['dice'] for r in rows) / max(len(rows), 1), rows
