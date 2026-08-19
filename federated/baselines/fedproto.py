
from __future__ import annotations
from typing import Dict, List
import torch
from federated.param_filter import BASELINE_UPLOAD_KEYWORDS


class FedProtoStrategy:
    # global foreground/background prototypes
    name = 'fedproto'
    aggregate_keywords = BASELINE_UPLOAD_KEYWORDS
    use_fisher = False

    def __init__(self, config=None):
        self.config = config or {}
        self.global_protos: Dict[str, torch.Tensor] = {}
        self.client_states: Dict[str, Dict[str, torch.Tensor]] = {}
        self.proto_lambda = float(self.config.get('fedproto_lambda', 0.05))

    def train_client(self, client, server_model, aggregate_keys):
        init_state = self.client_states.get(client.client_id, None)
        local_state, info = client.train_one_round(
            server_model, aggregate_keys, init_state=init_state,
            feature_align_lambda=0.0, fedprox_mu=0.0,
            global_protos=self.global_protos, proto_lambda=self.proto_lambda if self.global_protos else 0.0,
            save_personal=True,
        )
        self.client_states[client.client_id] = local_state
        protos, counts = client.collect_prototypes(server_model, init_state=local_state)
        info['protos'] = protos
        info['proto_counts'] = counts
        return local_state, info

    def aggregate(self, server, local_states, infos, clients=None):
        sums, counts = {}, {}
        for info in infos:
            for key, proto in info.get('protos', {}).items():
                c = float(info.get('proto_counts', {}).get(key, 1.0))
                sums[key] = proto.detach().cpu() * c if key not in sums else sums[key] + proto.detach().cpu() * c
                counts[key] = counts.get(key, 0.0) + c
        self.global_protos = {k: sums[k] / max(counts[k], 1.0) for k in sums}

    def evaluate(self, server, clients):
        rows = []
        for client in clients:
            res = client.evaluate_state(server.model, self.client_states.get(client.client_id, None))
            rows.append({'client_id': client.client_id, 'dice': res['dice']})
        return sum(r['dice'] for r in rows) / max(len(rows), 1), rows
