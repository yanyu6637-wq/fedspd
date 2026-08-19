# main training entry
from __future__ import annotations

import argparse
import copy
import os
import sys
import random
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

# for direct launch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets
import models
import utils
from federated.fed_client import FedClient
from federated.fed_server import FedServer
from federated.fed_utils import ensure_dir, save_csv, set_seed
from federated.param_filter import (
    BASELINE_UPLOAD_KEYWORDS,
    FEDSPD_UPLOAD_KEYWORDS,
    freeze_for_fedspd,
    get_aggregated_param_names,
    count_total_params,
    count_trainable_params,
    communication_cost_mb,
)
from federated.mamlpp_init import mamlpp_warm_start
from federated.baselines import make_strategy


def make_loader(spec: dict, shuffle: bool = False) -> DataLoader:
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    return DataLoader(
        dataset,
        batch_size=int(spec.get('batch_size', 1)),
        shuffle=shuffle,
        num_workers=int(spec.get('num_workers', 4)),
        pin_memory=True,
        drop_last=False,
    )


def build_client_specs(config: dict):
    # supports explicit client paths or data_path/client_name
    if 'clients' in config and config['clients']:
        clients = config['clients']
    else:
        data_path = config['data_path']
        names = config.get('client_names', ['client1', 'client2', 'client3', 'client4', 'client5'])
        clients = []
        for name in names:
            base = os.path.join(data_path, name)
            clients.append({
                'name': name,
                'support_img': os.path.join(base, 'support_train', 'img'),
                'support_mask': os.path.join(base, 'support_train', 'mask'),
                'query_img': os.path.join(base, 'query_train', 'img'),
                'query_mask': os.path.join(base, 'query_train', 'mask'),
                'supervision': 'pixel',
            })

    support_template = copy.deepcopy(config['train_dataset'])
    query_template = copy.deepcopy(config['val_dataset'])
    out = []
    for c in clients:
        support = copy.deepcopy(support_template)
        query = copy.deepcopy(query_template)
        support['dataset']['args']['root_path_1'] = c['support_img']
        support['dataset']['args']['root_path_2'] = c['support_mask']
        query['dataset']['args']['root_path_1'] = c['query_img']
        query['dataset']['args']['root_path_2'] = c['query_mask']
        out.append((c['name'], support, query, c.get('supervision', 'pixel')))
    return out


def build_model(config: dict, num_points: int | None = None, num_task_tokens: int | None = None):
    model_spec = copy.deepcopy(config['model'])
    enc = model_spec['args']['encoder_mode']
    if num_points is not None:
        enc['num_points'] = int(num_points)
    if num_task_tokens is not None:
        enc['num_task_tokens'] = int(num_task_tokens)
    model = models.make(model_spec)
    freeze_for_fedspd(model)
    return model


@torch.no_grad()
def evaluate_global(model, clients, device):
    rows = []
    for client in clients:
        rows.append(client.evaluate(model))
    avg = sum(r['dice'] for r in rows) / max(len(rows), 1)
    return avg, rows


def run(config: dict, args):
    set_seed(int(config.get('seed', 2026)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir or config.get('out_dir', './work_dirs/fedspd'))
    ensure_dir(out_dir)

    model = build_model(config, args.num_points, args.num_task_tokens).to(device)
    if config.get('sam_checkpoint') and os.path.exists(config['sam_checkpoint']):
        checkpoint = torch.load(config['sam_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        print(f"[FedSPD] loaded SAM checkpoint: {config['sam_checkpoint']}")
    strategy = make_strategy(args.algorithm, config)
    if strategy is not None:
        aggregate_keywords = strategy.aggregate_keywords
        use_fisher = bool(getattr(strategy, 'use_fisher', False)) and (not args.no_fisher)
    elif args.algorithm == 'fedspd':
        aggregate_keywords = FEDSPD_UPLOAD_KEYWORDS
        use_fisher = not args.no_fisher
    else:
        aggregate_keywords = BASELINE_UPLOAD_KEYWORDS
        use_fisher = False
    aggregate_keys = get_aggregated_param_names(model, keywords=aggregate_keywords)
    server = FedServer(model, aggregate_keys, algorithm=args.algorithm, use_fisher=use_fisher)

    lr = float(config.get('optimizer', {}).get('args', {}).get('lr', 2e-4))
    local_epochs = int(config.get('local_epochs', config.get('train_dataset', {}).get('inner_epochs', 1)))
    feature_align_lambda = float(config.get('feature_align_lambda', 0.05))
    fedprox_mu = float(config.get('fedprox_mu', 0.01)) if args.algorithm == 'fedprox' else 0.0
    rounds = int(args.rounds or config.get('communication_rounds', 25))
    max_steps = config.get('max_steps_per_client', None)
    if max_steps is not None:
        max_steps = int(max_steps)

    client_objs = []
    for name, support_spec, query_spec, supervision in build_client_specs(config):
        support_loader = make_loader(support_spec, shuffle=True)
        query_loader = make_loader(query_spec, shuffle=True)
        client_objs.append(FedClient(
            name, support_loader, query_loader, device=device, lr=lr,
            local_epochs=local_epochs, feature_align_lambda=feature_align_lambda,
            fedprox_mu=fedprox_mu, num_points=int(args.num_points or config['model']['args']['encoder_mode'].get('num_points', 40)),
            max_steps=max_steps, supervision=supervision,
        ))

    # optional meta initialization before communication rounds
    maml_steps = int(args.mamlpp_steps if args.mamlpp_steps is not None else config.get('mamlpp_steps', 0))
    if maml_steps > 0:
        loader_pairs = [(c.support_loader, c.query_loader) for c in client_objs]
        server.model = mamlpp_warm_start(
            server.model,
            loader_pairs,
            aggregate_keys,
            device=device,
            steps=maml_steps,
            inner_lr=float(config.get('mamlpp_inner_lr', 1e-3)),
            outer_lr=float(config.get('mamlpp_outer_lr', 1e-1)),
            num_points=int(args.num_points or config['model']['args']['encoder_mode'].get('num_points', 40)),
        )

    summary = {
        'total_params': count_total_params(model),
        'trainable_params': count_trainable_params(model),
        'communication_mb': communication_cost_mb(model),
        'aggregate_keys': len(aggregate_keys),
    }
    print('[FedSPD] parameter summary:', summary)

    history = []
    for rnd in range(1, rounds + 1):
        local_states, infos = [], []
        print(f'\n===== Communication round {rnd}/{rounds} | {args.algorithm} =====')
        drop_n = int(args.drop_clients if args.drop_clients is not None else config.get('drop_clients_per_round', 0))
        participating = list(client_objs)
        if drop_n > 0 and drop_n < len(participating):
            dropped = set(random.sample(range(len(participating)), drop_n))
            participating = [c for idx, c in enumerate(participating) if idx not in dropped]
        for client in participating:
            if args.algorithm == 'local' and rnd > 1:
                continue
            if strategy is not None:
                if args.algorithm == 'fedfomo':
                    local_state, info = strategy.train_client(client, server.model, aggregate_keys, clients=client_objs)
                else:
                    local_state, info = strategy.train_client(client, server.model, aggregate_keys)
            else:
                local_state, info = client.train_one_round(
                    server.model,
                    aggregate_keys,
                    feature_align_lambda=(feature_align_lambda if args.algorithm == 'fedspd' else 0.0),
                    fedprox_mu=(fedprox_mu if args.algorithm == 'fedprox' else 0.0),
                )
            local_states.append(local_state)
            infos.append(info)
            row = {'round': rnd, 'algorithm': args.algorithm, **info['metrics'], 'fisher_score': info.get('fisher_score', 0.0)}
            if 'ala_alpha' in info:
                row['ala_alpha'] = info['ala_alpha']
            print(row)
            history.append(row)
        if args.algorithm != 'local':
            if strategy is not None:
                strategy.aggregate(server, local_states, infos, clients=client_objs)
            else:
                server.aggregate(local_states, infos)
        if strategy is not None:
            avg_dice, client_eval = strategy.evaluate(server, client_objs)
        else:
            avg_dice, client_eval = evaluate_global(server.model, client_objs, device)
        print(f'[round {rnd}] global average Dice={avg_dice:.4f}')
        history.append({'round': rnd, 'algorithm': args.algorithm, 'client_id': 'GLOBAL_AVG', 'loss': 0.0, 'dice': avg_dice, 'num_steps': 0})
        torch.save(server.model.state_dict(), out_dir / f'{args.algorithm}_round_{rnd}.pth')

    if strategy is not None and hasattr(strategy, 'post_training'):
        strategy.post_training(server, client_objs)
        avg_dice, _ = strategy.evaluate(server, client_objs)
        print(f'[post-training] average Dice={avg_dice:.4f}')
        history.append({'round': rounds, 'algorithm': args.algorithm, 'client_id': 'POST_TRAIN_AVG', 'loss': 0.0, 'dice': avg_dice, 'num_steps': 0})

    save_csv(out_dir / f'{args.algorithm}_history.csv', history)
    save_csv(out_dir / f'{args.algorithm}_params.csv', [summary])
    return history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/fedspd.yaml')
    parser.add_argument('--algorithm', default='fedspd', choices=['fedspd', 'fedavg', 'fedprox', 'local', 'fedper', 'fedmix', 'fedbabu', 'fedfomo', 'fedproto', 'fedala'])
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--num_points', type=int, default=None, help='N_p for sensitivity analysis')
    parser.add_argument('--num_task_tokens', type=int, default=None, help='N_t for sensitivity analysis')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--drop_clients', type=int, default=None)
    parser.add_argument('--no_fisher', action='store_true')
    parser.add_argument('--mamlpp_steps', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    run(config, args)
