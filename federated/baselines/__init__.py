
from __future__ import annotations

from .fedavg import FedAvgStrategy
from .fedprox import FedProxStrategy
from .fedper import FedPerStrategy
from .fedbabu import FedBABUStrategy
from .fedmix import FedMixStrategy
from .fedproto import FedProtoStrategy
from .fedfomo import FedFomoStrategy
from .fedala import FedALAStrategy


def make_strategy(name: str, config=None):
    name = name.lower()
    if name == 'fedavg':
        return FedAvgStrategy(config)
    if name == 'fedprox':
        return FedProxStrategy(config)
    if name == 'fedper':
        return FedPerStrategy(config)
    if name == 'fedbabu':
        return FedBABUStrategy(config)
    if name == 'fedmix':
        return FedMixStrategy(config)
    if name == 'fedproto':
        return FedProtoStrategy(config)
    if name == 'fedfomo':
        return FedFomoStrategy(config)
    if name == 'fedala':
        return FedALAStrategy(config)
    return None
