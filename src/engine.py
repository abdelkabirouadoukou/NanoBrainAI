from __future__ import annotations

import random
from typing import Dict, List

from .activation import sigmoid


def initialize_neuron(input_size: int, seed: int | None = None) -> Dict[str, float]:
    if input_size <= 0:
        raise ValueError("input_size must be a positive integer")

    if seed is not None:
        random.seed(seed)

    neuron = {f"w{i}": random.uniform(-1.0, 1.0) for i in range(input_size)}
    neuron["b"] = random.uniform(-1.0, 1.0)
    return neuron


def forward(neuron: Dict[str, float], inputs: List[float]) -> float:
    weights = [neuron[f"w{i}"] for i in range(len(inputs))]

    if len(weights) != len(inputs):
        raise ValueError("inputs length does not match neuron weights")

    z = sum(w * x for w, x in zip(weights, inputs)) + neuron["b"]
    return sigmoid(z)