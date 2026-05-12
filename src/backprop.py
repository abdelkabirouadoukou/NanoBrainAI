from __future__ import annotations

from typing import Dict, List, Tuple

from .activation import sigmoid
from .loss import mse


def sigmoid_derivative_from_output(a: float) -> float:
    """
    If a = sigmoid(z), then sigmoid'(z) = a * (1 - a)
    """
    return a * (1.0 - a)


def forward(neuron: Dict[str, float], inputs: List[float]) -> float:
    z = sum(neuron[f"w{i}"] * inputs[i] for i in range(len(inputs))) + neuron["b"]
    return sigmoid(z)


def backward(
    neuron: Dict[str, float],
    inputs: List[float],
    target: float,
    learning_rate: float,
) -> Tuple[Dict[str, float], float]:
    """
    Compute gradients and update weights in place.

    Returns:
        (updated_neuron, loss)
    """
    prediction = forward(neuron, inputs)
    loss = mse(prediction, target)

    # dL/da
    dL_da = 2.0 * (prediction - target)

    # da/dz
    da_dz = sigmoid_derivative_from_output(prediction)

    # dL/dz
    dL_dz = dL_da * da_dz

    # Update each weight: w = w - LR * gradient
    for i, x_i in enumerate(inputs):
        grad_wi = dL_dz * x_i
        neuron[f"w{i}"] -= learning_rate * grad_wi

    # Update bias
    grad_b = dL_dz
    neuron["b"] -= learning_rate * grad_b

    return neuron, loss