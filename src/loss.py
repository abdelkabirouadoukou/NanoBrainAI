from __future__ import annotations


def mse(prediction: float, target: float) -> float:
    """
    Mean Squared Error for one sample.

    L = (target - prediction)^2
    """
    return (target - prediction) ** 2


def mse_derivative(prediction: float, target: float) -> float:
    """
    Derivative of the loss with respect to the prediction.

    L = (target - prediction)^2
    dL/d(prediction) = 2 * (prediction - target)
    """
    return 2.0 * (prediction - target)

def mse_with_derivative(prediction: float, target: float) -> tuple[float, float]:
    loss = (target - prediction) ** 2
    derivative = 2.0 * (prediction - target)
    return loss, derivative