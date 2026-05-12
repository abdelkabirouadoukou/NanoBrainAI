import math

def sigmoid(z):
    """
    Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
    Maps R to (0, 1).
    """
    # Prevent overflow for very negative z
    if z < -709:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_derivative(z):
    """
    Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))
    Used for Backpropagation.
    """
    s = sigmoid(z)
    return s * (1.0 - s)