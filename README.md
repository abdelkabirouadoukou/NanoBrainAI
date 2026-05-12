# NanoBrainAI

**NanoBrainAI** is a from-scratch neural network project developed in Python to explore the mathematical foundations of artificial intelligence through **Linear Algebra**, **Differential Calculus**, and **Gradient-Based Optimization**.

Created by **Abdelkabir Ouadoukou**, this repository presents a complete implementation of a basic artificial neuron without relying on machine learning frameworks. It focuses on clarity, mathematical rigor, and clean modular engineering to bridge the gap between theoretical mathematics and practical machine learning.

---

## Abstract

This project implements the core mechanics of a neuron:

- weighted sum
- sigmoid activation
- mean squared error loss
- gradient descent
- backpropagation
- JSON model persistence

The goal is not only to make the code work, but also to show the mathematics behind learning.

---

## Core Idea

A neuron computes:

z = sum_{i=1..n} w_i x_i + b

Then applies a nonlinear activation:

a = σ(z) = 1 / (1 + e^{-z})

The neuron learns by comparing its prediction (a) to the target (y), measuring the error, and updating its parameters to reduce that error.

---

## Mathematical Derivations

### 1. Sigmoid Function

The activation function is:

σ(z) = 1 / (1 + e^{-z})

Its limits are: as z → +∞, σ(z) → 1; as z → -∞, σ(z) → 0.

So the output stays between 0 and 1, which makes it useful for a neuron that behaves like a smooth switch.

### 2. Derivative of Sigmoid

Starting from σ(z) = (1 + e^{-z})^{-1} and differentiating gives:

σ'(z) = e^{-z} / (1 + e^{-z})^2

A useful equivalent form is:

σ'(z) = σ(z) * (1 - σ(z))

This identity is especially important in backpropagation.

### 3. Mean Squared Error

For a single prediction:

L = (y - y_hat)^2

where y is the target and y_hat is the prediction.

The derivative with respect to the prediction is:

dL/d(y_hat) = 2 * (y_hat - y)

This tells the model how to correct itself.

### 4. Backpropagation by the Chain Rule

For one neuron:

z = sum_i w_i x_i + b
a = σ(z)
L = (a - y)^2

Using the chain rule:

dL/dw_i = dL/da * da/dz * dz/dw_i

with dL/da = 2 * (a - y)
and da/dz = a * (1 - a)
and dz/dw_i = x_i

Therefore:

dL/dw_i = 2 * (a - y) * a * (1 - a) * x_i

For the bias:

dL/db = 2 * (a - y) * a * (1 - a)

### 5. Gradient Descent Update

Each parameter is updated by:

w_i <- w_i - η * dL/dw_i

b <- b - η * dL/db

where η is the learning rate.

This is the mechanism that makes the neuron learn.

---

## Example Use Case

The neuron is trained on the AND gate:

\[
[0,0] \mapsto 0,\quad
[0,1] \mapsto 0,\quad
[1,0] \mapsto 0,\quad
[1,1] \mapsto 1
\]

This is a classic test because it shows whether the model can learn a simple logical function using gradients.

---

## Learning Outcome

This project demonstrates that a neural network is not magic.

It is built from:

- vectors and weights
- affine transformations
- nonlinear activation
- error measurement
- derivatives
- iterative optimization

In other words, it is a mathematical machine that learns by reducing loss.

---

## Future Improvements

Possible next steps:

- multiple neurons per layer
- matrix-based implementation
- ReLU and softmax
- cross-entropy loss
- mini-batch training
- multi-layer perceptron
- visualization of loss curves

---

## Author

Built by **Abdelkabir Ouadoukou** as a scientific and educational AI project.

---

## License

MIT License. See the `LICENSE` file for details.

---

## Acknowledgment

This project is inspired by classical mathematics, neural network theory, and the idea that AI should be understandable from first principles.