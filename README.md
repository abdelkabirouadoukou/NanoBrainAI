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

\[
z = \sum_{i=1}^{n} w_i x_i + b
\]

Then applies a nonlinear activation:

\[
a = \sigma(z) = \frac{1}{1 + e^{-z}}
\]

The neuron learns by comparing its prediction \(a\) to the target \(y\), measuring the error, and updating its parameters to reduce that error.

---

## Mathematical Derivations

### 1. Sigmoid Function

The activation function is:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Its limits are:

\[
\lim_{z \to +\infty} \sigma(z) = 1
\qquad\text{and}\qquad
\lim_{z \to -\infty} \sigma(z) = 0
\]

So the output stays between 0 and 1, which makes it useful for a neuron that behaves like a smooth switch.

### 2. Derivative of Sigmoid

Starting from:

\[
\sigma(z) = (1 + e^{-z})^{-1}
\]

we differentiate:

\[
\sigma'(z)
= - (1 + e^{-z})^{-2}\cdot(-e^{-z})
\]

which gives:

\[
\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2}
\]

A useful equivalent form is:

\[
\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)
\]

This identity is especially important in backpropagation.

### 3. Mean Squared Error

For a single prediction:

\[
L = (y - \hat{y})^2
\]

where:

- \(y\) is the target
- \(\hat{y}\) is the prediction

The derivative with respect to the prediction is:

\[
\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)
\]

This tells the model how to correct itself.

### 4. Backpropagation by the Chain Rule

For one neuron:

\[
z = \sum_i w_i x_i + b,\qquad
a = \sigma(z),\qquad
L = (a - y)^2
\]

Using the chain rule:

\[
\frac{\partial L}{\partial w_i}
=
\frac{\partial L}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial w_i}
\]

with:

\[
\frac{\partial L}{\partial a} = 2(a-y)
\]

\[
\frac{\partial a}{\partial z} = a(1-a)
\]

\[
\frac{\partial z}{\partial w_i} = x_i
\]

Therefore:

\[
\frac{\partial L}{\partial w_i}
=
2(a-y)\,a(1-a)\,x_i
\]

For the bias:

\[
\frac{\partial L}{\partial b}
=
2(a-y)\,a(1-a)
\]

### 5. Gradient Descent Update

Each parameter is updated by:

\[
w_i \leftarrow w_i - \eta \frac{\partial L}{\partial w_i}
\]

\[
b \leftarrow b - \eta \frac{\partial L}{\partial b}
\]

where \(\eta\) is the learning rate.

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