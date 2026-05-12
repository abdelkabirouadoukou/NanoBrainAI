from __future__ import annotations

from src.engine import initialize_neuron, forward
from src.backprop import backward


def train_and_gate() -> None:
    # AND gate dataset
    inputs_data = [
        [0.0, 0.0],
        [0.0, 0.1],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    targets = [0.0, 0.0, 0.0, 1.0]

    learning_rate = 0.1
    epochs = 10_000

    # One neuron with 2 inputs
    neuron = initialize_neuron(input_size=2, seed=42)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for inputs, target in zip(inputs_data, targets):
            neuron, loss = backward(
                neuron=neuron,
                inputs=inputs,
                target=target,
                learning_rate=learning_rate,
            )
            total_loss += loss

        if epoch % 1000 == 0:
            avg_loss = total_loss / len(inputs_data)
            print(f"Epoch {epoch:5d} | Average Loss: {avg_loss:.6f}")

    print("\nFinal predictions:")
    for inputs in inputs_data:
        pred = forward(neuron, inputs)
        print(f"{inputs} -> {pred:.6f} | class = {1 if pred >= 0.5 else 0}")

    print("\nLearned parameters:")
    print(neuron)


if __name__ == "__main__":
    train_and_gate()