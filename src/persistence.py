from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def save_model(neuron: Dict[str, float], filename: str = "models/neuron.json") -> None:
    """
    Save the neuron parameters to a JSON file.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(neuron, f, indent=4)


def load_model(filename: str = "models/neuron.json") -> Dict[str, float]:
    """
    Load the neuron parameters from a JSON file.
    """
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {filename}")

    with path.open("r", encoding="utf-8") as f:
        neuron = json.load(f)

    # Optional safety check: ensure values are numeric
    return {k: float(v) for k, v in neuron.items()}