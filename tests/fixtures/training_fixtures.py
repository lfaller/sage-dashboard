"""Training fixture loading utilities.

Provides functions to load training fixtures for reproducible model training tests.
Fixtures include balanced, imbalanced, and small-scale datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


FIXTURES_DIR = Path(__file__).parent / "training_data"

# Fixture metadata
FIXTURE_SPECS = {
    "tiny": {
        "filename": "sage_training_tiny.json",
        "n_samples": 10,
        "n_features": 20,
        "n_female": 5,
        "n_male": 5,
        "balanced": True,
        "description": "Small balanced fixture for quick tests",
    },
    "v1": {
        "filename": "sage_training_v1.json",
        "n_samples": 100,
        "n_features": 100,
        "n_female": 50,
        "n_male": 50,
        "balanced": True,
        "description": "Standard balanced fixture for training",
    },
    "imbalanced": {
        "filename": "sage_training_imbalanced.json",
        "n_samples": 60,
        "n_features": 100,
        "n_female": 15,
        "n_male": 45,
        "balanced": False,
        "description": "Imbalanced fixture to test handling of class imbalance",
    },
}


def load_training_fixture(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training fixture data.

    Args:
        name: Fixture name ("tiny", "v1", or "imbalanced")

    Returns:
        Tuple of (X, y) where:
        - X: Feature matrix (n_samples, n_features)
        - y: Labels (n_samples,) with 0=female, 1=male

    Raises:
        FileNotFoundError: If fixture file not found
        KeyError: If fixture name not recognized
    """
    if name not in FIXTURE_SPECS:
        raise FileNotFoundError(
            f"Unknown fixture '{name}'. Available: {list(FIXTURE_SPECS.keys())}"
        )

    spec = FIXTURE_SPECS[name]
    fixture_path = FIXTURES_DIR / spec["filename"]

    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    # Load JSON fixture
    with open(fixture_path) as f:
        data = json.load(f)

    # Extract features and labels
    features = [sample["features"] for sample in data]
    labels = [sample["sex"] for sample in data]

    X = np.array(features, dtype=np.float64)
    y = np.array(labels, dtype=np.int64)

    return X, y


def get_fixture_metadata(name: str) -> Dict[str, any]:
    """Get metadata for a training fixture.

    Args:
        name: Fixture name ("tiny", "v1", or "imbalanced")

    Returns:
        Dictionary with fixture metadata

    Raises:
        FileNotFoundError: If fixture not found
    """
    if name not in FIXTURE_SPECS:
        raise FileNotFoundError(
            f"Unknown fixture '{name}'. Available: {list(FIXTURE_SPECS.keys())}"
        )

    spec = FIXTURE_SPECS[name]

    return {
        "name": name,
        "n_samples": spec["n_samples"],
        "n_features": spec["n_features"],
        "n_female": spec["n_female"],
        "n_male": spec["n_male"],
        "balanced": spec["balanced"],
        "description": spec["description"],
    }


def list_available_fixtures() -> List[str]:
    """List all available fixture names.

    Returns:
        List of fixture names
    """
    return list(FIXTURE_SPECS.keys())
