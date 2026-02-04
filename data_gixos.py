import os
from pathlib import Path
from typing import TypedDict, Optional

import yaml


class Sample(TypedDict):
    name: str
    full_name: str
    index: list[int]
    pressure: list[float]


# Select experiment (can be overridden via EXPERIMENT environment variable)
# Accepts "1", "2", "experiment_1", "experiment_2", etc.
_EXPERIMENT_ENV = os.getenv("EXPERIMENT", "1")
_EXPERIMENT_NUM = _EXPERIMENT_ENV.replace("experiment_", "") if "experiment" in _EXPERIMENT_ENV else _EXPERIMENT_ENV

# Load data from experiment-specific YAML file
_DATA_FILE = Path(__file__).parent / "data" / _EXPERIMENT_NUM / "gixos.yaml"
with _DATA_FILE.open() as f:
    _DATA = yaml.safe_load(f)

# GIXOS fitting does not use a separate water reference, but kept for parity
REFERENCE: Optional[Sample] = _DATA.get("reference")

SAMPLES: list[Sample] = _DATA["sample"]
SAMPLES_TEST: list[Sample] = _DATA["test_sample"]


def get_samples(test: bool = False) -> list[Sample]:
    """
    Returns a list of samples to process.

    Args:
        test: If True, returns a single sample for testing.

    Returns:
        A list of samples.
    """
    return SAMPLES_TEST if test else SAMPLES
