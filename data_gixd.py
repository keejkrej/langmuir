import os
from pathlib import Path
from typing import TypedDict

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

# Load data from experiment-specific YAML file (./data in project)
_DATA_FILE = Path(__file__).resolve().parent / "data" / _EXPERIMENT_NUM / "gixd.yaml"
with _DATA_FILE.open() as f:
    _DATA = yaml.safe_load(f)

# Background is always a list - use the first background
WATER: Sample = _DATA["background"][0]

SAMPLES: list[Sample] = _DATA["sample"]
SAMPLES_TEST: list[Sample] = _DATA["test_sample"]

def get_samples(test: bool = False) -> list[Sample]:
    """
    Returns a list of samples to process.

    Args:
        test: If True, returns the configured test samples.

    Returns:
        A list of samples.
    """
    return SAMPLES_TEST if test else SAMPLES


def get_water() -> Sample:
    """
    Returns the reference sample (water).

    Returns:
        The water sample.
    """
    return WATER


ROI_IQ: list[float] = _DATA["roi_iq"]  # [q_min, q_max, tau_min, tau_max]
ROI_ITAU: list[float] = _DATA["roi_itau"]  # [q_min, q_max, tau_min, tau_max]
QZ_MAX: float = _DATA.get("qz_max", 1.0)  # Max qz to include when loading (filter out high qz)
