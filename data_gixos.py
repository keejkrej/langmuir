import os
from pathlib import Path
from typing import TypedDict, Optional

import yaml


class Sample(TypedDict):
    name: str
    full_name: str
    index: list[int]
    pressure: list[float]


class FitConfig(TypedDict):
    water_sld: float
    tails_sld_init: float
    heads_sld_init: float
    tails_thick_init: float
    heads_thick_init: float
    rough_init: float
    heads_vfsolv_init: float
    tails_thick_bounds: tuple[float, float]
    tails_sld_bounds: tuple[float, float]
    heads_thick_bounds: tuple[float, float]
    heads_sld_bounds: tuple[float, float]
    heads_vfsolv_bounds: tuple[float, float]
    rough_bounds: tuple[float, float]
    r_q_bounds: tuple[float, float]
    rfxsf_q_bounds: tuple[float, float]
    scale_q_bounds: tuple[float, float]
    r_outlier_neighbor_factor: float
    r_outlier_min_intensity: float
    r_de_maxiter: int
    rfxsf_de_maxiter: int


# Select experiment (can be overridden via EXPERIMENT environment variable)
# Accepts "1", "2", "experiment_1", "experiment_2", etc.
_EXPERIMENT_ENV = os.getenv("EXPERIMENT", "1")
_EXPERIMENT_NUM = _EXPERIMENT_ENV.replace("experiment_", "") if "experiment" in _EXPERIMENT_ENV else _EXPERIMENT_ENV

# Load data from experiment-specific YAML file (./data in project)
_DATA_FILE = Path(__file__).resolve().parent / "data" / _EXPERIMENT_NUM / "gixos.yaml"
with _DATA_FILE.open() as f:
    _DATA = yaml.safe_load(f)


def _normalize_bounds(value: object, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return default

# GIXOS fitting does not use a separate water reference, but kept for parity
REFERENCE: Optional[Sample] = _DATA.get("reference")

SAMPLES: list[Sample] = _DATA["sample"]
SAMPLES_TEST: list[Sample] = _DATA["test_sample"]

_FIT_DATA = _DATA.get("fit", {})
FIT_CONFIG: FitConfig = {
    "water_sld": float(_FIT_DATA.get("water_sld", 9.43)),
    "tails_sld_init": float(_FIT_DATA.get("tails_sld_init", 8.6)),
    "heads_sld_init": float(_FIT_DATA.get("heads_sld_init", 14.5)),
    "tails_thick_init": float(_FIT_DATA.get("tails_thick_init", 18.0)),
    "heads_thick_init": float(_FIT_DATA.get("heads_thick_init", 8.0)),
    "rough_init": float(_FIT_DATA.get("rough_init", 2.5)),
    "heads_vfsolv_init": float(_FIT_DATA.get("heads_vfsolv_init", 0.2)),
    "tails_thick_bounds": _normalize_bounds(
        _FIT_DATA.get("tails_thick_bounds"), (10.0, 30.0)
    ),
    "tails_sld_bounds": _normalize_bounds(_FIT_DATA.get("tails_sld_bounds"), (8.0, 9.0)),
    "heads_thick_bounds": _normalize_bounds(
        _FIT_DATA.get("heads_thick_bounds"), (5.0, 20.0)
    ),
    "heads_sld_bounds": _normalize_bounds(
        _FIT_DATA.get("heads_sld_bounds"), (13.5, 15.5)
    ),
    "heads_vfsolv_bounds": _normalize_bounds(
        _FIT_DATA.get("heads_vfsolv_bounds"), (0.0, 0.8)
    ),
    "rough_bounds": _normalize_bounds(_FIT_DATA.get("rough_bounds"), (0.5, 6.0)),
    "r_q_bounds": _normalize_bounds(_FIT_DATA.get("r_q_bounds"), (0.025, 0.6)),
    "rfxsf_q_bounds": _normalize_bounds(_FIT_DATA.get("rfxsf_q_bounds"), (0.02, 0.8)),
    "scale_q_bounds": _normalize_bounds(_FIT_DATA.get("scale_q_bounds"), (0.08, 0.3)),
    "r_outlier_neighbor_factor": float(
        _FIT_DATA.get("r_outlier_neighbor_factor", 100.0)
    ),
    "r_outlier_min_intensity": float(_FIT_DATA.get("r_outlier_min_intensity", 1000.0)),
    "r_de_maxiter": int(_FIT_DATA.get("r_de_maxiter", 200)),
    "rfxsf_de_maxiter": int(_FIT_DATA.get("rfxsf_de_maxiter", 150)),
}


def get_samples(test: bool = False) -> list[Sample]:
    """
    Returns a list of samples to process.

    Args:
        test: If True, returns a single sample for testing.

    Returns:
        A list of samples.
    """
    return SAMPLES_TEST if test else SAMPLES


def get_fit_config() -> FitConfig:
    """Return GIXOS fitting configuration for the selected experiment."""
    return FIT_CONFIG
