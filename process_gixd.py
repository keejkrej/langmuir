"""
GIXD Data Processing Pipeline

Background subtraction:
- Uses inverse quadratic background subtraction: I = A*Qxy^-2 + B
"""

import argparse
import importlib
import os
from pathlib import Path
from typing import Optional

from utils.data.gixd import (
    load_gixd_xarray,
    gixd_cartesian2polar,
    extract_intensity_q,
    extract_intensity_tau,
)

# Inverse quadratic background subtraction
# Uses model I = A*Qxy^-2 + B fitted to edge points
from utils.background import subtract_invquad_background
import xarray as xr
import data_gixd
# Constants (PROCESSED_DIR is now set dynamically based on experiment)
QZ_CUTOFF = 0.04
QZ_BIN = 5  # channels
QXY_BIN = 5  # channels for qxy binning before background fitting
Q_BIN = 0.05
TAU_BIN = 0.02

# Inverse quadratic background subtraction settings with global fitting
# Uses model: I(qxy,qz) = (A*qz + B)*Qxy^-2 + C fitted globally across all slices
# Edges are automatically detected by scanning for jumps from zero-filled regions
INVQUAD_NUM_FITTING_POINTS = 10  # Points to use from each edge for fitting
INVQUAD_USE_GLOBAL_FIT = True

# Multiple qz slice ranges for horizontal slice analysis
QZ_SLICE_RANGES = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
    (1.0, 1.2),
]


def process_sample(
    data_dir: Path,
    processed_dir: Path,
    data,
    roi_iq,
    roi_itau,
) -> Optional[xr.DataArray]:
    """Process sample data with inverse quadratic background subtraction.

    Performs the following steps:
    * Load the raw cartesian data.
    * Subtract inverse quadratic background.
    * Slice, bin and save the cartesian data.
    * Convert to polar coordinates and save.
    * Extract ``I(q)`` and ``I(tau)`` intensity profiles and save them.
    """
    name, index, pressure = data["name"], data["index"], data["pressure"]
    out_dir = processed_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {name}...")

    # Collect all data for organized saving
    all_2d_data = {}  # Will store 2D maps
    all_1d_data = {}  # Will store 1D profiles
    all_horizontal_slices = {}  # Will store horizontal slices
    da_cart_binned = None

    for i, p in zip(index, pressure):
        da_cart = load_gixd_xarray(data_dir, name, i, qz_max=data_gixd.QZ_MAX)

        # Preserve the original cartesian data (pre-subtraction)
        da_cart_raw = da_cart.sel(qz=slice(QZ_CUTOFF, None))

        # Bin both qz and qxy early for better SNR
        da_cart_binned = da_cart_raw.coarsen(
            qz=QZ_BIN, qxy=QXY_BIN, boundary="trim"
        ).mean()

        # Store original cartesian data
        all_2d_data[f"{i}_{p}_raw_cart"] = da_cart_binned

        # Apply inverse quadratic background subtraction
        da_cart_sub_invquad = None

        try:
            # Fit and subtract inverse quadratic background on binned data
            da_cart_sub_invquad, background, fit_mask = subtract_invquad_background(
                da_cart_binned,
                num_fitting_points=INVQUAD_NUM_FITTING_POINTS,
                use_global_fit=INVQUAD_USE_GLOBAL_FIT,
            )

            # Store inverse quadratic data
            all_2d_data[f"{i}_{p}_sub_invquad_cart"] = da_cart_sub_invquad
            all_2d_data[f"{i}_{p}_bg_invquad_cart"] = background
            # Note: fit_mask is not saved as it's not used in current plotting pipeline

            # Extract horizontal slice comparison using binned original and background
            horizontal_slices_ds = extract_horizontal_slice_comparison(
                da_cart_binned, background, name, i, p
            )

            if horizontal_slices_ds is not None:
                # Store horizontal slices for consolidated saving
                all_horizontal_slices[f"{i}_{p}_horizontal_slices"] = (
                    horizontal_slices_ds
                )

        except Exception as e:
            print(
                f"Warning: Failed to subtract inverse quadratic background for {name}_{i}_{p}: {e}"
            )
            da_cart_sub_invquad = None

        # Process polar and 1D intensity profiles
        if da_cart_sub_invquad is not None:
            da_polar_sub_invquad = gixd_cartesian2polar(
                da_cart_sub_invquad,
                dq=Q_BIN,
                dtau=TAU_BIN,  # Uses binned cartesian
            )

            # Extract I(q) profile
            intensity_q_sub_invquad = extract_intensity_q(
                da_polar_sub_invquad,
                q_range=(roi_iq[0], roi_iq[1]),
                tau_range=(roi_iq[2], roi_iq[3]),
            )

            # Extract I(tau) profile
            intensity_tau_sub_invquad = extract_intensity_tau(
                da_polar_sub_invquad,
                q_range=(roi_itau[0], roi_itau[1]),
                tau_range=(roi_itau[2], roi_itau[3]),
            )

            # Store 2D polar and 1D profile data
            all_2d_data[f"{i}_{p}_sub_invquad_polar"] = da_polar_sub_invquad
            all_1d_data[f"{i}_{p}_sub_invquad_Iq"] = intensity_q_sub_invquad
            all_1d_data[f"{i}_{p}_sub_invquad_Itau"] = intensity_tau_sub_invquad

    # Save organized data files consolidated by sample
    # Save 2D maps in one file
    if all_2d_data:
        ds_2d = xr.Dataset(all_2d_data)
        ds_2d.attrs = {
            "description": f"2D maps for {name}",
            "sample_name": name,
            "data_types": "cartesian maps, polar maps, background, fit masks",
        }
        ds_2d.to_netcdf(out_dir / f"{name}_2d_maps.nc")

    # Save 1D profiles in one file
    if all_1d_data:
        ds_1d = xr.Dataset(all_1d_data)
        ds_1d.attrs = {
            "description": f"1D intensity profiles for {name}",
            "sample_name": name,
            "data_types": "I(q) and I(tau) profiles",
        }
        ds_1d.to_netcdf(out_dir / f"{name}_1d_profiles.nc")

    # Save consolidated horizontal slices for the entire sample
    if all_horizontal_slices:
        # Create a consolidated dataset with all horizontal slices
        consolidated_horizontal_slices = {}
        for key, ds in all_horizontal_slices.items():
            # Add all variables from each horizontal slice dataset
            for var_name, var_data in ds.data_vars.items():
                consolidated_horizontal_slices[f"{key}_{var_name}"] = var_data

        if consolidated_horizontal_slices:
            ds_horizontal = xr.Dataset(consolidated_horizontal_slices)
            ds_horizontal.attrs = {
                "description": f"Horizontal slice profiles for {name}",
                "sample_name": name,
                "data_types": "horizontal slice profiles for multiple qz ranges",
                "qz_ranges": str(QZ_SLICE_RANGES),
            }
            ds_horizontal.to_netcdf(out_dir / f"{name}_horizontal_slices.nc")

    return None


def extract_horizontal_slice_comparison(
    da_cart_raw: xr.DataArray,
    da_background: xr.DataArray,
    name: str,
    index: int,
    pressure: float,
) -> Optional[xr.Dataset]:
    """
    Extract horizontal slice (constant qz) profiles for original data and background.

    Takes the mean over multiple qz ranges to create 1D profiles for each range.
    Returns a Dataset containing all slices.

    Parameters:
    -----------
    da_cart_raw : xr.DataArray
        Raw unsubtracted cartesian data
    da_background : xr.DataArray
        Fitted background data
    name : str
        Sample name
    index : int
        Sample index
    pressure : float
        Sample pressure

    Returns:
    --------
    xr.Dataset or None
        Dataset containing original, background, and difference profiles for all qz ranges
    """
    try:
        # Create lists to store profiles for each qz range
        raw_profiles = []
        bg_profiles = []
        diff_profiles = []
        qz_range_labels = []

        for qz_min, qz_max in QZ_SLICE_RANGES:
            # Select qz range
            da_raw_slice = da_cart_raw.sel(qz=slice(qz_min, qz_max))
            da_bg_slice = da_background.sel(qz=slice(qz_min, qz_max))

            # Average over qz dimension to get 1D profiles
            raw_profile = da_raw_slice.mean(dim="qz")
            bg_profile = da_bg_slice.mean(dim="qz")
            diff_profile = raw_profile - bg_profile

            # Add metadata to profiles
            common_attrs = {
                "description": f"Horizontal slice profile (qz mean: {qz_min}-{qz_max})",
                "sample_name": name,
                "index": index,
                "pressure": pressure,
                "qz_range_min": qz_min,
                "qz_range_max": qz_max,
                "model": "I(qxy,qz) = (A*qz + B)*qxy^-2 + C",
            }
            raw_profile.attrs = {**common_attrs, "profile_type": "original"}
            bg_profile.attrs = {**common_attrs, "profile_type": "background"}
            diff_profile.attrs = {**common_attrs, "profile_type": "difference"}

            raw_profiles.append(raw_profile)
            bg_profiles.append(bg_profile)
            diff_profiles.append(diff_profile)
            qz_range_labels.append(f"{qz_min:.1f}_{qz_max:.1f}")

        # Create Dataset with all profiles
        data_vars = {}
        for i, (orig, bg, diff, label) in enumerate(
            zip(raw_profiles, bg_profiles, diff_profiles, qz_range_labels)
        ):
            data_vars[f"original_qz_{label}"] = orig
            data_vars[f"background_qz_{label}"] = bg
            data_vars[f"difference_qz_{label}"] = diff

        # Create Dataset
        ds = xr.Dataset(data_vars)

        # Add global attributes
        ds.attrs = {
            "description": f"Horizontal slice profiles for {name} idx={index} p={pressure}",
            "sample_name": name,
            "index": index,
            "pressure": pressure,
            "qz_ranges": str(QZ_SLICE_RANGES),
            "model": "I(qxy,qz) = (A*qz + B)*qxy^-2 + C",
        }

        return ds

    except Exception as e:
        print(
            f"Warning: Failed to extract horizontal slice profiles for {name}_{index}_{pressure}: {e}"
        )
        return None


def main():
    parser = argparse.ArgumentParser(description="Process GIXD data")
    parser.add_argument(
        "--experiment",
        type=str,
        default="1",
        help='Experiment number (e.g., "1", "2"). Defaults to "1"',
    )
    args = parser.parse_args()

    # Normalize experiment number
    experiment_num = args.experiment.replace("experiment_", "") if "experiment" in args.experiment else args.experiment
    
    # Set environment variable internally for data module and reload to pick up the experiment
    os.environ["EXPERIMENT"] = experiment_num
    importlib.reload(data_gixd)
    from data_gixd import get_samples, ROI_IQ, ROI_ITAU

    data_dir = Path.home() / "data" / "langmuir" / experiment_num / "gixd"
    processed_dir = Path.home() / "results" / "langmuir" / experiment_num / "gixd"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing experiment {experiment_num}...")
    for s in get_samples():
        process_sample(data_dir, processed_dir, s, ROI_IQ, ROI_ITAU)
    print("GIXD processing completed.")


if __name__ == "__main__":
    main()
