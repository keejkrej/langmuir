import numpy as np
import xarray as xr
from pathlib import Path
from utils.math.transform import cartesian2polar
from utils.math.peak import detect_peaks_median


def load_gixd_xarray(data_path, name, index, qz_max: float = 1.0):
    """
    Load GIXD combined intensity data as xarray.DataArray.

    Filters out data at high qz (qz > qz_max) during loading. Typically qz_max
    is set via the gixd.yaml config (qz_max).
    """
    data_path = Path(data_path) / name
    intensity = np.loadtxt(data_path / f"{name}_{index}_{index}_combined_I.dat")
    qxy = np.loadtxt(data_path / f"{name}_{index}_{index}_combined_Qxy.dat")
    qz = np.loadtxt(data_path / f"{name}_{index}_{index}_combined_Qz.dat")
    da = xr.DataArray(
        intensity, dims=("qz", "qxy"), coords={"qz": qz, "qxy": qxy}, name="intensity"
    )
    return da.sel(qz=slice(None, qz_max))


def gixd_cartesian2polar(da_cart, dq, dtau):
    intensity_polar, q, tau = cartesian2polar(
        da_cart.values, da_cart["qxy"].values, da_cart["qz"].values, dq, dtau
    )
    return xr.DataArray(
        intensity_polar,
        dims=("tau", "q"),
        coords={"tau": np.rad2deg(tau), "q": q},
        name="intensity_polar",
    )


def extract_intensity_q(da_polar, q_range=None, tau_range=None, method="mean"):
    da = da_polar
    if q_range:
        da = da.sel(q=slice(q_range[0], q_range[1]))
    if tau_range:
        da = da.sel(tau=slice(tau_range[0], tau_range[1]))
    if method == "mean":
        return da.mean(dim="tau")
    if method == "sum":
        return da.sum(dim="tau")
    raise ValueError("method must be 'mean' or 'sum'")


def extract_intensity_tau(da_polar, q_range=None, tau_range=None, method="mean"):
    da = da_polar
    if q_range:
        da = da.sel(q=slice(q_range[0], q_range[1]))
    if tau_range:
        da = da.sel(tau=slice(tau_range[0], tau_range[1]))
    if method == "mean":
        return da.mean(dim="q")
    if method == "sum":
        return da.sum(dim="q")
    raise ValueError("method must be 'mean' or 'sum'")


def remove_peaks_from_1d(
    intensity_array, coords_array, window_size, sigma_threshold, exclusion_radius
):
    """
    Remove peaks from 1D intensity profile by filtering out peak regions
    """
    peaks = detect_peaks_median(
        intensity_array, coords_array, window_size, sigma_threshold
    )

    # Create mask for non-peak regions
    mask = np.ones_like(intensity_array, dtype=bool)
    for peak_coord, _ in peaks:
        peak_mask = np.abs(coords_array - peak_coord) > exclusion_radius
        mask = mask & peak_mask

    # Return filtered arrays (only non-peak regions)
    filtered_intensity = intensity_array[mask]
    filtered_coords = coords_array[mask]

    return filtered_intensity, filtered_coords
