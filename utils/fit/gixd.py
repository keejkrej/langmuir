from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


def mirrored_gaussian(
    x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float = 0.0
) -> np.ndarray:
    """
    Mirrored Gaussian function: gauss(x0) + gauss(-x0) + offset

    Parameters:
    -----------
    x : array-like
        Input coordinates
    amplitude : float
        Amplitude of each Gaussian component
    center : float
        Center position x0
    sigma : float
        Standard deviation of each Gaussian
    offset : float, optional
        Baseline offset (default: 0.0)

    Returns:
    --------
    array-like
        Fitted intensity values
    """
    gauss1 = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    gauss2 = amplitude * np.exp(-0.5 * ((x + center) / sigma) ** 2)
    return gauss1 + gauss2 + offset


def centered_gaussian(
    x: np.ndarray, amplitude: float, sigma: float, offset: float = 0.0
) -> np.ndarray:
    """
    Gaussian constrained to be centered at tau = 0.

    Parameters:
    -----------
    x : array-like
        Input coordinates
    amplitude : float
        Peak amplitude above offset
    sigma : float
        Standard deviation
    offset : float, optional
        Baseline offset (default: 0.0)

    Returns:
    --------
    array-like
        Fitted intensity values
    """
    return amplitude * np.exp(-0.5 * (x / sigma) ** 2) + offset


def fit_mirrored_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    initial_guess: Optional[Tuple[float, float, float, float]] = None,
    bounds: Optional[Tuple[list, list]] = None,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Fit a mirrored Gaussian function: gauss(x0) + gauss(-x0) + offset to data.
    Ignores points with tau < 5 for fitting.

    Parameters:
    -----------
    x : array-like
        Coordinate values (tau values)
    y : array-like
        Intensity values
    initial_guess : tuple of (amplitude, center, sigma, offset), optional
        Initial parameter guesses. If None, uses automatic estimation.
    bounds : tuple of (lower_bounds, upper_bounds), optional
        Parameter bounds for fitting. If None, uses default bounds.

    Returns:
    --------
    tuple of (amplitude, center, sigma, offset, fitted_y)
        Fitted parameters and fitted curve evaluated on full x range
    """
    mask = x >= 5
    x_fit = x[mask]
    y_fit = y[mask]

    if len(x_fit) < 5:
        print("Warning: Not enough points (tau >= 5) for mirrored Gaussian fitting")
        center_idx = np.argmax(y)
        center = abs(x[center_idx])
        amplitude = np.max(y) / 2.0
        sigma = 0.5
        offset = max(0, min(2, np.min(y)))
        fitted_y = mirrored_gaussian(x, amplitude, center, sigma, offset)
        return amplitude, center, sigma, offset, fitted_y

    if initial_guess is None:
        amplitude = np.max(y_fit) / 2.0

        center_idx_fit = np.argmax(y_fit)
        center = abs(x_fit[center_idx_fit])

        half_max = amplitude
        left_idx = center_idx_fit
        while left_idx > 0 and y_fit[left_idx] > half_max:
            left_idx -= 1
        right_idx = center_idx_fit
        while right_idx < len(y_fit) - 1 and y_fit[right_idx] > half_max:
            right_idx += 1

        if right_idx > left_idx:
            fwhm = x_fit[right_idx] - x_fit[left_idx]
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            sigma = 0.1

        offset = max(0, min(2, np.min(y_fit)))
        initial_guess = (amplitude, center, sigma, offset)

    if bounds is None:
        bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, 2])

    try:
        popt, _ = curve_fit(
            mirrored_gaussian,
            x_fit,
            y_fit,
            p0=initial_guess,
            bounds=bounds,
        )

        amplitude_fit, center_fit, sigma_fit, offset_fit = popt
        fitted_y = mirrored_gaussian(
            x, amplitude_fit, center_fit, sigma_fit, offset_fit
        )

        return amplitude_fit, center_fit, sigma_fit, offset_fit, fitted_y

    except Exception as e:
        print(f"Warning: Mirrored Gaussian fitting failed: {e}")
        amplitude, center, sigma, offset = initial_guess
        fitted_y = mirrored_gaussian(x, amplitude, center, sigma, offset)
        return amplitude, center, sigma, offset, fitted_y


def fit_centered_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    initial_guess: Optional[Tuple[float, float, float]] = None,
    bounds: Optional[Tuple[list, list]] = None,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Fit a Gaussian centered at tau = 0 to data.
    Ignores points with tau < 5 for fitting, matching mirrored Gaussian fitting.

    Parameters:
    -----------
    x : array-like
        Coordinate values (tau values)
    y : array-like
        Intensity values
    initial_guess : tuple of (amplitude, sigma, offset), optional
        Initial parameter guesses. If None, uses automatic estimation.
    bounds : tuple of (lower_bounds, upper_bounds), optional
        Parameter bounds for fitting. If None, uses default bounds.

    Returns:
    --------
    tuple of (amplitude, sigma, offset, fitted_y)
        Fitted parameters and fitted curve evaluated on full x range
    """
    mask = x >= 5
    x_fit = x[mask]
    y_fit = y[mask]

    if len(x_fit) < 4:
        print("Warning: Not enough points (tau >= 5) for centered Gaussian fitting")
        amplitude = np.max(y) - np.min(y)
        sigma = 0.5
        offset = max(0, min(2, np.min(y)))
        fitted_y = centered_gaussian(x, amplitude, sigma, offset)
        return amplitude, sigma, offset, fitted_y

    if initial_guess is None:
        offset = max(0, min(2, np.min(y_fit)))
        amplitude = max(np.max(y_fit) - offset, 0)
        sigma = 10.0
        initial_guess = (amplitude, sigma, offset)

    if bounds is None:
        bounds = ([0, 0, 0], [np.inf, np.inf, 2])

    try:
        popt, _ = curve_fit(
            centered_gaussian,
            x_fit,
            y_fit,
            p0=initial_guess,
            bounds=bounds,
        )

        amplitude_fit, sigma_fit, offset_fit = popt
        fitted_y = centered_gaussian(x, amplitude_fit, sigma_fit, offset_fit)

        return amplitude_fit, sigma_fit, offset_fit, fitted_y

    except Exception as e:
        print(f"Warning: Centered Gaussian fitting failed: {e}")
        amplitude, sigma, offset = initial_guess
        fitted_y = centered_gaussian(x, amplitude, sigma, offset)
        return amplitude, sigma, offset, fitted_y
