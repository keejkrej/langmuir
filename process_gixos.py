import argparse
import importlib
import os
from pathlib import Path
import yaml
import xarray as xr
from utils.fit.gixos import fit_rfxsf, fit_r
import data_gixos


def save_fit_nc(
    sample: str, idx: int, pressure: float, method: str, results: dict, out_dir: Path
):
    """Save raw and fitted curve to NetCDF with parameters in attrs."""
    q = results["q_data"]
    ds = xr.Dataset(
        data_vars=dict(
            R_data=("q", results["R_data"]),
            dR_data=("q", results["dR_data"]),
            R_fit=("q", results["R_fit"]),
        ),
        coords=dict(q=("q", q)),
        attrs={
            "method": method.upper(),
            "sample": sample,
            "index": int(idx),
            "pressure_mN_per_m": float(pressure),
            "file": results.get("file", ""),
            "total_thickness_A": float(results.get("total_thickness", float("nan"))),
            "tails_thick_A": float(results.get("tails_thick", float("nan"))),
            "tails_sld": float(results.get("tails_sld", float("nan"))),
            "heads_thick_A": float(results.get("heads_thick", float("nan"))),
            "heads_sld": float(results.get("heads_sld", float("nan"))),
            "heads_vfsolv": float(results.get("heads_vfsolv", float("nan"))),
            "scale": float(results.get("scale", float("nan"))),
            "bkg": float(results.get("bkg", float("nan"))),
            "chi2_red": float(results.get("chi2_red", float("nan"))),
            "sigma_R_A": float(results.get("sigma_R", float("nan"))),
            "npts": int(results.get("npts", 0)),
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{sample}_{idx}_{pressure}_{method.lower()}.nc"
    path = out_dir / fname
    ds.to_netcdf(path)
    print(f"Saved {path}")


def save_fit_yaml(
    sample: str, idx: int, pressure: float, method: str, results: dict, out_dir: Path
):
    """Save fitted parameter summary to YAML."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{sample}_{idx}_{pressure}_{method.lower()}.yaml"
    path = out_dir / fname
    payload = {
        "method": method.upper(),
        "sample": sample,
        "index": int(idx),
        "pressure_mN_per_m": float(pressure),
        "file": results.get("file", ""),
        "total_thickness_A": float(results.get("total_thickness", float("nan"))),
        "tails_thick_A": float(results.get("tails_thick", float("nan"))),
        "tails_sld": float(results.get("tails_sld", float("nan"))),
        "heads_thick_A": float(results.get("heads_thick", float("nan"))),
        "heads_sld": float(results.get("heads_sld", float("nan"))),
        "heads_vfsolv": float(results.get("heads_vfsolv", float("nan"))),
        "scale": float(results.get("scale", float("nan"))),
        "bkg": float(results.get("bkg", float("nan"))),
        "chi2_red": float(results.get("chi2_red", float("nan"))),
        "sigma_R_A": float(results.get("sigma_R", float("nan"))),
        "npts": int(results.get("npts", 0)),
    }
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Process GIXOS data")
    parser.add_argument(
        "--experiment",
        type=str,
        default="1",
        help='Experiment number (e.g., "1", "2"). Defaults to "1"',
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Use test_sample from the experiment YAML for faster iteration",
    )
    args = parser.parse_args()

    # Normalize experiment number
    experiment_num = args.experiment.replace("experiment_", "") if "experiment" in args.experiment else args.experiment
    
    # Set environment variable internally for data module and reload to pick up the experiment
    os.environ["EXPERIMENT"] = experiment_num
    importlib.reload(data_gixos)
    from data_gixos import get_fit_config, get_samples

    data_path = Path.home() / "data" / "langmuir" / experiment_num / "gixos"
    processed_dir = Path.home() / "results" / "langmuir" / experiment_num / "gixos"
    processed_dir.mkdir(parents=True, exist_ok=True)
    fit_config = get_fit_config()

    print(f"Processing experiment {experiment_num}...")
    print("Starting GIXOS fitting.")

    # No global aggregation; per-measurement YAMLs/NCs only

    for s in get_samples(args.test):
        name, indices, pressures = s["name"], s["index"], s["pressure"]
        print(f"Processing sample {name}...")

        for idx, p in zip(indices, pressures):
            sf_file = data_path / name / f"{name}_{idx:05d}_SF.dat"
            r_file = data_path / name / f"{name}_{idx:05d}_R.dat"

            if not sf_file.exists() or not r_file.exists():
                print(f"Skip {name}_{idx}: missing SF or R file")
                continue

            try:
                rfxsf_results = fit_rfxsf(
                    str(sf_file),
                    save_plot=None,
                    show_plot=False,
                    verbose=False,
                    de_maxiter=fit_config["rfxsf_de_maxiter"],
                )
                r_results = fit_r(
                    str(r_file),
                    save_plot=None,
                    show_plot=False,
                    verbose=False,
                    de_maxiter=fit_config["r_de_maxiter"],
                )

                # Save intermediate NetCDFs and YAMLs per method
                out_dir = processed_dir / name
                save_fit_nc(name, idx, p, "rfxsf", rfxsf_results, out_dir)
                save_fit_yaml(name, idx, p, "rfxsf", rfxsf_results, out_dir)
                save_fit_nc(name, idx, p, "r", r_results, out_dir)
                save_fit_yaml(name, idx, p, "r", r_results, out_dir)

                # all parameters are persisted per-measurement
            except Exception as e:
                print(f"Error processing {name}_{idx}: {e}")

    # No summary file generated; per-measurement YAMLs contain the parameter summaries

    print("GIXOS fitting completed.")


if __name__ == "__main__":
    main()
