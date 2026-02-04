"""
Data validation script.

Validates that data files exist for all indices specified in the YAML configuration files.
Uses tqdm to show progress per sample name.
"""

import sys
from pathlib import Path
from typing import TypedDict

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

console = Console()


class Sample(TypedDict):
    name: str
    full_name: str
    index: list[int]
    pressure: list[float]


def check_gixd_files(data_path: Path, name: str, index: int) -> tuple[bool, list[str]]:
    """
    Check if GIXD files exist for a given sample name and index.

    Args:
        data_path: Base directory containing sample subfolders
        name: Sample name (subfolder name)
        index: Integer index

    Returns:
        Tuple of (all_exist, missing_files)
    """
    sample_dir = data_path / name
    missing = []

    files = [
        f"{name}_{index}_{index}_combined_I.dat",
        f"{name}_{index}_{index}_combined_Qxy.dat",
        f"{name}_{index}_{index}_combined_Qz.dat",
    ]

    for filename in files:
        filepath = sample_dir / filename
        if not filepath.exists():
            missing.append(str(filepath))

    return len(missing) == 0, missing


def check_gixos_files(data_path: Path, name: str, index: int) -> tuple[bool, list[str]]:
    """
    Check if GIXOS files exist for a given sample name and index.

    Args:
        data_path: Base directory containing sample subfolders
        name: Sample name (subfolder name)
        index: Integer index

    Returns:
        Tuple of (all_exist, missing_files)
    """
    sample_dir = data_path / name
    missing = []

    suffixes = ["SF", "R", "DS2RRF"]
    for suffix in suffixes:
        filename = f"{name}_{index:05d}_{suffix}.dat"
        filepath = sample_dir / filename
        if not filepath.exists():
            missing.append(str(filepath))

    return len(missing) == 0, missing


def validate_experiment(experiment_num: str, data_base: Path) -> dict:
    """
    Validate data files for a given experiment.

    Args:
        experiment_num: Experiment number (e.g., "1", "2")
        data_base: Base directory containing data subdirectories (e.g., ./data)

    Returns:
        Dictionary with validation results
    """
    results: dict = {
        "experiment": experiment_num,
        "gixd": {"valid": True, "samples": {}},
        "gixos": {"valid": True, "samples": {}},
    }

    # Load GIXD YAML
    gixd_yaml = data_base / experiment_num / "gixd.yaml"
    if not gixd_yaml.exists():
        console.print(f"[yellow]Warning:[/yellow] {gixd_yaml} not found, skipping GIXD validation")
    else:
        with gixd_yaml.open() as f:
            gixd_data = yaml.safe_load(f)

        gixd_path = data_base / experiment_num / "gixd"
        if gixd_path.exists():
            # Validate backgrounds
            backgrounds = gixd_data.get("background", [])
            for bg in backgrounds:
                bg_name = bg["name"]
                bg_indices = bg["index"]
                bg_missing = []
                for idx in bg_indices:
                    exists, missing = check_gixd_files(gixd_path, bg_name, idx)
                    if not exists:
                        bg_missing.extend(missing)
                if bg_missing:
                    results["gixd"]["samples"][bg_name] = {
                        "valid": False,
                        "missing": bg_missing,
                    }
                    results["gixd"]["valid"] = False

            # Validate samples
            samples = gixd_data.get("sample", [])
            for sample in tqdm(samples, desc=f"GIXD exp {experiment_num}", leave=False):
                name = sample["name"]
                indices = sample["index"]
                missing_all = []
                for idx in indices:
                    exists, missing = check_gixd_files(gixd_path, name, idx)
                    if not exists:
                        missing_all.extend(missing)
                if missing_all:
                    results["gixd"]["samples"][name] = {
                        "valid": False,
                        "missing": missing_all,
                    }
                    results["gixd"]["valid"] = False
                else:
                    results["gixd"]["samples"][name] = {"valid": True, "missing": []}
        else:
            console.print(f"[yellow]Warning:[/yellow] {gixd_path} directory not found")

    # Load GIXOS YAML
    gixos_yaml = data_base / experiment_num / "gixos.yaml"
    if not gixos_yaml.exists():
        console.print(f"[yellow]Warning:[/yellow] {gixos_yaml} not found, skipping GIXOS validation")
    else:
        with gixos_yaml.open() as f:
            gixos_data = yaml.safe_load(f)

        gixos_path = data_base / experiment_num / "gixos"
        if gixos_path.exists():
            # Validate samples
            samples = gixos_data.get("sample", [])
            for sample in tqdm(samples, desc=f"GIXOS exp {experiment_num}", leave=False):
                name = sample["name"]
                indices = sample["index"]
                missing_all = []
                for idx in indices:
                    exists, missing = check_gixos_files(gixos_path, name, idx)
                    if not exists:
                        missing_all.extend(missing)
                if missing_all:
                    results["gixos"]["samples"][name] = {
                        "valid": False,
                        "missing": missing_all,
                    }
                    results["gixos"]["valid"] = False
                else:
                    results["gixos"]["samples"][name] = {"valid": True, "missing": []}
        else:
            console.print(f"[yellow]Warning:[/yellow] {gixos_path} directory not found")

    return results


def print_summary(results: dict, data_base: Path):
    """
    Print a summary of data statistics for each experiment.

    Args:
        results: Validation results dictionary
        data_base: Base directory containing data subdirectories
    """
    experiment_num = results["experiment"]
    
    # GIXD Summary
    gixd_table = Table(title=f"GIXD - Experiment {experiment_num}", show_header=True, header_style="bold cyan")
    gixd_table.add_column("Type", style="cyan")
    gixd_table.add_column("Name", style="green")
    gixd_table.add_column("Measurements", justify="right", style="yellow")
    
    gixd_yaml = data_base / experiment_num / "gixd.yaml"
    if gixd_yaml.exists():
        with gixd_yaml.open() as f:
            gixd_data = yaml.safe_load(f)
        
        backgrounds = gixd_data.get("background", [])
        samples = gixd_data.get("sample", [])
        
        total_gixd_indices = 0
        for bg in backgrounds:
            num_indices = len(bg["index"])
            total_gixd_indices += num_indices
            full_name = bg.get("full_name", bg["name"])
            gixd_table.add_row("Background", full_name, str(num_indices))
        
        for sample in samples:
            num_indices = len(sample["index"])
            total_gixd_indices += num_indices
            full_name = sample.get("full_name", sample["name"])
            gixd_table.add_row("Sample", full_name, str(num_indices))
        
        gixd_table.add_row("[bold]Total[/bold]", "", f"[bold]{total_gixd_indices}[/bold]", style="bold")
        console.print(gixd_table)
    
    # GIXOS Summary
    gixos_table = Table(title=f"GIXOS - Experiment {experiment_num}", show_header=True, header_style="bold magenta")
    gixos_table.add_column("Name", style="green")
    gixos_table.add_column("Measurements", justify="right", style="yellow")
    
    gixos_yaml = data_base / experiment_num / "gixos.yaml"
    if gixos_yaml.exists():
        with gixos_yaml.open() as f:
            gixos_data = yaml.safe_load(f)
        
        samples = gixos_data.get("sample", [])
        
        total_gixos_indices = 0
        for sample in samples:
            num_indices = len(sample["index"])
            total_gixos_indices += num_indices
            full_name = sample.get("full_name", sample["name"])
            gixos_table.add_row(full_name, str(num_indices))
        
        gixos_table.add_row("[bold]Total[/bold]", f"[bold]{total_gixos_indices}[/bold]", style="bold")
        console.print(gixos_table)


def main():
    """Main validation function."""
    # Use ./data as the data base path
    script_dir = Path(__file__).parent
    data_base = script_dir / "data"

    # Find all experiment directories
    experiments = [
        d.name
        for d in data_base.iterdir()
        if d.is_dir() and ((d / "gixd.yaml").exists() or (d / "gixos.yaml").exists())
    ]
    experiments.sort()

    console.print(f"\n[bold cyan]Validating data files for experiment(s):[/bold cyan] {', '.join(experiments)}\n")

    all_valid = True
    for exp in experiments:
        console.print(Panel(f"Experiment {exp}", style="bold blue", expand=False))
        results = validate_experiment(exp, data_base)

        # Print GIXD results
        gixd_results = results["gixd"]
        if gixd_results["valid"]:
            console.print("[bold green]✓ GIXD:[/bold green] All files exist")
        else:
            all_valid = False
            console.print("[bold red]✗ GIXD:[/bold red] Missing files found:")
            for sample_name, sample_result in gixd_results["samples"].items():
                if not sample_result["valid"]:
                    console.print(f"  [red]{sample_name}:[/red]")
                    for missing_file in sample_result["missing"]:
                        console.print(f"    - [dim]{missing_file}[/dim]")

        # Print GIXOS results
        gixos_results = results["gixos"]
        if gixos_results["valid"]:
            console.print("[bold green]✓ GIXOS:[/bold green] All files exist")
        else:
            all_valid = False
            console.print("[bold red]✗ GIXOS:[/bold red] Missing files found:")
            for sample_name, sample_result in gixos_results["samples"].items():
                if not sample_result["valid"]:
                    console.print(f"  [red]{sample_name}:[/red]")
                    for missing_file in sample_result["missing"]:
                        console.print(f"    - [dim]{missing_file}[/dim]")
        
        # Print summary for this experiment
        console.print()
        print_summary(results, data_base)
        console.print()

    # Exit with error code if validation failed
    if not all_valid:
        console.print("[bold red]\n✗ Validation failed - some files are missing![/bold red]")
        sys.exit(1)
    else:
        console.print("[bold green]\n✓ All data files validated successfully![/bold green]")


if __name__ == "__main__":
    main()
