# GIXD-GIXOS Langmuir Analysis Pipeline

## Development Commands

### Running the Pipeline
```bash
# Validate data files (check if all files exist for configured indices)
uv run validate_data.py   # Validate all experiments

# Run full GIXD pipeline (clean, process, plot)
./run_gixd.sh

# Run individual components
uv run process_gixd.py    # Process GIXD data
uv run plot_gixd.py       # Plot GIXD results
uv run process_gixos.py   # Process GIXOS data  
uv run plot_gixos.py      # Plot GIXOS results
uv run plot_paper.py      # Generate paper figures

# Select specific experiment (defaults to experiment 1)
EXPERIMENT=2 uv run process_gixd.py
```

### Code Quality
```bash
# No formal test framework - validation is done through pipeline execution
# Use ruff for linting (if available)
uv run ruff check .       # Lint code
uv run ruff format .      # Format code
```

## Code Style Guidelines

### Project Structure
- Modular organization under `utils/` with domain-specific subdirectories:
  - `utils/data/` - Data loading utilities (gixd.py, gixos.py)
  - `utils/math/` - Mathematical functions (com.py, detrend.py, peak.py, transform.py)
  - `utils/fit/` - Fitting routines (gixd.py, gixos.py)  
  - `utils/background/` - Background subtraction (invquad.py)
- Processing scripts: `process_*.py`
- Plotting scripts: `plot_*.py`
- Data configuration: `data/{experiment}/gixd.yaml`, `gixos.yaml` - YAML configs for each experiment
- Data loading: `data_*.py` - Python modules that load from YAML configs
- Validation: `validate_data.py` - Script to validate data file existence

### Import Conventions
- Standard library imports first, then third-party, then local imports
- Use `from pathlib import Path` for file paths
- Scientific imports: `numpy as np`, `import xarray as xr`, `from scipy.optimize import curve_fit`
- Local imports: `from utils.data.gixd import ...`

### Type Hints
- Use type hints consistently: `def func(x: np.ndarray, y: float) -> Optional[np.ndarray]:`
- Import from `typing`: `Optional`, `Tuple`, `List`
- Use `Path` type for file paths

### Naming Conventions  
- Functions: snake_case with descriptive names (`extract_intensity_q`, `subtract_invquad_background`)
- Constants: UPPER_SNAKE_CASE (`SUBTRACT_WATER`, `QZ_CUTOFF`)
- Variables: snake_case, descriptive but concise (`da_cart`, `intensity_q`)
- DataArray/Dataset names: prefix with `da_` or `ds_`

### Error Handling
- Use try-except blocks for fitting operations with fallback behavior
- Print warnings for failed operations but continue processing
- Validate inputs early (e.g., array length checks)
- Return `None` for optional operations that may fail

### Documentation
- Use docstrings in NumPy format for all functions
- Include parameter descriptions with types and return value descriptions
- Add inline comments for complex mathematical operations
- Module-level docstrings explain purpose and configuration options

### Data Configuration
- Data is configured in YAML files under `data/{experiment}/`:
  - `data/{experiment}/gixd.yaml` - GIXD sample configuration
  - `data/{experiment}/gixos.yaml` - GIXOS sample configuration
- Each experiment has its own YAML files with:
  - `background` - Background/reference samples (list for GIXD)
  - `sample` - Sample definitions with `name`, `full_name`, `index`, `pressure`
  - `test_sample` - Test samples for quick validation
  - `is_test` - Boolean flag (GIXD only)
  - `roi_iq`, `roi_itau` - Region of interest definitions (GIXD only)
- Experiment selection via `EXPERIMENT` environment variable (defaults to "1")
- Data files are stored in `data/{experiment}/gixd/{sample_name}/` and `data/{experiment}/gixos/{sample_name}/`

### Data Processing Patterns
- Use xarray for multi-dimensional data with labeled coordinates
- Process data in loops over samples/pressures/indices
- Store intermediate results in organized dictionaries for batch saving
- Use xarray Dataset attributes for metadata
- Apply coordinate slicing with `sel()` and `mean()` for reductions
- Load sample configurations from `data_gixd.py` and `data_gixos.py` modules