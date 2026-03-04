## GUV scale-up model

This repository contains a Monte Carlo scale-up model for **germicidal UV (GUV) lamps** and **repurposed fluorescent lamps**.  
The model estimates how quickly global clean air delivery rate (CADR) can ramp up and compares it to CADR requirements for vital workers.

The codebase follows ALLFED's recommended project structure (`src/`, `scripts/`, `data/`, `results/`, `docs/`) and is fully tested and documented via GitHub Actions.

---

## What this model does

- **Goal**: Estimate how quickly GUV and repurposed fluorescent lamps can ramp up to meet CADR needs for vital workers during a severe pandemic scenario.
- **Core components**:
  - **Monte Carlo sampling** of:
    - Global markets for UV and fluorescent lamps.
    - Usable fractions of those markets.
    - Costs per unit and CADR per unit.
  - **Scale-up dynamics** (`src/scale_up_model.py`):
    - Factory utilization ramp-up.
    - Additional production and (optional) repurposing.
  - **Outputs**:
    - Time series of CADR for UV and repurposed fluorescent lamps.
    - Comparison to CADR requirements for vital workers.
    - Summary CSV and publication-ready plot.

---

## Repository structure

- **`src/`**
  - **`mc_distributions.py`**: Monte Carlo sampling helpers (normal, lognormal, GPD).
  - **`scale_up_model.py`**: Growth model for CADR production over time.
  - **`plotter.py`**: Plotting utilities for UV vs requirement ramp-up.
  - **`run_analysis.py`**: Command-line entry point that runs the full analysis from a YAML config, saving CSV and PNG results.
- **`scripts/`**
  - **`estimate_scaleup.ipynb`**: Walkthrough notebook mirroring the main analysis step by step.
- **`results/`**
  - Default output location for CSVs and figures produced by `run_analysis.py` and the notebook.
- **`docs/`**
  - MkDocs-based documentation (auto-generated via GitHub Actions when configured).

---

## Installation

### 1. Prerequisites

- **conda** (or **mamba**) installed (e.g. via Miniconda or Miniforge).
- Git.

### 2. Clone the repository

```bash
git clone https://github.com/ALLFED/guv-scaleup-model.git
cd guv-scaleup-model
```

### 3. Create and activate the environment

Using **mamba** (recommended for speed):

```bash
mamba env create -f environment.yml
conda activate guv-scaleup-model
```

Using **conda**:

```bash
conda env create -f environment.yml
conda activate guv-scaleup-model
```

The environment file installs:

- Core stack: **numpy**, **pandas**, **scipy**, **matplotlib**.
- Testing: **pytest**.
- Docs: **mkdocs**, **mkgendocs**, **pyyaml**.
- Editable install of this package (`pip install -e .`).

---

## How to run the model

You can run the model either via the **notebook** or via the **command-line entry point**.

### Option 1: Notebook walkthrough

1. Activate the environment:

   ```bash
   conda activate guv-scaleup-model
   ```

2. Start Jupyter (or VS Code, or your preferred IDE) and open:

   - `scripts/estimate_scaleup.ipynb`

3. Run all cells from top to bottom.  
   The notebook will:

   - Sample parameters.
   - Compute baseline CADR.
   - Run the growth model for UV and repurposed fluorescent lamps.
   - Plot supply vs CADR requirement over time.

### Option 2: Command-line analysis (`src/run_analysis.py`)

The script `src/run_analysis.py` runs the full analysis from a YAML configuration file and writes outputs to the `results/` folder.

#### 1. Create a config file

Create `analysis_config.yml` in the **repository root** (next to `environment.yml`), for example:

```yaml
n_sims: 1000
months: 6
confidence_interval: 90

uv:
  global_market_low: 6.3e8
  global_market_high: 5.0e9
  percent_usable_low: 0.60
  percent_usable_high: 0.90
  percent_of_annual_low: 0.5/12
  percent_of_annual_high: 1/12
  cost_per_unit_low: 100
  cost_per_unit_high: 700
  cadr_per_unit_low: 100
  cadr_per_unit_high: 500
  growth:
    utilization_start: 1.0
    utilization_end: 1.0
    utilization_ramp_months: 3
    additional_annual_production: 1/12
    repurposed_ramp_months: 1
    repurposed_annual_production: 3/12

fluoro:
  global_market_low: 2.8e9
  global_market_high: 7.9e9
  percent_usable_low: 0.60
  percent_usable_high: 0.90
  cost_per_unit_low: 1
  cost_per_unit_high: 3
  growth:
    utilization_start: 0.0
    utilization_end: 0.7
    utilization_ramp_months: 3
    additional_annual_production: 1/12
    repurposed_ramp_months: 0
    repurposed_annual_production: 0

requirement:
  ashrae_low: 35*5
  ashrae_high: 45*5
  vital_workers_low: 0.5e9
  vital_workers_high: 1e9

outputs:
  csv_path: analysis_summary.csv
  png_path: uv_fluoro_ramp.png
```

You can change any of these values to explore different scenarios (e.g. more months, different utilization ramps, different vital worker counts).

#### 2. Run the analysis

From the **repository root**:

```bash
conda activate guv-scaleup-model
python -m src.run_analysis --config analysis_config.yml
```

If you omit `--config`, the script defaults to `analysis_config.yml` in the repo root.

---

## Outputs

By default, the command-line analysis writes to the `results/` folder:

- **CSV**: `results/analysis_summary.csv`
  - Columns include:
    - `month`
    - `uv_median`, `uv_p_lower`, `uv_p_upper`
    - `fluoro_median`, `fluoro_p_lower`, `fluoro_p_upper`
    - `median_cadr_requirement`, `cadr_requirement_p_lower`, `cadr_requirement_p_upper`
- **PNG**: `results/uv_fluoro_ramp.png`
  - High-resolution figure (configured at **300 dpi** in the plotting utility) suitable for reports or papers.

You can change filenames in the YAML config under the `outputs` section.

---

## Testing

- All tests live in the `tests/` directory and use **pytest**.
- To run the test suite locally (with the environment activated):

```bash
pytest
```

GitHub Actions (`.github/workflows/testing.yml`) automatically run the tests on every push and pull request to `main`.

---

## Documentation

MkDocs-based documentation is set up via GitHub Actions (`.github/workflows/docs.yml`).  
Once configured with GitHub Pages, pushes to `main` will:

- Regenerate API docs from Python docstrings.
- Build the MkDocs site.
- Deploy it to the `gh-pages` branch.

For now, the main user-facing documentation is this `README` plus the example notebook in `scripts/`.

---

## License

This project is released under the **Apache 2.0 License**.  
See the `LICENSE` file in this repository or the [Apache 2.0 overview](https://opensource.org/licenses/Apache-2.0) for details.

You are free to use, modify, and redistribute the code, but ALLFED and the authors assume no liability for its use.
