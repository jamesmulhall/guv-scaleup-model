## GUV scale-up model

[![DOI](https://zenodo.org/badge/520046482.svg)]()
![Testing](https://github.com/jamesmulhall/guv-scaleup-model/actions/workflows/testing.yml/badge.svg?branch=main)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


This repository contains a Monte Carlo scale-up model for **germicidal UV (GUV) lamps** and **repurposed fluorescent lamps**.  
The model estimates how quickly global clean air delivery rate (CADR) can ramp up and compares it to CADR requirements for vital workers.

![Ramp up plot](https://github.com/jamesmulhall/guv-scaleup-model/blob/main/results/uv_scaleup_5x_covid.png)

---

![Pathogen susceptibility plot](https://github.com/jamesmulhall/guv-scaleup-model/blob/main/results/uv_scaleup_by_pathogen.png)


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

Modify `analysis_config.yml` in the **repository root** (next to `environment.yml`) to change parameters.


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
