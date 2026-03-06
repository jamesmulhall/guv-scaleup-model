## GUV scale-up model

<!-- [![DOI](https://zenodo.org/badge/520046482.svg)]() -->
![Testing](https://github.com/jamesmulhall/guv-scaleup-model/actions/workflows/testing.yml/badge.svg?branch=main)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


This repository contains a Monte Carlo scale-up model for **germicidal UV (GUV) lamps** and **repurposed fluorescent lamps**. The model estimates how quickly global clean air delivery rate (CADR) can ramp up and compares it to CADR requirements for vital workers.

![Ramp up plot](https://github.com/jamesmulhall/guv-scaleup-model/blob/main/results/uv_scaleup_5x_covid.png)

---

![Pathogen susceptibility plot](https://github.com/jamesmulhall/guv-scaleup-model/blob/main/results/uv_scaleup_by_pathogen.png)


---

## Repository structure

- **`src/`**
  - **`mc_distributions.py`**: Monte Carlo sampling helpers (normal, lognormal, GPD).
  - **`scale_up_model.py`**: Growth model for CADR production over time.
  - **`plotter.py`**: Plotting utilities for UV vs requirement ramp-up.
- **`scripts/`**
  - **`estimate_scaleup.ipynb`**: Walkthrough notebook showing the main analysis step by step.
- **`results/`**
  - Default output location for CSVs and figures.

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
mamba activate guv-scaleup-model
```

Using **conda**:

```bash
conda env create -f environment.yml
conda activate guv-scaleup-model
```

---

## How to run the model

You can run the model either via the **notebook** or via the **command-line entry point** (coming soon).

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

---

## License

This project is released as open access under the **Apache 2.0 License**.  
See the `LICENSE` file in this repository or the [Apache 2.0 overview](https://opensource.org/licenses/Apache-2.0) for details.
