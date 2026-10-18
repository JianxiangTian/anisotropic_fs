# Finite-domain force sampling for anisotropic particles

Code and data accompanying **Finite-domain force sampling of orientation-resolved pair correlations in two and three dimensions**, by Jianxiang Tian and Yuan Liu.

This package implements extended direct histograms (EDH) and finite-domain extended force sampling (EFS) for planar dimers, planar Gay–Berne particles, and three-dimensional Gay–Berne particles. It provides simulation and analysis tools, example configurations, and scripts for reproducing the paper figures and statistics.

## Installation

Use Python 3.11 or newer. Run the following commands from the `Code` directory.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Then install the package:

```bash
python -m pip install -e .
```

Dependencies are listed in `requirements.txt`. The tested Python 3.12 environment is recorded in `requirements-tested.txt`.

## Reproduce figures and statistics

```bash
python scripts/reproduce_figures.py
python scripts/reproduce_statistics.py
```

The figure script produces six main figures and three supplementary figures in EPS, PDF, and PNG formats in `outputs/figures`.

The statistics script calculates reference uncertainties, efficiency confidence intervals, sample-count thresholds, and the primary and secondary statistical tests from the supplied data. Results are checked against the reference values and saved in `outputs/statistics`, including `table_II.csv` and `reproduction_report.json`.

See the [figure and table index](docs/PAPER_MAP.md) for the input files associated with each result, and [data and analysis details](docs/REPRODUCIBILITY.md) for array layouts and statistical conventions.

## Analyze an example configuration

```bash
python scripts/analyze_example.py --model dimer2d
python scripts/analyze_example.py --model gb2d
python scripts/analyze_example.py --model gb3d
```

Each command evaluates the EDH and EFS estimators for the selected model and saves the arrays in `outputs/examples`. The estimator interface uses a dimer bond length of 0.5 and a Gay–Berne aspect ratio of 3, matching the paper.

## Run a simulation

```bash
python scripts/simulate.py --config configs/runs/dimer2d_310053.json --out runs/dimer
python scripts/analyze_trajectory.py --model dimer2d --run runs/dimer --out outputs/dimer_blocks
```

Choose a configuration from `configs/runs` and a new output directory. The trajectory analysis writes block-averaged EDH and EFS estimates. The block size defaults to 100 configurations and can be set with `--block-size`.

For three-dimensional chain continuation, use `scripts/continue_gb3d.py`. Command options are available through `--help`.

## Check the installation and data

```bash
python scripts/verify_integrity.py
python -m unittest discover -s tests -v
```

The integrity check verifies file hashes. The numerical tests cover example estimates, normalization identities, force and torque derivatives, the planar restriction, and the finite-shell kernel.

## Package contents

| Directory | Contents |
|---|---|
| `src/anisotropic_fs` | Pair interactions, samplers, estimators, and statistical routines |
| `scripts` | Simulation, analysis, figure generation, and verification commands |
| `configs` | Model parameters, run configurations, and analysis protocols |
| `data` | Example configurations, block averages, repeat errors, and reference results |
| `tests` | Numerical regression and identity checks |
| `docs` | Data formats and figure/table index |
| `provenance` | Source mappings, environment information, validation results, and file hashes |

## Citation

Please cite the associated paper when using this code and data. Software citation metadata are provided in `CITATION.cff`.
