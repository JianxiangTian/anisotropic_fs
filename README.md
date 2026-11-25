# Resolution dependence of force sampling for orientation-resolved pair correlations in anisotropic fluids

Jianxiang Tian and Yuan Liu. Reproducibility package for the revised manuscript, version 3, 25 November 2026.

## Reproduce the paper

Extract the archive into a writable directory. From that directory, install the requirements in a Python environment and run:

```sh
python -m pip install -r requirements.txt
python reproduce.py all
```

The release was tested on Windows with Python 3.14, NumPy 2.4.4, Matplotlib 3.10.9 and Numba 0.68.1. Install Times New Roman locally for manuscript-matched rendering. The font is checked at plot startup. No external data downloads or model fitting are needed. Allow several minutes and about 2 GB of available memory; NumPy is run with one OpenBLAS thread for the main analysis.

| Command | Outputs |
| --- | --- |
| `python reproduce.py figures` | Four main and three supplementary figures under `outputs/figures`, and CSV values for Tables I and S1–S6 under `outputs/paper_tables`. |
| `python reproduce.py statistics` | Recompute the planar resolution, alignment, covariance, radial-region and sensitivity analyses; compare every column of seven CSV tables with the archived outputs. |
| `python reproduce.py torque` | Recompute the first 100-frame test block from planar GB coordinates and verify both direct and torque-weighted profiles for all 218 radial shells. |
| `python reproduce.py all` | Run all three operations. |

## Data and figure map

- **Fig. 1.** Pair geometry and the exact planar shell-averaged force kernel at a specified source separation. Drawn directly by `scripts/figures.py`.
- **Fig. 2.** Within-cell radial shape and angular-mean errors at matched resolution. The input table `results/joint_errors.csv` is reproduced by `scripts/analyse.py`.
- **Fig. 3.** Direct counts, radial force with either anchor, and shellwise torque for the planar alignment mode. The two test chains have 10,000 frames each and the two independent reference chains have 20,000 each. The error and profile tables contain paired block-bootstrap results.
- **Fig. 4.** Three-dimensional radial and alignment profiles from 500-frame block means and the radial and joint EFS/EDH ratios from the repeated selections in `gb3d_eff.json`.
- **Fig. S1.** The planar profile files contain 500-frame block means from the demonstration runs, exact shell edges and the selected angular-section index. The GB center is 2.4875.
- **Figs. S2 and S3.** Archived error summaries for dimers, planar GB and three-dimensional GB. These and the associated thresholds supply Table I.
- **Tables S3–S6.** Computed from the supplied error tables. Section-wise contributions to the full-annulus MSE difference include the appropriate fraction of the full annular area.
- **Timing.** The measured per-frame stage times for the paper's coordinate-based benchmark are in `data/timing_grid.json`. Absolute timings depend on hardware.

## Statistical inputs

For each planar seed, the edh and efs files contain 100-frame block means with radial shell as the second axis and the two orientation indices as the last axes. Each file holds at most 20 blocks, ordered by the numeric suffix (starting block, zero-based). Only direct angular counts enter the reference profiles. The planar GB mode files store the actual real Fourier mode (1, −1) block means in direct, count-anchor radial force, torque-anchor radial force, and (for test seeds) shellwise torque order. The coordinate example is the first 100 retained frames of seed 410082. The complete planar statistics restart from the distributed block means. Full MC trajectory generation and production-scale torque accumulation are upstream operations. The established model implementations and original trajectory protocols are also available in the [model repository at the baseline revision](https://github.com/JianxiangTian/anisotropic_fs/tree/0ad2402).

Each window contains contiguous 100-frame blocks; test seeds are evaluated separately. Reference chains are combined with equal weights. The statistical scripts specify the resampling designs, random seeds, tested widths and intervals. The original sample-count and three-dimensional confidence bounds are distributed as evaluated summaries. The release recomputes the displayed count thresholds from them. The timing and three-dimensional endpoint measurements are archived results. SHA256SUMS identifies the release inputs and scripts. Figure binaries can have different metadata across runs; statistical results are checked numerically.

