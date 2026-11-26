# anisotropic_fs

Data and code for *Resolution dependence of force sampling for orientation-resolved pair correlations in anisotropic fluids* by Jianxiang Tian and Yuan Liu.

## Reproduce the paper

From the repository root, install the requirements in a Python environment and run the following commands.

```sh
python -m pip install -r requirements.txt
python reproduce.py all
```

The code was tested on Windows with Python 3.14, NumPy 2.4.4, Matplotlib 3.10.9 and Numba 0.68.1. The plotting scripts check that Times New Roman is installed. All inputs for these commands are included. Allow several minutes and about 2 GB of available memory. The main analysis uses one OpenBLAS thread.

| Command | Outputs |
| --- | --- |
| `python reproduce.py figures` | Four main and three supplementary figures under `outputs/figures`, and CSV values for Tables I and S1–S6 under `outputs/paper_tables`. |
| `python reproduce.py statistics` | Recompute the planar resolution, alignment, covariance, radial-region and sensitivity analyses; compare every column of seven CSV tables with the archived outputs. |
| `python reproduce.py torque` | Recompute the first 100-frame test block from planar GB coordinates and verify both direct and torque-weighted profiles for all 218 radial shells. |
| `python reproduce.py s7` | Reproduce the full-interval and interior error decompositions in `outputs/tables/s7_reproduced.csv`. |
| `python reproduce.py all` | Run all four operations. |

## Data and figure map

- **Fig. 1.** Pair geometry and the exact planar shell-averaged force kernel at a specified source separation. Drawn directly by `scripts/figures.py`.
- **Fig. 2.** Within-cell radial shape and angular-mean errors at matched resolution. The input table `results/joint_errors.csv` is reproduced by `scripts/analyse.py`.
- **Fig. 3.** Direct counts, radial force with either anchor, and shellwise torque for the planar alignment mode. The two test chains have 10,000 frames each and the two independent reference chains have 20,000 each. The error and profile tables contain paired block-bootstrap results.
- **Fig. 4.** Three-dimensional radial and alignment profiles from 500-frame block means and the radial and joint EFS/EDH ratios from the repeated selections in `gb3d_eff.json`.
- **Fig. S1.** The planar profile files contain 500-frame block means from the demonstration runs, exact shell edges and the selected angular-section index. The GB center is 2.4875.
- **Figs. S2 and S3.** Archived error summaries for dimers, planar GB and three-dimensional GB. These and the associated thresholds supply Table I.
- **Sec. S7.** The three-dimensional full-interval and interior RMS ratios and the contributions of the angular-cell mean and radial shape to the squared-error reduction. See the [S7 instructions and input description](s7/README.md) for calculation from both the compact data and the original shell and angular bins.
- **Tables S3–S6.** Computed from the supplied error tables. Section-wise contributions to the full-annulus MSE difference include the appropriate fraction of the full annular area.
- **Timing.** The measured per-frame stage times for the paper's coordinate-based benchmark are in `data/timing_grid.json`. Absolute timings depend on hardware.

## Statistical inputs

For each planar seed, the EDH and EFS files contain 100-frame block means with radial shell as the second axis and the two orientation indices as the last axes. Each file holds at most 20 blocks, ordered by the numeric suffix, which gives the zero-based starting block. The reference profiles use direct angular counts. The planar GB mode files store block means of the real Fourier mode (1, −1), ordered as direct counts, count-anchored radial force, torque-anchored radial force, and shellwise torque for the test seeds. The coordinate example contains the first 100 retained frames of seed 410082. The planar statistics are recomputed from the supplied block means. The [simulation code and trajectory protocols](https://github.com/JianxiangTian/anisotropic_fs/tree/0ad2402) document how the blocks were generated.

The planar windows contain contiguous 100-frame blocks, and the test seeds are evaluated separately. Reference chains are combined with equal weights. The statistical scripts specify the resampling designs, random seeds, tested widths and intervals. The package recomputes the count thresholds from the supplied confidence bounds. The timing and three-dimensional endpoint files contain the reported measurements. `SHA256SUMS` lists the digests of the distributed files.

