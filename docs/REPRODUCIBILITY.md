# Data and analysis details

## Data layouts

Method index 0 denotes EDH and index 1 denotes EFS.

| File in `data/blocks` | Arrays and indexing |
|---|---|
| `*_demo.npz` | `radial100[block, method, radial_bin]` and `angular100[block, method, section, alpha_bin]`; 100-configuration means. Section indices 0 and 1 correspond to radii nearest 1.25 and 2.5, respectively, with gamma in the first angular cell. |
| `gb3d_*_moments100.npz` | `blocks[800, 2, 109, 5]`; moment order `1`, `(P2(x)+P2(y))/2`, `P2(ui.uj)`, `P2(x)P2(y)`, `P4(ui.uj)`. |
| `gb3d_targets100.npz` | `targets100[3,800]`; paired EFS-minus-EDH P2 residual over the shell [0.9,1.5], ordered by seeds 410131, 410132, and 410133. |
| `gb3d_coarse100.npz` | `features[3,800,2,20]`, `weak[3,800,8]`, and `angles[3,800,2,512]`; four radial shells containing five moments, two torque quantities, or 128 angular cells per shell. |
| `dimer2d_reference500.npz`, `gb2d_reference500.npz` | Two reference chains with forty 500-configuration blocks each, containing joint distributions and radial marginals. |
| `gb3d_reference500.npz` | `blocks[2,160,92,129]`; two reference chains, 160 blocks each. Channels 0–127 contain joint angular cells and channel 128 contains the radial marginal. |

`data/examples` contains one configuration and its reference estimator arrays for each model. Planar configurations have columns `(x, y, theta)`. Three-dimensional configurations have columns `(x, y, z, ux, uy, uz)`. Angles are in radians and orientation vectors have unit length. Planar reference estimator arrays use float32 precision.

`data/efficiency` contains per-repeat errors and efficiency summaries. `data/reported` contains numerical reference results. File-to-source mappings and SHA-256 hashes are recorded in `provenance/source_manifest.json`.

## Statistical calculations

The planar analysis uses the `nrmse_with_reference_uncertainty` column in the per-repeat CSV files. Each test estimate is compared with a block-bootstrap realization of its independent reference. The statistics script recomputes reference uncertainties, the 95th percentiles of the repeat errors, and their bootstrap confidence intervals.

The three-dimensional analysis calculates RMS errors and paired EFS/EDH ratios from per-repeat errors. Reference uncertainty is calculated from the 500-configuration reference blocks. The sample-count criterion combines the upper conditional RMS bound with the reference-error allowance and requires all larger tested counts to pass.

The endpoint calculations use a stratified studentized bootstrap. Secondary tests apply Holm corrections within the specified observable families. Block lengths, random seeds, and analysis settings are provided in `configs` and the corresponding data files. Statistical comparisons use relative tolerance `1e-9` and absolute tolerance `1e-11`.

## Simulation parameters

Run configurations are stored in `configs/runs`. The three model identifiers are `dimer2d`, `gb2d`, and `gb3d`. The simulation commands write coordinates to `snapshots.npy`, parameters to `parameters.json`, and observables to `observables.csv`.

The estimator interface uses the paper's potential parameters. Radial grids start at 0.55, with shell widths of 0.025 in two dimensions and 0.05 in three dimensions. The outer radius is 4.0 for dimers and 6.0 for Gay–Berne particles, subject to `b < L/2`.

Three-dimensional continuation carries the final positions, orientation vectors, and production move sizes into the next segment. The physical chain identifier is retained, while `--rng-seed` specifies the random stream for that segment.

```bash
python scripts/continue_gb3d.py --source runs/gb3d_410131_initial --out runs/gb3d_410131_extension --rng-seed 510131 --frames 10000
```

For statistical analysis, concatenate segment blocks in chronological order within each physical chain. Segment parameters and stream identifiers are recorded in the run configuration files.
