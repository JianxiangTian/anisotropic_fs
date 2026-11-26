# Three-dimensional error decomposition in Sec. S7

These files are part of [anisotropic_fs](../README.md) and reproduce the full-interval and interior error decompositions in Sec. S7 of the supplementary material.

## Run

From the repository root, with NumPy installed, run:

```sh
python reproduce.py s7
```

The script computes the three RMS ratios and the contributions of the angular-cell mean and radial shape to the reduction in mean squared error. It checks the results against independent direct summation and writes `outputs/tables/s7_reproduced.csv`. The compact calculation uses `s7_inputs.npz`.

To recompute the input statistics from the original radial and angular bins, run:

```sh
python s7/reproduce_s7.py --from-bins
```

This command writes `s7_reproduced.csv` in the current directory. It joins the seven files in `joint_bins_parts/` in order and checks the SHA-256 digest of the restored NPZ. It then calculates the weighted means and products and compares them with the compact inputs before reproducing the table. Allow about 132 MB of temporary disk space. The temporary NPZ is removed when the command finishes. If you already have the original NPZ, you can also pass its path with `--bins`.

## Observations and weights

All angular channels refer to the same 4 × 4 × 8 equal-measure grid (128 cells). Chain 410133 is the test. The full-domain reference is the equally weighted EDH mean of chains 410131 and 410132, each with 80,000 retained configurations. The two test estimators are ordered EDH, EFS.

- **Full interval [0.55, 6].** 800 successive 100-configuration blocks of the test chain supply 791 windows of ten consecutive blocks each. The full reference is constructed on the same 109 shells.
- **Interior [0.9, 5.5].** The 200 contiguous selections and 200 dispersed selections of 1000 frames are those in Fig. 4(d). Their 92-shell means are paired with the archived reference used for Fig. 4. These two designs use the selected frames directly. Their averaging protocol differs from the full-interval block-boundary windows.

Shell weights are proportional to differences of the cubed bin edges and sum to one on the selected interval. Each angular cell has weight 1/128. Ratios are square roots of the mean squared errors after averaging over all bins and samples for each design. The shape and angular-mean shares are ratios of MSE differences, with positive sign for an EFS improvement.

## Exact compact inputs

For each test block or interior selection, `*_mean` contains the shell-weighted mean of its error relative to the independent reference in each angular cell. The interior `*_norm` arrays contain the shell- and angle-weighted squared norm of each selection's error. For the full interval, `full_norm` contains weighted inner products between each block error and the block errors at lags zero through nine. Values beyond the last valid block pair are unused. The squared norm of a ten-block mean is the sum of its ten diagonal terms plus twice the sum of its 45 distinct off-diagonal terms, divided by 100. These inputs retain all terms needed for the stated windows. Subtracting the MSE of the angular means from the total MSE gives the radial shape contribution.

The optional bin-resolved input has arrays `full` (800 × 2 × 109 × 128), `contiguous` and `dispersed` (each 200 × 2 × 92 × 128), their two EDH reference profiles, and the radial edges. These are estimator averages from the original simulations. The expected table was computed independently by constructing each window profile explicitly and squaring its radial residual after subtraction of the angular-cell mean.

## Interpretation

The full-interval ratio rounds to 0.717. The two anchors coincide up to floating-point error and the entire MSE reduction is in the radial shape. The interior ratios round to 0.675 and 0.682. The mean-error contribution increases slightly, while the shape improvements amount to 102.3% and 102.4% of the respective net improvements. The machine-precision residual can vary with the arithmetic reduction order.

