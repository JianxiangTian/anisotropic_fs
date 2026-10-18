# Figure and table index

Run `python scripts/reproduce_figures.py` to generate the figures and `python scripts/reproduce_statistics.py` to calculate the statistical summaries.

| Paper item | Input files and calculation |
|---|---|
| Fig. 1 | Pair geometry and estimation workflow in `scripts/reproduce_figures.py` |
| Fig. 2 | `data/blocks/dimer2d_demo.npz` and `gb2d_demo.npz`; 100-configuration blocks grouped into 500-configuration blocks |
| Fig. 3 | `data/blocks/gb3d_*_moments100.npz`; radial and P2 moments averaged over the three chains |
| Fig. 4 | Planar JSON and CSV files in `data/efficiency`; percentile confidence bounds |
| Fig. 5 | `data/efficiency/gb3d.json`; RMS errors and confidence intervals |
| Fig. 6 | `data/blocks/gb3d_targets100.npz`; endpoint estimates for the 20,000-, 60,000-, and 80,000-configuration datasets |
| Fig. S1 | Analytic finite-domain kernel |
| Fig. S2 | Angular section from `data/blocks/gb2d_demo.npz` |
| Fig. S3 | Reference assignments in `data/efficiency/gb3d_rotations.json` |
| Tables I and S2 | `configs/datasets.json` and parameter files in `configs/runs` |
| Table II | Sample-count thresholds written to `outputs/statistics/table_II.csv` |
| Table S1 | Numerical identity checks in `data/reported` and `tests/test_numerics.py` |
| Table S3 | Planar reference blocks and `data/blocks/gb3d_reference500.npz` |
| Table S4 | Studentized bootstrap with 20,000 resamples of `data/blocks/gb3d_targets100.npz` |
| Sec. S10 | Angular moments, weak-torque quantities, and angular-cell tests from `data/blocks/gb3d_coarse100.npz` |
