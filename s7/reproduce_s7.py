"""Reproduce the finite-interval three-dimensional comparison in supplementary Sec. S7."""
import argparse
import csv
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

HERE = Path(__file__).resolve().parent
M = 10  # 100-frame blocks per window
JOINT_BINS_SHA256 = "e626a2065897f3dc18ac4e84d3bc2ce755a6c5780bed0e4135ff796d95f1ee0b"


def assemble_joint_bins(destination):
    """Restore the original NPZ from seven parts and check its SHA-256."""
    parent = HERE / "joint_bins_parts"
    digest = hashlib.sha256()
    with destination.open("wb") as restored:
        for index in range(1, 8):
            with (parent / f"S7_joint_bins.npz.part{index:02d}").open("rb") as part:
                for chunk in iter(lambda: part.read(1024**2), b""):
                    restored.write(chunk)
                    digest.update(chunk)
    if digest.hexdigest() != JOINT_BINS_SHA256:
        raise ValueError("SHA-256 mismatch for the restored S7_joint_bins.npz")


def reduce_bins(test, reference, edges, grouped):
    """Project errors onto shell means and evaluate the weighted squared norm."""
    error = test - reference
    weights = np.diff(edges**3)
    weights /= weights.sum()
    cell_means = np.einsum("h,tmhc->tmc", weights, error)
    if not grouped:
        norms = np.einsum("h,tmhc,tmhc->tm", weights, error, error) / error.shape[-1]
        return cell_means, norms
    # A window mean is a sum of ten vectors. Only lags 0 through 9 enter its norm.
    n = len(error)
    products = np.zeros((M, n, 2))
    for lag in range(M):
        limit = n - lag
        products[lag, :limit] = np.einsum(
            "h,tmhc,tmhc->tm", weights, error[:limit], error[lag:], optimize=True
        ) / error.shape[-1]
    return cell_means, products


def decompose(cell_means, squared_norms, grouped):
    if grouped:
        n = len(cell_means) - M + 1
        anchors = np.stack([cell_means[i:i+M].mean(axis=0) for i in range(n)])
        totals = np.zeros((n, 2))
        for lag in range(M):
            for offset in range(M-lag):
                totals += (1 if lag == 0 else 2) * squared_norms[lag, offset:offset+n]
        totals /= M**2
    else:
        anchors, totals = cell_means, squared_norms
    mean_error = (anchors**2).mean(axis=(0, 2))
    mse = totals.mean(axis=0)
    shape = mse - mean_error
    gain = mse[0] - mse[1]
    return dict(samples=len(totals), EDH_mse=mse[0], EFS_mse=mse[1],
                rms_ratio=np.sqrt(mse[1]/mse[0]), EDH_anchor_mse=mean_error[0],
                EFS_anchor_mse=mean_error[1], EDH_shape_mse=shape[0], EFS_shape_mse=shape[1],
                shape_share=(shape[0]-shape[1])/gain,
                anchor_share=(mean_error[0]-mean_error[1])/gain,
                max_paired_cell_mean_difference=np.abs(anchors[:,0]-anchors[:,1]).max())


def reproduce(bins_path, output):
    rows=[]
    with np.load(HERE/"s7_inputs.npz", allow_pickle=False) as data:
        for label in ("full", "contiguous", "dispersed"):
            means, norms = data[label+"_mean"], data[label+"_norm"]
            if bins_path is not None:
                with np.load(bins_path,allow_pickle=False) as bins:
                    tag="full" if label=="full" else "inside"
                    m, n = reduce_bins(bins[label],bins[tag+"_reference"],bins[tag+"_edges"],label=="full")
                np.testing.assert_allclose(m,means,atol=1e-13,rtol=1e-12)
                np.testing.assert_allclose(n,norms,atol=1e-13,rtol=1e-12)
                means,norms=m,n
            rows.append(dict(domain_design=label,**decompose(means,norms,label=="full")))
    with (HERE/"s7_expected.csv").open(encoding="utf8") as stream:
        expected=list(csv.DictReader(stream))
    for row, ref in zip(rows,expected,strict=True):
        assert row["domain_design"]==ref["domain_design"]
        for k,v in row.items():
            if k!="domain_design":
                np.testing.assert_allclose(v,float(ref[k]),rtol=1e-10,atol=2e-14)
        print(row["domain_design"],row["samples"],f"RMS ratio {row['rms_ratio']:.9f}")
    with output.open("w",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bins", type=Path, help="Use an existing S7_joint_bins.npz file")
    group.add_argument("--from-bins", action="store_true", help="Rebuild the NPZ from the bundled parts")
    parser.add_argument("--output", type=Path, default=Path("s7_reproduced.csv"))
    args = parser.parse_args()
    if args.from_bins:
        with TemporaryDirectory(prefix="s7_bins_") as temp:
            bins_path = Path(temp) / "S7_joint_bins.npz"
            assemble_joint_bins(bins_path)
            reproduce(bins_path, args.output)
    else:
        reproduce(args.bins, args.output)


if __name__=="__main__":
    main()

