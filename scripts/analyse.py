"""Paired analysis of planar Gay–Berne pair-structure estimators."""
import csv
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = (410082,410093,410091,410092)
TESTS = (410082, 410093)
REFS = (410091, 410092)
EDGES = np.linspace(0.55, 6, 219)
WEIGHTS = np.diff(EDGES**2)
WIDTHS = (500, 1000, 2000)
BLOCKS = (500, 1000)
SEED = 4839001
REPEATS = 2000
METHODS = ("EDH", "EFS", "Torque_anchor+EFS", "Shell_torque")


def interval(draws):
    return np.quantile(draws, [0.025, 0.975], axis=0)


def windows(a, n):
    prefix = np.concatenate([np.zeros_like(a[:1]), np.cumsum(a, axis=0)])
    return (prefix[n:] - prefix[:-n]) / n


def merge(a, n):
    return a.reshape(len(a) // n, n, *a.shape[1:]).mean(axis=1)


def angular_grid(a, n, stride):
    m = 12 // n
    w = WEIGHTS.reshape(-1, stride)
    a = a.reshape(*a.shape[:-3], len(w), stride, n, m, n, m)
    a = a.mean(axis=(-1, -3))
    return np.einsum("...rsij,rs->...rij", a, w / w.sum(axis=1, keepdims=True))


def boot_design(seed, n, m, block):
    # Adjacent starts share m-1 underlying blocks. Resample contiguous
    # groups of starts long enough to cover overlap and residual dependence.
    rng = np.random.default_rng(seed)
    length = m + block - 1
    starts = rng.integers(n, size=(REPEATS, int(np.ceil(n / length)), 1))
    test_ix = ((starts + np.arange(length)) % n).reshape(REPEATS, -1)[:, :n]
    draws = rng.integers(200 // block, size=(2, REPEATS, 200 // block))
    return test_ix, draws


def projected_error(tests, refs, weights, m, block, key):
    """RMS over starting positions and normalized integration weights."""
    w = np.sqrt(weights / weights.sum())
    t = (windows(tests, m) * w).reshape(len(tests) - m + 1, tests.shape[1], -1)
    refs = [(r * w).reshape(len(r), -1) for r in refs]
    ref = np.mean([r.mean(axis=0) for r in refs], axis=0)
    ref_gap = float(np.linalg.norm(refs[0].mean(axis=0) - refs[1].mean(axis=0)))
    t -= ref
    # Work in the span of the reference blocks so resampling needs a small
    # Gram matrix rather than thousands of joint-distribution reconstructions.
    b = np.concatenate([merge(r, block) - ref for r in refs])
    ix, ref_ix = boot_design(key, len(t), m, block)
    nb = len(b) // 2
    counts = np.concatenate(
        [np.eye(nb)[index].mean(axis=1) / 2 for index in ref_ix], axis=1
    )
    gram = b @ b.T
    ref_sq = np.einsum("ij,ij->i", counts @ gram, counts)
    t_sq = np.einsum("nmd,nmd->nm", t, t)
    cross = (counts @ (b @ t.reshape(-1, t.shape[-1]).T)).reshape(
        REPEATS, len(t), t.shape[1]
    )
    sq = t_sq[None, :, :] + ref_sq[:, None, None] - 2 * cross
    boot = np.sqrt(np.maximum(sq[np.arange(REPEATS)[:, None], ix].mean(axis=1), 0))
    return np.sqrt(t_sq.mean(axis=0)), boot, float(np.sqrt(np.quantile(ref_sq, .95))), ref_gap


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def blocks(tag):
    paths = sorted(DATA.glob(tag + "_*.npz"))
    if not paths:
        raise FileNotFoundError(tag)
    arrays = []
    for path in paths:
        with np.load(path) as z:
            arrays.append(z["values"])
    return np.concatenate(arrays)

def load_data():
    data = {}
    for seed in SOURCES:
        arrays = [blocks(f"gb_edh_{seed}")]
        if seed in TESTS:
            arrays.append(blocks(f"gb_efs_{seed}"))
        with np.load(DATA/f"gb_mode_{seed}.npz") as z:
            np.testing.assert_allclose(EDGES,z["edges"],rtol=0,atol=1e-12)
            data[seed] = np.stack(arrays, axis=1), z["mode"]
    return data, list(SOURCES)

def loss_rows(data):
    rows = []
    for stride in (1, 2):
        edges = EDGES[::stride]
        w = np.diff(edges**2)
        mask = (edges[:-1] >= .9 - 1e-10) & (edges[1:] <= 5.5 + 1e-10)
        for n in (1, 3, 6, 12):
            joint = {s: angular_grid(data[s][0], n, stride) for s in SOURCES}
            anchor = {s: np.einsum("bmrij,r->bmij", a, w / w.sum()) for s, a in joint.items()}
            shapes = {s: joint[s] - a[:, :, None] for s, a in anchor.items()}
            for seed in TESTS:
                projections = (
                    ("anchor", anchor, np.ones((n, n))),
                    ("shape", shapes, w[:, None, None] * np.ones((n, n))),
                    ("full", joint, w[:, None, None] * np.ones((n, n))),
                    ("interior", {s: a[:, :, mask] for s, a in joint.items()},
                     w[mask, None, None] * np.ones((n, n))),
                )
                for m in WIDTHS:
                    for block in BLOCKS:
                        components = {}
                        for label, values, weights in projections:
                            point, draws, ref95, ref_gap = projected_error(
                                values[seed], [values[s][:, 0] for s in REFS],
                                weights, m // 100, block // 100, SEED + seed + m + block,
                            )
                            components[label] = (point, draws)
                            ratio = draws[:, 1] / draws[:, 0]
                            delta = draws[:, 1] - draws[:, 0]
                            lo, hi = interval(ratio)
                            dlo, dhi = interval(delta)
                            dh_ci, fs_ci = interval(draws).T
                            rows.append(dict(seed=seed, M=m, bootstrap_block=block, bins=n,
                                             dr=round(.025 * stride, 3), quantity=label,
                                             EDH=point[0], EFS=point[1], ratio=point[1]/point[0],
                                             ratio_lo=lo, ratio_hi=hi, ref_q95=ref95, ref_gap=ref_gap,
                                             EDH_lo=dh_ci[0], EDH_hi=dh_ci[1],
                                             EFS_lo=fs_ci[0], EFS_hi=fs_ci[1],
                                             delta=point[1]-point[0], delta_lo=dlo, delta_hi=dhi))
                        for idx in (0, 1):
                            np.testing.assert_allclose(
                                components["full"][idx]**2,
                                components["anchor"][idx]**2 + components["shape"][idx]**2,
                                rtol=1e-8, atol=1e-13,
                            )
    return rows


def mode_rows(data):
    modes = {s: data[s][1] for s in SOURCES}
    keep = (EDGES[:-1] >= .9 - 1e-10) & (EDGES[1:] <= 5.5 + 1e-10)
    weights = WEIGHTS[keep]
    rows, decomposition, profiles = [], [], []
    for seed in TESTS:
        for block in BLOCKS:
            references = [merge(modes[s][:, 0], block // 100) for s in REFS]
            test = merge(modes[seed], block // 100)
            ref = np.mean([r.mean(axis=0) for r in references], axis=0)
            rng = np.random.default_rng(SEED + seed + block)
            tboot = test[rng.integers(len(test), size=(REPEATS, len(test)))].mean(axis=1)
            rboot = np.mean([r[rng.integers(len(r), size=(REPEATS, len(r)))].mean(axis=1)
                             for r in references], axis=0)
            diff_ci = interval(tboot - rboot[:, None])
            ref_ci = interval(rboot)
            for j, name in enumerate(METHODS[:test.shape[1]]):
                for h in range(len(WEIGHTS)):
                    profiles.append(dict(seed=seed, block=block, method=name,
                                         left=EDGES[h], right=EDGES[h+1], reference=ref[h],
                                         reference_lo=ref_ci[0,h], reference_hi=ref_ci[1,h],
                                         test=test[:,j,h].mean(), difference=test[:,j,h].mean()-ref[h],
                                         difference_lo=diff_ci[0,j,h], difference_hi=diff_ci[1,j,h]))
            for m in WIDTHS:
                g = windows(modes[seed], m // 100)
                anchor = g @ (WEIGHTS / WEIGHTS.sum())
                correction = g[:, 1, :] - anchor[:, 1, None]
                mean_correction = correction[:, keep] @ (weights / weights.sum())
                for j in (1, 2):
                    decomposition.append(dict(
                        seed=seed, M=m, method=METHODS[j], anchor_mean=anchor[:,j].mean(),
                        reference_anchor=ref @ (WEIGHTS / WEIGHTS.sum()),
                        anchor_variance=np.var(anchor[:,j], ddof=1),
                        correction_mean_variance=np.var(mean_correction, ddof=1),
                        covariance=np.cov(anchor[:,j], mean_correction)[0,1],
                    ))
                point, draws, ref95, ref_gap = projected_error(
                    modes[seed][:, :, keep], [modes[s][:,0,keep] for s in REFS],
                    weights, m // 100, block // 100, SEED + seed + m + block,
                )
                for j, name in enumerate(METHODS[:len(point)]):
                    lo, hi = interval(draws[:,j])
                    dlo, dhi = interval(draws[:,j] - draws[:,0])
                    flo, fhi = interval(draws[:,j] - draws[:,1])
                    bias = np.sqrt(np.average((test[:,j].mean(axis=0)[keep]-ref[keep])**2,
                                              weights=weights))
                    rows.append(dict(seed=seed, M=m, bootstrap_block=block, method=name,
                                     RMS=point[j], lo=lo, hi=hi, delta_EDH=point[j]-point[0],
                                     delta_EDH_lo=dlo, delta_EDH_hi=dhi,
                                     delta_EFS=point[j]-point[1], delta_EFS_lo=flo, delta_EFS_hi=fhi,
                                     reference_q95=ref95, reference_gap=ref_gap, full_chain_RMS=bias))
    return rows, decomposition, profiles


def main():
    start = time.perf_counter()
    data, audit = load_data()
    load_seconds = time.perf_counter() - start
    write_csv("joint_errors.csv", loss_rows(data))
    rows, decomposition, profiles = mode_rows(data)
    write_csv("mode_errors.csv", rows)
    unique = {(r["seed"], r["M"], r["method"]): r for r in decomposition}
    write_csv("mode_covariance.csv", list(unique.values()))
    write_csv("mode_profiles.csv", profiles)
    provenance = dict(seed=SEED, bootstrap_replicates=REPEATS, widths=WIDTHS, blocks=BLOCKS,
                      sources=audit, archive_load_seconds=load_seconds,
                      analysis_seconds=time.perf_counter()-start,
                      numpy_version=np.__version__,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / "analysis_audit.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
