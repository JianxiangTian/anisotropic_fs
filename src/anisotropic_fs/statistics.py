"""Endpoint statistics and stratified studentized bootstrap."""
import numpy as np
from scipy.stats import t as student

def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    ans = np.empty_like(p)
    ans[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return ans

def grouped(x, m):
    return x.reshape(len(x) // m, m, *x.shape[1:]).mean(1)

def stats(x):
    se = x.std(0, ddof=1) / np.sqrt(len(x))
    mean = x.mean(0)
    p = 2 * student.sf(abs(mean) / (se + 1e-30), len(x) - 1)
    return (mean, se, p)

def boot(x, rng_seed=810134):
    rng = np.random.default_rng(rng_seed)
    S, n = x.shape
    mean = x.mean()
    se = np.sqrt(np.sum(x.var(1, ddof=1) / n)) / S
    centered = x - x.mean(1)[:, None]
    null = []
    uncentered = []
    for start in range(0, 20000, 500):
        ix = rng.integers(n, size=(500, S, n))
        draw = np.stack([x[s][ix[:, s]] for s in range(S)], axis=1)
        bse = np.sqrt(np.sum(draw.var(2, ddof=1) / n, axis=1)) / S
        dc = np.stack([centered[s][ix[:, s]] for s in range(S)], axis=1)
        null.extend((dc.mean((1, 2)) / bse).tolist())
        uncentered.extend(((draw.mean((1, 2)) - mean) / bse).tolist())
    q = np.quantile(uncentered, [0.025, 0.975])
    ci = [float(mean - q[1] * se), float(mean - q[0] * se)]
    return dict(mean=float(mean), SE=float(se), p=float((1 + np.sum(abs(np.asarray(null)) >= abs(mean / se))) / 20001), CI95_bootstrap_t=ci, blocks_per_seed=n, within_margin=bool(ci[0] > -0.00075 and ci[1] < 0.00075))
