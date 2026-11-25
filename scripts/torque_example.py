"""Recover the first 100-frame torque block from planar GB coordinates."""
import numpy as np
from numba import njit
from pathlib import Path
from gb_pair import gb_pair as pair


@njit(cache=True)
def build_cells(pos, L, nc):
    head = np.full(nc*nc, -1, np.int64)
    nxt = np.full(len(pos), -1, np.int64)
    prev = np.full(len(pos), -1, np.int64)
    cells = np.empty(len(pos), np.int64)
    for i in range(len(pos)):
        c = int(pos[i,0]/L*nc) + nc*int(pos[i,1]/L*nc)
        cells[i] = c
        nxt[i] = head[c]
        if head[c] >= 0:
            prev[head[c]] = i
        head[c] = i
    return head,nxt,prev,cells

@njit
def neighbors(pos, L):
    nc = int(L / 6.)
    head, nxt, _, cells = build_cells(pos, L, nc)
    ii = np.empty(16 * len(pos), np.int64)
    jj = np.empty_like(ii)
    disp = np.empty((len(ii), 2))
    size = 0
    for i in range(len(pos)):
        cx, cy = cells[i] % nc, cells[i] // nc
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                j = head[(cx + ox) % nc + nc * ((cy + oy) % nc)]
                while j >= 0:
                    if j > i:
                        dx = pos[j, 0] - pos[i, 0]
                        dy = pos[j, 1] - pos[i, 1]
                        dx -= L * np.rint(dx / L)
                        dy -= L * np.rint(dy / L)
                        if dx * dx + dy * dy < 36.:
                            if size == len(ii):
                                raise ValueError("Neighbor list capacity exceeded")
                            ii[size], jj[size] = i, j
                            disp[size, 0], disp[size, 1] = dx, dy
                            size += 1
                    j = nxt[j]
    return ii[:size], jj[:size], disp[:size]


@njit
def mechanical(theta, pairs, kappa):
    ii, jj, dr = pairs
    axes = np.column_stack((np.cos(theta), np.sin(theta)))
    force = np.zeros((len(theta), 2))
    tau = np.zeros(len(theta))
    energy = 0.
    for p in range(len(ii)):
        i, j = ii[p], jj[p]
        v = pair(dr[p, 0], dr[p, 1], axes[i], axes[j], kappa)
        energy += v[0]
        force[i, 0] += v[1]
        force[i, 1] += v[2]
        force[j, 0] -= v[1]
        force[j, 1] -= v[2]
        tau[i] += v[3]
        tau[j] += v[4]
    return energy, force, tau


@njit
def accumulate(theta, pairs, tau, L, T, edges, use_torque):
    ii, jj, dr = pairs
    coeff = np.zeros(len(edges) - 1)
    counts = np.zeros_like(coeff)
    C = L * L / (len(theta) * (len(theta) - 1))
    for p in range(len(ii)):
        dx, dy = dr[p]
        r = np.sqrt(dx * dx + dy * dy)
        if edges[0] < r < edges[-1]:
            i, j = ii[p], jj[p]
            h = np.searchsorted(edges, r) - 1
            delta = 2 * (theta[i] - theta[j])
            counts[h] += 2 * np.cos(delta)
            if use_torque:
                # Two ordered contributions, eigenvalue 2^2 + (-2)^2.
                coeff[h] -= (tau[i] - tau[j]) * np.sin(delta) / (2 * T)
    norm = C / (np.pi * np.diff(edges**2))
    return counts * norm, coeff * norm



if __name__ == "__main__":
    data=Path(__file__).resolve().parents[1]/"data"
    with np.load(data/"gb_mode_410082.npz") as z:
        edges=z["edges"]; expected=z["mode"][0]
    with np.load(data/"coordinate_example.npz") as z:
        frames=z["frames"]; L=float(z["L"])
    direct=[]; shell=[]
    for frame in frames:
        pos=frame[:,:2].copy(); theta=frame[:,2].copy()
        pairs=neighbors(pos,L)
        _,_,tau=mechanical(theta,pairs,3.)
        dh,torque=accumulate(theta,pairs,tau,L,2.,edges,True)
        direct.append(dh);shell.append(torque)
    np.testing.assert_allclose(np.mean(direct,axis=0),expected[0],rtol=0,atol=1e-11)
    np.testing.assert_allclose(np.mean(shell,axis=0),expected[3],rtol=0,atol=1e-11)
    print("Verified count and torque profiles across all 218 shells.")
