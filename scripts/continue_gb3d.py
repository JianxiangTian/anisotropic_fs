"""Continue an N=1024 GB3D chain in a new segment using an explicit random stream."""
from _common import ROOT
from anisotropic_fs.legacy.gb3d_mc import *
from pathlib import Path
import traceback
ROOT=Path.cwd()
def extend(source, destination, rng_seed, frames=10000):
    src = ROOT / source
    out = ROOT / destination
    assert json.loads((src / 'status.json').read_text())['phase'] == 'completed'
    p = json.loads((src / 'parameters.json').read_text())
    fr = np.load(src / 'snapshots.npy', mmap_mode='r')
    assert p['N'] == 1024 and p['rho'] == 0.05 and (p['T'] == 2) and (p['stride'] == 10)
    pos = np.array(fr[-1, :, :3])
    ori = np.array(fr[-1, :, 3:])
    L = p['L']
    N = p['N']
    obs = np.genfromtxt(src / 'observables.csv', names=True, delimiter=',', dtype=None, encoding='utf8')
    last = obs[-1]
    step = float(last['step'])
    turn = float(last['turn'])
    sweep = int(last['sweep'])
    offset = int(last['index']) + 1
    nc = max(1, int(L / 6))
    cells = build_cells(pos, L, nc)
    seed_numba(rng_seed)
    out.mkdir(parents=True, exist_ok=False)
    p.update(out=destination, frames=frames, equil=0, continuation_of=source, continuation_rng_seed=rng_seed, physical_seed=p['seed'], continuation_note='Same physical Markov chain; new RNG stream, not a new independent seed', pid=os.getpid(), first_global_index=offset)
    (out / 'parameters.json').write_text(json.dumps(p, indent=2))
    f = np.lib.format.open_memmap(out / 'snapshots.npy', mode='w+', dtype='float64', shape=(frames, N, 6))
    start = time.time()

    def status(phase, n, **kw):
        tmp = out / 'status.tmp'
        tmp.write_text(json.dumps(dict(phase=phase, frames_written=n, sweep=sweep, pid=os.getpid(), elapsed_seconds=time.time() - start, **kw)))
        tmp.replace(out / 'status.json')
    status('sampling', 0)
    try:
        with (out / 'observables.csv').open('x', buffering=1) as log:
            log.write('phase,index,sweep,energy_per_particle,S2,Qxx,Qyy,Qxy,Qxz,Qyz,acceptance,step,turn,seconds\n')
            for k in range(frames):
                ac = sweeps(pos, ori, L, 1 / p['T'], step, turn, p['stride'], nc, *cells)
                sweep += p['stride']
                f[k, :, :3] = pos
                f[k, :, 3:] = ori
                U, _, _ = forces(pos, ori, L)
                Q = 1.5 * ori.T @ ori / N - 0.5 * np.eye(3)
                S = np.linalg.eigvalsh(Q)[-1]
                if not np.isfinite(U) or not np.isfinite(pos).all() or np.max(abs(np.linalg.norm(ori, axis=1) - 1)) > 1e-10:
                    raise RuntimeError('Nonfinite or axis integrity failure')
                log.write(f'production,{k + offset},{sweep},{U / N},{S},{Q[0, 0]},{Q[1, 1]},{Q[0, 1]},{Q[0, 2]},{Q[1, 2]},{ac},{step},{turn},{time.time() - start}\n')
                if (k + 1) % 100 == 0 or k + 1 == frames:
                    f.flush()
                    status('completed' if k + 1 == frames else 'sampling', k + 1)
    except Exception:
        status('failed', int(json.loads((out / 'status.json').read_text())['frames_written']), error=traceback.format_exc())
        raise
    return time.time() - start
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--rng-seed',type=int,required=True);p.add_argument('--frames',type=int,default=10000);a=p.parse_args()
    extend(str(a.source.resolve()),str(a.out.resolve()),a.rng_seed,a.frames)
