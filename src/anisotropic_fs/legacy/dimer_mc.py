"""Planar apolar two-site LJ MC. Reduced units, smooth site cutoff.

Each molecule has sites at R +/- ell*n/2. The molecular pair energy is
one quarter of the four site energies. Total molecular forces and torques
are analytic derivatives; MC uses energy only. No production adaptation.
"""
import sys
from pathlib import Path
ROOT = Path.cwd()
import numpy as np
from numba import njit
import json, time, argparse, os

@njit(cache=True)
def site(q):
    if q >= 6.25:
        return 0., 0.
    if q < 1.e-12:
        return 1.e80, -1.e90
    z = 1./q
    z3 = z*z*z
    u = 4.*(z3*z3-z3)
    # du/dr divided by r
    du = 24.*(z3-2.*z3*z3)*z
    if q > 4.:
        r = np.sqrt(q)
        t = (r-2.)/.5
        s = 1.-10.*t**3+15.*t**4-6.*t**5
        ds = (-30.*t*t+60.*t**3-30.*t**4)/.5
        du = du*s + u*ds/r
        u *= s
    return u, du

@njit(cache=True)
def pair(dx, dy, ni, nj, ell):
    if dx*dx+dy*dy >= (2.5+ell)**2:
        return np.zeros(5)
    out = np.zeros(5)  # U, Fx_on_i, Fy_on_i, torque_i, torque_j
    for sa in (-1., 1.):
        ax = sa*.5*ell*ni[0]
        ay = sa*.5*ell*ni[1]
        for sb in (-1., 1.):
            bx = sb*.5*ell*nj[0]
            by = sb*.5*ell*nj[1]
            x, y = dx+bx-ax, dy+by-ay
            u, du = site(x*x+y*y)
            fx, fy = .25*du*x, .25*du*y
            out[0] += .25*u
            out[1] += fx
            out[2] += fy
            out[3] += ax*fy-ay*fx
            out[4] += -bx*fy+by*fx
    return out

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

@njit(cache=True)
def energy_one(i, x, y, n, pos, orient, L, nc, head, nxt, ell):
    cx, cy = int(x/L*nc), int(y/L*nc)
    e = 0.
    # nc >= 3 avoids duplicate neighbors.
    for ox in (-1,0,1):
        for oy in (-1,0,1):
            j = head[(cx+ox)%nc + nc*((cy+oy)%nc)]
            while j >= 0:
                if j != i:
                    dx, dy = pos[j,0]-x, pos[j,1]-y
                    dx -= L*np.rint(dx/L)
                    dy -= L*np.rint(dy/L)
                    e += pair(dx,dy,n,orient[j],ell)[0]
                j = nxt[j]
    return e

@njit(cache=True)
def sweeps(pos, theta, orient, L, ell, beta, step, turn, count,
           nc, head, nxt, prev, cells):
    accept = 0
    N = len(pos)
    for sweep in range(count):
        for trial in range(N):
            i = np.random.randint(N)
            x = (pos[i,0]+step*(2*np.random.random()-1))%L
            y = (pos[i,1]+step*(2*np.random.random()-1))%L
            angle = (theta[i]+turn*(2*np.random.random()-1))%np.pi
            n = np.array([np.cos(angle),np.sin(angle)])
            old = energy_one(i,pos[i,0],pos[i,1],orient[i],pos,orient,L,nc,head,nxt,ell)
            new = energy_one(i,x,y,n,pos,orient,L,nc,head,nxt,ell)
            if new <= old or np.random.random() < np.exp(-beta*(new-old)):
                accept += 1
                cnew = int(x/L*nc)+nc*int(y/L*nc)
                cold = cells[i]
                if cnew != cold:
                    if prev[i] >= 0:
                        nxt[prev[i]] = nxt[i]
                    else:
                        head[cold] = nxt[i]
                    if nxt[i] >= 0:
                        prev[nxt[i]] = prev[i]
                    prev[i] = -1
                    nxt[i] = head[cnew]
                    if head[cnew] >= 0:
                        prev[head[cnew]] = i
                    head[cnew] = i
                    cells[i] = cnew
                pos[i,0],pos[i,1] = x,y
                theta[i] = angle
                orient[i] = n
    return accept/(count*N)

@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)

@njit(cache=True)
def forces(pos, orient, L, ell):
    N = len(pos)
    f = np.zeros((N,2))
    tau = np.zeros(N)
    U = 0.
    for i in range(N):
        for j in range(i+1,N):
            dx,dy = pos[j,0]-pos[i,0],pos[j,1]-pos[i,1]
            dx -= L*np.rint(dx/L)
            dy -= L*np.rint(dy/L)
            if dx*dx+dy*dy >= (2.5+ell)**2:
                continue
            v = pair(dx,dy,orient[i],orient[j],ell)
            U += v[0]
            f[i,0] += v[1]; f[i,1] += v[2]
            f[j,0] -= v[1]; f[j,1] -= v[2]
            tau[i] += v[3]; tau[j] += v[4]
    return U,f,tau

def run(args):
    out = ROOT / args.out
    out.mkdir(parents=True,exist_ok=True)
    if (out/'parameters.json').exists():
        raise RuntimeError('Refusing to overwrite an existing run')
    np.random.seed(args.seed)
    seed_numba(args.seed)
    L = np.sqrt(args.N/args.rho)
    side = int(np.ceil(np.sqrt(args.N)))
    pos = np.array([[(i%side+.5)*L/side,(i//side+.5)*L/side] for i in range(args.N)])
    theta = np.random.uniform(0,np.pi,args.N)
    orient = np.column_stack([np.cos(theta),np.sin(theta)])
    nc = int(L/(2.5+args.ell))
    if nc < 3:
        raise ValueError('Box too small for this cell-list implementation')
    head,nxt,prev,cells = build_cells(pos,L,nc)
    parameters = vars(args).copy()
    parameters.update(L=L,site_switch_on=2.,site_cutoff=2.5,pair_site_prefactor=.25,
                      units='sigma=epsilon=kB=1',pid=os.getpid(),version='v1')
    (out/'parameters.json').write_text(json.dumps(parameters,indent=2),encoding='utf-8')
    frames = np.lib.format.open_memmap(out/'snapshots.npy',mode='w+',dtype='float64',shape=(args.frames,args.N,3))
    start = time.time()
    step,turn = args.step,args.turn
    log = open(out/'observables.csv','w',buffering=1)
    log.write('phase,index,sweep,energy_per_particle,Qcos,Qsin,acceptance,step,turn,seconds\n')
    sweep=0
    def status(phase,frame):
        (out/'status.json').write_text(json.dumps(dict(phase=phase,frames_written=frame,
            sweep=sweep,elapsed_seconds=time.time()-start,pid=os.getpid())),encoding='utf-8')
    for k in range(0,args.equil,100):
        count=min(100,args.equil-k)
        ac=sweeps(pos,theta,orient,L,args.ell,1/args.T,step,turn,count,nc,head,nxt,prev,cells)
        sweep+=count
        U,f,tau=forces(pos,orient,L,args.ell)
        log.write(f'equil,{k//100},{sweep},{U/args.N},{np.cos(2*theta).mean()},{np.sin(2*theta).mean()},{ac},{step},{turn},{time.time()-start}\n')
        # Adapt only first half of discarded equilibration.
        if k < args.equil//2:
            scale=np.exp(.5*(ac-.4))
            step=min(.7,max(.03,step*scale))
            turn=min(np.pi,max(.05,turn*scale))
        status('equilibration',0)
    for k in range(args.frames):
        ac=sweeps(pos,theta,orient,L,args.ell,1/args.T,step,turn,args.stride,nc,head,nxt,prev,cells)
        sweep+=args.stride
        frames[k,:,:2]=pos
        frames[k,:,2]=theta
        U,f,tau=forces(pos,orient,L,args.ell)
        log.write(f'production,{k},{sweep},{U/args.N},{np.cos(2*theta).mean()},{np.sin(2*theta).mean()},{ac},{step},{turn},{time.time()-start}\n')
        if (k+1)%100==0:
            frames.flush()
            status('sampling',k+1)
            np.savez(out/'checkpoint.npz',pos=pos,theta=theta,sweep=sweep)
    frames.flush()
    status('completed',args.frames)
    log.close()
    print(json.dumps(dict(out=str(out),seconds=time.time()-start,frames=args.frames)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--N',type=int,default=1024)
    p.add_argument('--rho',type=float,default=.3)
    p.add_argument('--T',type=float,default=2.)
    p.add_argument('--ell',type=float,default=.5)
    p.add_argument('--equil',type=int,default=5000)
    p.add_argument('--frames',type=int,default=10000)
    p.add_argument('--stride',type=int,default=10)
    p.add_argument('--seed',type=int,default=20261005)
    p.add_argument('--step',type=float,default=.25)
    p.add_argument('--turn',type=float,default=.7)
    p.add_argument('--out',required=True)
    run(p.parse_args())
