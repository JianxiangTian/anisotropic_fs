"""Local 3D uniaxial GB NVT MC, symmetric translation and SO(3) proposals."""
from .gb3d_pair import *
from .dimer_mc import ROOT,json,time,os,argparse

@njit(cache=True)
def energy_pair(d,u,v):
    r=np.sqrt(np.dot(d,d))
    if r>=6:return 0.
    if r<1e-10:return 1e80
    e=d/r;sig,_,_,_,eps,_,_,_=geometry(np.dot(u,e),np.dot(v,e),np.dot(u,v),3.,5.,2.,1.)
    s=r-sig+1
    if s<=1e-8:return 1e80
    z6=s**-6;value=4*eps*(z6*z6-z6)
    if r>4.5:
        t=(r-4.5)/1.5;value*=1-10*t**3+15*t**4-6*t**5
    return value

@njit(cache=True)
def cell_index(x,L,nc):
    return int(x[0]/L*nc)+nc*(int(x[1]/L*nc)+nc*int(x[2]/L*nc))
@njit(cache=True)
def build_cells(pos,L,nc):
    head=np.full(nc**3,-1,np.int64);nxt=np.full(len(pos),-1,np.int64);prev=nxt.copy();cells=nxt.copy()
    for i in range(len(pos)):
        c=cell_index(pos[i],L,nc);cells[i]=c;nxt[i]=head[c]
        if head[c]>=0:prev[head[c]]=i
        head[c]=i
    return head,nxt,prev,cells
@njit(cache=True)
def energy_one(i,x,u,pos,orient,L,nc,head,nxt):
    total=0.
    if nc<3:
        for j in range(len(pos)):
            if j!=i:
                d=pos[j]-x;d-=L*np.rint(d/L);total+=energy_pair(d,u,orient[j])
        return total
    c=(x/L*nc).astype(np.int64)
    for ox in (-1,0,1):
      for oy in (-1,0,1):
       for oz in (-1,0,1):
        idx=(c[0]+ox)%nc+nc*((c[1]+oy)%nc+nc*((c[2]+oz)%nc));j=head[idx]
        while j>=0:
            if j!=i:
                d=pos[j]-x;d-=L*np.rint(d/L);total+=energy_pair(d,u,orient[j])
            j=nxt[j]
    return total
@njit(cache=True)
def propose(u,turn):
    z=2*np.random.random()-1;phi=2*np.pi*np.random.random();axis=np.array([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z])
    angle=turn*(2*np.random.random()-1)
    v=u*np.cos(angle)+np.cross(axis,u)*np.sin(angle)+axis*np.dot(axis,u)*(1-np.cos(angle))
    # Exact head-tail symmetry move; makes the S2 double cover explicitly sampled.
    if np.random.random()<.5:v=-v
    return v/np.sqrt(np.dot(v,v))
@njit(cache=True)
def sweeps(pos,orient,L,beta,step,turn,count,nc,head,nxt,prev,cells):
    accepted=0
    for sweep in range(count):
      for trial in range(len(pos)):
        i=np.random.randint(len(pos));x=(pos[i]+step*(2*np.random.random(3)-1))%L;u=propose(orient[i],turn)
        old=energy_one(i,pos[i],orient[i],pos,orient,L,nc,head,nxt);new=energy_one(i,x,u,pos,orient,L,nc,head,nxt)
        if new<=old or np.random.random()<np.exp(-beta*(new-old)):
            accepted+=1;cnew=cell_index(x,L,nc);cold=cells[i]
            if cnew!=cold:
                if prev[i]>=0:nxt[prev[i]]=nxt[i]
                else:head[cold]=nxt[i]
                if nxt[i]>=0:prev[nxt[i]]=prev[i]
                prev[i]=-1;nxt[i]=head[cnew]
                if head[cnew]>=0:prev[head[cnew]]=i
                head[cnew]=i;cells[i]=cnew
            pos[i]=x;orient[i]=u
    return accepted/(count*len(pos))
@njit(cache=True)
def seed_numba(seed):np.random.seed(seed)
@njit(cache=True)
def forces(pos,orient,L):
    f=np.zeros_like(pos);tau=np.zeros_like(pos);U=0.
    for i in range(len(pos)):
      for j in range(i+1,len(pos)):
        d=pos[j]-pos[i];d-=L*np.rint(d/L)
        if np.dot(d,d)>=36:continue
        v=gb3d_pair(d,orient[i],orient[j]);U+=v[0];f[i]+=v[1:4];f[j]-=v[1:4];tau[i]+=v[4:7];tau[j]+=v[7:10]
    return U,f,tau
def initial(N,rho,seed):
    rng=np.random.default_rng(seed);L=(N/rho)**(1/3);side=int(np.ceil(N**(1/3)-1e-10))
    pos=np.array([[(i%side+.5)*L/side,((i//side)%side+.5)*L/side,(i//(side*side)+.5)*L/side] for i in range(N)])
    orient=rng.normal(size=(N,3));orient/=np.linalg.norm(orient,axis=1)[:,None]
    return pos,orient,L
def run(args):
    out=ROOT/args.out;out.mkdir(exist_ok=True,parents=True)
    if (out/'parameters.json').exists():raise RuntimeError('Existing run; refusing overwrite')
    pos,orient,L=initial(args.N,args.rho,args.seed);nc=max(1,int(L/6));head,nxt,prev,cells=build_cells(pos,L,nc)
    if forces(pos,orient,L)[0]>1e50:raise ValueError('Forbidden initial core')
    seed_numba(args.seed);p=vars(args).copy();p.update(L=L,kappa=3,kprime=5,mu=2,nu=1,center_switch_on=4.5,center_cutoff=6,model='3D_uniaxial_GB',physical_branch='r-sigma+1>0',units='sigma0=epsilon0=kB=1',pid=os.getpid(),orientation='unit vector on S2, apolar double cover')
    (out/'parameters.json').write_text(json.dumps(p,indent=2));frames=np.lib.format.open_memmap(out/'snapshots.npy',mode='w+',dtype='float64',shape=(args.frames,args.N,6))
    start=time.time();step=args.step;turn=args.turn;sweep=0
    log=(out/'observables.csv').open('w',buffering=1);log.write('phase,index,sweep,energy_per_particle,S2,Qxx,Qyy,Qxy,Qxz,Qyz,acceptance,step,turn,seconds\n')
    def record(phase,index,ac):
        U,_,_=forces(pos,orient,L);Q=1.5*orient.T@orient/args.N-.5*np.eye(3);S=np.linalg.eigvalsh(Q)[-1]
        log.write(f'{phase},{index},{sweep},{U/args.N},{S},{Q[0,0]},{Q[1,1]},{Q[0,1]},{Q[0,2]},{Q[1,2]},{ac},{step},{turn},{time.time()-start}\n')
    def status(phase,n):
        tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(phase=phase,frames_written=n,sweep=sweep,elapsed_seconds=time.time()-start,pid=os.getpid())));tmp.replace(out/'status.json')
    for k in range(0,args.equil,100):
        count=min(100,args.equil-k);ac=sweeps(pos,orient,L,1/args.T,step,turn,count,nc,head,nxt,prev,cells);sweep+=count;record('equil',k,ac)
        if k<args.equil//2:
            scale=np.exp(.5*(ac-.4));step=min(1.2,max(.03,step*scale));turn=min(np.pi,max(.05,turn*scale))
        status('equilibration',0)
    for k in range(args.frames):
        ac=sweeps(pos,orient,L,1/args.T,step,turn,args.stride,nc,head,nxt,prev,cells);sweep+=args.stride
        frames[k,:,:3]=pos;frames[k,:,3:]=orient;record('production',k,ac)
        if (k+1)%100==0:
            frames.flush();status('sampling',k+1);np.savez(out/'checkpoint.npz',pos=pos,orient=orient,sweep=sweep)
    frames.flush();status('completed',args.frames);log.close()
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name,default,typ in [('N',216,int),('rho',.05,float),('T',2.,float),('equil',5000,int),('frames',2000,int),('stride',10,int),('seed',410111,int),('step',.4,float),('turn',1.,float)]:p.add_argument('--'+name,type=typ,default=default)
    p.add_argument('--out',required=True);run(p.parse_args())
