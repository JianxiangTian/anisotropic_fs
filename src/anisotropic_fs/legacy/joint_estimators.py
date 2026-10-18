"""Shell-averaged, pi-periodic Fourier DH and force/torque estimators.

All modes use ordered pairs, represented by both orientations of i<j.
Finite annulus anchor is exact; there is no assumed g(b)=1.
"""
from .dimer_mc import *

@njit(cache=True)
def estimate_frame(pos,theta,L,ell,T,edges,modes):
    orient=np.column_stack((np.cos(theta),np.sin(theta)))
    U,F,tau=forces(pos,orient,L,ell)
    nb,nm=len(edges)-1,len(modes)
    counts=np.zeros((nb,nm),np.complex128)
    vbin=np.zeros((nb,nm),np.complex128)
    v2bin=np.zeros((nb,nm),np.complex128)
    torque=np.zeros(nm,np.complex128)
    direct=np.zeros((nb,12,12))
    angular_v=np.zeros((nb,12,12))
    angular_v2=np.zeros((nb,12,12))
    N=len(pos); C=L*L/(N*(N-1)); beta=1/T
    a,b=edges[0],edges[-1]; span=b*b-a*a
    for i in range(N):
        for j in range(i+1,N):
            dx,dy=pos[j,0]-pos[i,0],pos[j,1]-pos[i,1]
            dx-=L*np.rint(dx/L);dy-=L*np.rint(dy/L)
            r2=dx*dx+dy*dy
            if r2<=a*a or r2>=b*b:
                continue
            r=np.sqrt(r2)
            h=min(nb-1,np.searchsorted(edges,r)-1)
            phi=np.arctan2(dy,dx)
            alpha=(theta[i]-phi)%np.pi
            gamma=(theta[j]-phi)%np.pi
            fr=((F[j,0]-F[i,0])*dx+(F[j,1]-F[i,1])*dy)/(2*r)
            v=beta*C/(2*np.pi)*fr/r
            ai=min(11,int(12*alpha/np.pi));gi=min(11,int(12*gamma/np.pi))
            direct[h,ai,gi]+=1;direct[h,gi,ai]+=1
            angular_v[h,ai,gi]+=144*v;angular_v[h,gi,ai]+=144*v
            angular_v2[h,ai,gi]+=144*v*r2;angular_v2[h,gi,ai]+=144*v*r2
            for k in range(nm):
                ka,kb=2*modes[k,0],2*modes[k,1]
                phase1=np.exp(-1j*(ka*alpha+kb*gamma))
                phase2=np.exp(-1j*(ka*gamma+kb*alpha))
                phase=phase1+phase2
                counts[h,k]+=phase
                vbin[h,k]+=v*phase
                v2bin[h,k]+=v*r2*phase
                if ka*ka+kb*kb>0:
                    torque[k]+=(-1j*beta*C/(np.pi*span)/(ka*ka+kb*kb))*((ka*tau[i]+kb*tau[j])*phase1+(ka*tau[j]+kb*tau[i])*phase2)
    anchor=C/(np.pi*span)*counts.sum(axis=0)
    fs=np.zeros((nb,nm),np.complex128)
    ft=np.zeros((nb,nm),np.complex128)
    common=(b*b*vbin-v2bin).sum(axis=0)/span
    prefix=np.zeros(nm,np.complex128)
    angular_anchor=C*144/(np.pi*span)*direct.sum(axis=0)
    angular_common=(b*b*angular_v-angular_v2).sum(axis=0)/span
    angular_prefix=np.zeros((12,12))
    angular_fs=np.zeros_like(direct)
    dh=np.zeros((nb,nm),np.complex128)
    for h in range(nb):
        den=edges[h+1]**2-edges[h]**2
        radial=prefix+(edges[h+1]**2*vbin[h]-v2bin[h])/den-common
        fs[h]=anchor+radial
        for k in range(nm):
            if modes[k,0]==0 and modes[k,1]==0:
                ft[h,k]=fs[h,k]
            else:
                ft[h,k]=torque[k]+radial[k]
        dh[h]=C/(np.pi*den)*counts[h]
        direct[h]*=C*144/(np.pi*den)
        angular_fs[h]=angular_anchor+angular_prefix+(edges[h+1]**2*angular_v[h]-angular_v2[h])/den-angular_common
        angular_prefix+=angular_v[h]
        prefix+=vbin[h]
    return dh,fs,ft,direct,angular_fs,U/N

def analyze(path,output,block_size=100):
    path=ROOT/path;out=ROOT/output;out.mkdir(parents=True,exist_ok=True)
    par=json.loads((path/'parameters.json').read_text())
    status=json.loads((path/'status.json').read_text())
    n=status['frames_written']
    if n<block_size:raise ValueError('Insufficient completed frames')
    frames=np.load(path/'snapshots.npy',mmap_mode='r')
    edges=np.linspace(.55,min(4.,.49*par['L']),139)
    modes=np.array([(i,j) for i in range(-2,3) for j in range(-2,3)])
    blocks=[];directs=[];angular_fss=[]
    start=time.time()
    for lo in range(0,n-block_size+1,block_size):
        accum=np.zeros((3,len(edges)-1,len(modes)),complex)
        hist=np.zeros((len(edges)-1,12,12))
        ahist=np.zeros_like(hist)
        for k in range(lo,lo+block_size):
            frame=frames[k]
            dh,fs,ft,h,af,_=estimate_frame(frame[:,:2].copy(),frame[:,2].copy(),par['L'],par['ell'],par['T'],edges,modes)
            accum+=np.array([dh,fs,ft]);hist+=h;ahist+=af
        blocks.append(accum/block_size);directs.append(hist/block_size);angular_fss.append(ahist/block_size)
        (out/'status.json').write_text(json.dumps(dict(frames_analyzed=lo+block_size,seconds=time.time()-start)))
    np.savez(out/'estimators.npz',edges=edges,modes=modes,blocks=np.array(blocks),
             direct_blocks=np.array(directs),angular_fs_blocks=np.array(angular_fss),frames_used=len(blocks)*block_size,block_size=block_size)
    print(json.dumps(dict(output=str(out),seconds=time.time()-start,frames_used=len(blocks)*block_size)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('output');p.add_argument('--block-size',type=int,default=100)
    a=p.parse_args();analyze(a.path,a.output,a.block_size)
