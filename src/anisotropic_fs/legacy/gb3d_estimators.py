"""Exact 3D shell averages, full-S2 angular measure, total molecular forces."""
from .gb3d_mc import *
import sys
@njit(cache=True)
def estimate(frame,L,T,edges):
    pos=frame[:,:3].copy();ori=frame[:,3:].copy();U,F,tau=forces(pos,ori,L)
    n=len(pos);nr=len(edges)-1;na=128;nc=na+5
    counts=np.zeros((nr,nc));vbin=np.zeros_like(counts);v3bin=np.zeros_like(counts);weak=np.zeros((nr,2))
    C=L**3/(n*(n-1));beta=1/T;pref=beta*C/(4*np.pi);a=edges[0];b=edges[-1];B=b**3-a**3
    for i in range(n):
      for j in range(n):
        if j==i:continue
        d=pos[j]-pos[i];d-=L*np.rint(d/L);r=np.sqrt(np.dot(d,d))
        if r<a or r>=b:continue
        e=d/r;x=np.dot(ori[i],e);y=np.dot(ori[j],e);c=np.dot(ori[i],ori[j])
        psi=np.arctan2(np.dot(e,np.cross(ori[i],ori[j])),c-x*y)%(2*np.pi)
        ix=min(3,max(0,int((x+1)*2)));iy=min(3,max(0,int((y+1)*2)));ip=min(7,int(psi/(2*np.pi)*8));angle=(ix*4+iy)*8+ip
        h=min(nr-1,np.searchsorted(edges,r,side='right')-1)
        fr=.5*np.dot(F[j]-F[i],e);v=pref*fr/(r*r)
        px=(3*x*x-1)/2;py=(3*y*y-1)/2;pc=(3*c*c-1)/2
        weights=np.array([1.,.5*(px+py),pc,px*py,(35*c**4-30*c*c+3)/8])
        counts[h,angle]+=1;vbin[h,angle]+=v;v3bin[h,angle]+=v*r**3
        for k in range(5):
            ch=na+k;w=weights[k];counts[h,ch]+=w;vbin[h,ch]+=v*w;v3bin[h,ch]+=v*w*r**3
        weak[h,0]+=-6*px+beta*3*x*np.dot(tau[i],np.cross(ori[i],e))
        weak[h,1]+=-6*pc+beta*3*c*np.dot(tau[i],np.cross(ori[i],ori[j]))
    dh=np.zeros_like(counts);fs=np.zeros_like(counts);prefix=np.zeros(nc)
    anchor=C/(4*np.pi*B/3)*counts.sum(axis=0);common=(b**3*vbin.sum(axis=0)-v3bin.sum(axis=0))/B
    for h in range(nr):
        den=edges[h+1]**3-edges[h]**3
        dh[h]=C/(4*np.pi*den/3)*counts[h]
        fs[h]=anchor+prefix+(edges[h+1]**3*vbin[h]-v3bin[h])/den-common
        weak[h]*=C/(4*np.pi*den/3)
        prefix+=vbin[h]
    dh[:,:na]*=na;fs[:,:na]*=na
    return dh,fs,weak
def analyze(source,output):
    src=ROOT/source;out=ROOT/output;out.mkdir(parents=True,exist_ok=True)
    p=json.loads((src/'parameters.json').read_text());status=json.loads((src/'status.json').read_text());assert status['phase']=='completed'
    frames=np.load(src/'snapshots.npy',mmap_mode='r');edges=np.linspace(.55,min(6.,.49*p['L']),110);blocks=[];wblocks=[];start=time.time()
    for lo in range(0,len(frames),100):
        acc=np.zeros((2,len(edges)-1,133));wa=np.zeros((len(edges)-1,2));count=0
        for frame in frames[lo:lo+100]:
            dh,fs,weak=estimate(frame,p['L'],p['T'],edges);acc+=np.array([dh,fs]);wa+=weak;count+=1
        blocks.append(acc/count);wblocks.append(wa/count)
        (out/'status.json').write_text(json.dumps(dict(frames_analyzed=lo+count,seconds=time.time()-start,pid=os.getpid())))
    np.savez(out/'estimators.npz',edges=edges,blocks=np.array(blocks),weak_blocks=np.array(wblocks),frames_used=len(frames),block_size=100,angular_shape=np.array([4,4,8]))
    (out/'estimator_metadata.json').write_text(json.dumps(dict(measure='dx dy dpsi / (8*pi)',moments=['1','(P2(x)+P2(y))/2','P2(ui.uj)','P2(x)P2(y)','P4(ui.uj)'],radial_bin_width=float(edges[1]-edges[0]),normalization='V/[N(N-1)], ordered pairs',seconds=time.time()-start),indent=2))
if __name__=='__main__':analyze(*sys.argv[1:])
