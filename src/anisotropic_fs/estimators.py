"""Validated public interface to the retained production estimators."""
import numpy as np

MODELS = ("dimer2d", "gb2d", "gb3d")

def radial_grid(model, upper=None):
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {MODELS}")
    width = 0.05 if model == "gb3d" else 0.025
    end = (4.0 if model == "dimer2d" else 6.0) if upper is None else float(upper)
    count = int(round((end-0.55)/width))
    if count < 1 or not np.isclose(0.55+count*width, end):
        raise ValueError("Upper radius must align with the manuscript radial grid")
    return np.linspace(0.55, end, count+1)

def estimate_frame(model, frame, box_length, temperature=2.0, edges=None):
    """Return EDH/EFS angular arrays, radial marginals, and additional moments.

    Planar frame columns are x,y,theta (radians). Three-dimensional columns
    are x,y,z,ux,uy,uz. Coordinates are periodic in a square/cubic box.
    Potential parameters are fixed to the manuscript (dimer ell=0.5, GB kappa=3).
    Estimation never clips negative EFS values or removes low-count cells.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}")
    frame=np.asarray(frame,dtype=np.float64)
    edges=radial_grid(model) if edges is None else np.asarray(edges,dtype=np.float64)
    columns=6 if model=="gb3d" else 3
    if frame.ndim!=2 or frame.shape[1]!=columns or frame.shape[0]<2 or not np.isfinite(frame).all():
        raise ValueError(f"Expected finite (N,{columns}) frame with N >= 2")
    if not np.isfinite(box_length) or box_length<=0 or not np.isfinite(temperature) or temperature<=0:
        raise ValueError("Positive finite box length and temperature required")
    if edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all() or edges[0]<=0 or np.any(np.diff(edges)<=0) or edges[-1]>=box_length/2:
        raise ValueError("Strictly increasing positive edges and b < L/2 required")
    if model=="gb3d":
        from .legacy.gb3d_estimators import estimate
        if not np.allclose(np.linalg.norm(frame[:,3:],axis=1),1,atol=1e-9,rtol=0):
            raise ValueError("Three-dimensional axes must be unit vectors")
        dh,fs,weak=estimate(frame,float(box_length),float(temperature),edges)
        return dict(edges=edges,edh=dh[:,:128].reshape(-1,4,4,8),efs=fs[:,:128].reshape(-1,4,4,8),
                    radial_edh=dh[:,128],radial_efs=fs[:,128],moments_edh=dh[:,128:],moments_efs=fs[:,128:],weak=weak)
    if model=="dimer2d":
        from .legacy.joint_estimators import estimate_frame as estimate
        parameter=0.5
    else:
        from .legacy.gb_estimators import estimate_frame as estimate
        parameter=3.0
    modes=np.array([(0,0),(1,0),(1,-1),(1,1),(2,0)],dtype=np.int64)
    dh,fs,ft,h,af,energy=estimate(frame[:,:2].copy(),frame[:,2].copy(),float(box_length),parameter,float(temperature),edges,modes)
    return dict(edges=edges,edh=h,efs=af,radial_edh=h.mean((1,2)),radial_efs=af.mean((1,2)),
                fourier_edh=dh,fourier_efs=fs,torque_anchor_fourier=ft,modes=modes,energy_per_particle=energy)

def shell_kernel(t, lower, upper, a, b, dimension):
    """Exact target-shell kernel K_h(t), manuscript Eq. (10)/SI Eq. (S7)."""
    if dimension<2 or int(dimension)!=dimension or not 0<a<=lower<upper<=b:
        raise ValueError("Require integer d >= 2 and 0 < a <= lower < upper <= b")
    t=np.asarray(t,dtype=float);d=int(dimension)
    wh=np.where(t<=lower,1.0,np.where(t>=upper,0.0,(upper**d-t**d)/(upper**d-lower**d)))
    return wh-(b**d-t**d)/(b**d-a**d)
