"""3D uniaxial GB; independent vector forces and torques, same scalar GB convention."""
from .gb_pair import geometry,np,njit

@njit(cache=True)
def gb3d_pair(d,u,v,kappa=3.,kprime=5.,mu=2.,nu=1.):
    out=np.zeros(10) # U, F_i[3], torque_i[3], torque_j[3]
    r=np.sqrt(np.dot(d,d))
    if r>=6:return out
    if r<1e-10:out[0]=1e80;return out
    e=d/r;a=np.dot(u,e);b=np.dot(v,e);c=np.dot(u,v)
    sig,sa,sb,sc,eps,ea,eb,ec=geometry(a,b,c,kappa,kprime,mu,nu)
    s=r-sig+1
    if s<=1e-8:out[0]=1e80;return out
    z=1/s;z6=z**6;shape=4*(z6*z6-z6);energy=eps*shape
    us=24*eps*(z**7-2*z**13)
    ua=ea*shape-us*sa;ub=eb*shape-us*sb;uc=ec*shape-us*sc
    sw=1.;ds=0.
    if r>4.5:
        t=(r-4.5)/1.5;sw=1-10*t**3+15*t**4-6*t**5;ds=(-30*t*t+60*t**3-30*t**4)/1.5
    ur=us*sw+energy*ds;ua*=sw;ub*=sw;uc*=sw
    out[0]=energy*sw;out[1:4]=ur*e+(ua*(u-a*e)+ub*(v-b*e))/r
    out[4:7]=-np.cross(u,ua*e+uc*v)
    out[7:10]=-np.cross(v,ub*e+uc*u)
    return out
