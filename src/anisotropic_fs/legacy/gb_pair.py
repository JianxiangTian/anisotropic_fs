"""Uniaxial GB(3,5,2,1), planar axes, analytic chain-rule forces/torques.
Center-distance C2 switch 4.5 to 6. sigma0=epsilon0=1.
Only the physical branch s=r-sigma+1>0 is accessible; s<=0 is excluded.
"""
from .dimer_mc import np,njit

@njit(cache=True)
def shape_terms(a,b,c,q):
    p=a+b;m=a-b;dp=1+q*c;dm=1-q*c
    value=1-.5*q*(p*p/dp+m*m/dm)
    da=-q*(p/dp+m/dm)
    db=-q*(p/dp-m/dm)
    dc=.5*q*q*(p*p/(dp*dp)-m*m/(dm*dm))
    return value,da,db,dc

@njit(cache=True)
def geometry(a,b,c,kappa,kprime,mu,nu):
    chi=(kappa*kappa-1)/(kappa*kappa+1)
    cp=(kprime**(1/mu)-1)/(kprime**(1/mu)+1)
    h,ha,hb,hc=shape_terms(a,b,c,chi)
    j,ja,jb,jc=shape_terms(a,b,c,cp)
    sigma=h**(-.5);factor=-.5*h**(-1.5)
    eps=(1-chi*chi*c*c)**(-nu/2)*j**mu
    return sigma,factor*ha,factor*hb,factor*hc,eps,eps*mu*ja/j,eps*mu*jb/j,eps*(mu*jc/j+nu*chi*chi*c/(1-chi*chi*c*c))

@njit(cache=True)
def gb_pair(dx,dy,ni,nj,kappa=3.,kprime=5.,mu=2.,nu=1.):
    out=np.zeros(5);r2=dx*dx+dy*dy
    if r2>=36.:return out
    if r2<1e-20:
        out[0]=1e80;return out
    r=np.sqrt(r2);ex=dx/r;ey=dy/r
    a=ni[0]*ex+ni[1]*ey;b=nj[0]*ex+nj[1]*ey;c=ni[0]*nj[0]+ni[1]*nj[1]
    sig,sa,sb,sc,eps,ea,eb,ec=geometry(a,b,c,kappa,kprime,mu,nu)
    s=r-sig+1
    if s<=1e-8:
        out[0]=1e80;return out
    z=1/s;z6=z**6;v=4*(z6*z6-z6);u=eps*v
    us=24*eps*(z**7-2*z**13)
    ua=ea*v-us*sa;ub=eb*v-us*sb;uc=ec*v-us*sc
    switch=1.;ds=0.
    if r>4.5:
        t=(r-4.5)/1.5
        switch=1-10*t**3+15*t**4-6*t**5
        ds=(-30*t*t+60*t**3-30*t**4)/1.5
    ur=us*switch+u*ds;ua*=switch;ub*=switch;uc*=switch
    out[0]=u*switch
    out[1]=ur*ex+(ua*(ni[0]-a*ex)+ub*(nj[0]-b*ex))/r
    out[2]=ur*ey+(ua*(ni[1]-a*ey)+ub*(nj[1]-b*ey))/r
    out[3]=-(ua*(-ni[1]*ex+ni[0]*ey)+uc*(-ni[1]*nj[0]+ni[0]*nj[1]))
    out[4]=-(ub*(-nj[1]*ex+nj[0]*ey)+uc*(-nj[1]*ni[0]+nj[0]*ni[1]))
    return out
