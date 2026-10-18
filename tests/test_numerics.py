"""Numerical regression against archived data and independent derivative identities."""
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from anisotropic_fs.estimators import estimate_frame,radial_grid,shell_kernel

ROOT=Path(__file__).resolve().parents[1]

class ArchivedConfigurations(unittest.TestCase):
    def test_original_per_frame_estimates_and_normalizations(self):
        for model in ['dimer2d','gb2d','gb3d']:
            with self.subTest(model=model):
                z=np.load(ROOT/f'data/examples/{model}.npz',allow_pickle=False)
                x=estimate_frame(model,z['frame'],float(z['L']),float(z['T']))
                if model=='gb3d':
                    e=x['edges'];mask=(e[:-1]>=.9-1e-9)&(e[1:]<=5.5+1e-9)
                    got=np.stack([np.column_stack([x[k].reshape(len(e)-1,-1),x['radial_'+k]])[mask] for k in ['edh','efs']])
                    np.testing.assert_allclose(got,z['expected'],rtol=1e-11,atol=1e-11)
                else:
                    got=np.stack([x['edh'],x['efs']])
                    # Independent planar caches were originally stored in float32.
                    np.testing.assert_allclose(got,z['expected'],rtol=8e-8,atol=5e-8)
                d=3 if model=='gb3d' else 2;vol=np.diff(x['edges']**d);vol/=vol.sum()
                for k in ['edh','efs']:
                    np.testing.assert_allclose(x[k].reshape(len(vol),-1).mean(1),x['radial_'+k],atol=2e-12,rtol=1e-12)
                delta=(x['efs']-x['edh']).reshape(len(vol),-1)
                np.testing.assert_allclose(vol@delta,0,atol=2e-11,rtol=0)

    def test_all_primary_blocks_reconstruct_the_archived_target(self):
        saved=np.load(ROOT/'data/blocks/gb3d_targets100.npz')['targets100']
        for i,seed in enumerate([410131,410132,410133]):
            z=np.load(ROOT/f'data/blocks/gb3d_{seed}_moments100.npz');e=z['edges'];a=z['blocks'];mask=(e[:-1]>=.9-1e-9)&(e[1:]<=1.5+1e-9);w=np.diff(e**3)[mask];w/=w.sum()
            pair=np.einsum('r,bmr->bm',w,a[:,:,mask,2])
            np.testing.assert_allclose(pair[:,1]-pair[:,0],saved[i],atol=1e-14,rtol=0)

class Mechanics(unittest.TestCase):
    def test_three_dimensional_force_and_torque_derivatives(self):
        from anisotropic_fs.legacy.gb3d_mc import forces
        pos=np.array([[2.,2.,2.],[3.7,3.5,2.7],[7.1,2.4,3.4]])
        axes=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
        energy,force,torque=forces(pos,axes,20.);eps=1e-6
        self.assertTrue(np.isfinite(energy))
        for i in range(3):
            for j in range(3):
                p=pos.copy();m=pos.copy();p[i,j]+=eps;m[i,j]-=eps
                fd=-(forces(p,axes,20.)[0]-forces(m,axes,20.)[0])/(2*eps)
                self.assertAlmostEqual(fd,force[i,j],delta=2e-5*max(1,abs(fd)))
            p=axes.copy();m=axes.copy();u=axes[i];normal=np.array([0.,0.,1.])
            for sign,arr in [(1,p),(-1,m)]:
                t=sign*eps;arr[i]=u*np.cos(t)+np.cross(normal,u)*np.sin(t)+normal*np.dot(normal,u)*(1-np.cos(t))
            fd=-(forces(pos,p,20.)[0]-forces(pos,m,20.)[0])/(2*eps)
            self.assertAlmostEqual(fd,torque[i,2],delta=2e-5*max(1,abs(fd)))
        np.testing.assert_allclose(force.sum(0),0,atol=1e-12)

    def test_exact_planar_potential_force_torque_restriction(self):
        from anisotropic_fs.legacy.gb_pair import gb_pair
        from anisotropic_fs.legacy.gb3d_pair import gb3d_pair
        rng=np.random.default_rng(22714)
        for _ in range(32):
            th=rng.uniform(0,np.pi,2);u=np.array([np.cos(th[0]),np.sin(th[0])]);v=np.array([np.cos(th[1]),np.sin(th[1])]);phi=rng.uniform(0,2*np.pi);d=4.8*np.array([np.cos(phi),np.sin(phi)])
            a=gb_pair(d[0],d[1],u,v,3.);b=gb3d_pair(np.r_[d,0.],np.r_[u,0.],np.r_[v,0.])
            np.testing.assert_allclose(a,b[[0,1,2,6,9]],atol=1e-12,rtol=1e-11)

class FiniteDomainIdentity(unittest.TestCase):
    def test_nonconstant_distribution_exact_shell_reconstruction(self):
        g=lambda t:1+.2*np.sin(t)+.03*t*t
        gp=lambda t:.2*np.cos(t)+.06*t
        for d in [2,3,4,5,8]:
            a,b,lo,hi=.55,6.,1.2,1.6
            anchor=quad(lambda t:d*t**(d-1)*g(t)/(b**d-a**d),a,b,epsabs=1e-12)[0]
            fs=anchor+quad(lambda t:float(shell_kernel(t,lo,hi,a,b,d))*gp(t),a,b,points=[lo,hi],epsabs=1e-12)[0]
            dh=quad(lambda t:d*t**(d-1)*g(t)/(hi**d-lo**d),lo,hi,epsabs=1e-12)[0]
            self.assertAlmostEqual(fs,dh,delta=1e-10)

    def test_invalid_geometric_domain_is_rejected(self):
        with self.assertRaises(ValueError):estimate_frame('gb3d',np.zeros((2,6)),10.)
        with self.assertRaises(ValueError):shell_kernel(1.,2.,1.,.55,6.,3)

if __name__=='__main__':unittest.main()
