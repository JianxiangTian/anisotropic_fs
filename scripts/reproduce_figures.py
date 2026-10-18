"""Reproduce all six main and three supplementary figures from bundled data."""
from _common import ROOT
from pathlib import Path
import argparse,json,warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,FancyBboxPatch,Arc
from matplotlib.font_manager import findfont,FontProperties
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'outputs/figures');a=p.parse_args();O=a.out;O.mkdir(parents=True,exist_ok=True)
def read(rel):return json.loads((ROOT/rel).read_text(encoding='utf-8'))
def load(rel):return np.load(ROOT/rel,allow_pickle=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold','mathtext.bfit':'Times New Roman:bold:italic','axes.linewidth':.85,'axes.spines.top':True,'axes.spines.right':True,'legend.frameon':False,'legend.fontsize':9,'xtick.labelsize':10,'ytick.labelsize':10,'axes.labelsize':12,'axes.titlesize':12,'savefig.facecolor':'white','ps.fonttype':42,'pdf.fonttype':42})

def save(f,n):
 for ext in ['eps','pdf','png']:f.savefig(O/f'Fig. {n}.{ext}',bbox_inches='tight',dpi=350)
 plt.close(f)
def label(ax,s):ax.text(0,1.035,s,transform=ax.transAxes,ha='left',va='bottom',fontsize=12)
def ylab(ax,s):ax.set_ylabel(s,rotation=0,labelpad=25)
def style(ax):ax.tick_params(direction='in',top=False,right=False);ax.grid(False)
def panels(rows,cols,size=(8.2,5.8)):
 f,a=plt.subplots(rows,cols,figsize=size,squeeze=False);f.subplots_adjust(left=.13,right=.985,bottom=.12,top=.94,wspace=.40,hspace=.43)
 for ax in a.flat:style(ax)
 return f,a

try:
    findfont(FontProperties(family='Times New Roman'),fallback_to_default=False)
except ValueError:
    warnings.warn('Times New Roman is not installed. Using DejaVu Serif; install Times New Roman locally for manuscript-matched fonts. No font files are bundled.')
    plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.rm':'DejaVu Serif','mathtext.it':'DejaVu Serif:italic','mathtext.bf':'DejaVu Serif:bold','mathtext.bfit':'DejaVu Serif:bold:italic'})

# Figure 1. Pair geometry and estimation workflow.
f,axs=plt.subplots(1,3,figsize=(9.2,3.4),gridspec_kw={'width_ratios':[1,1,1.45]})
for ax in axs:ax.axis('off')
ax=axs[0];ax.set_xlim(-.7,3.1);ax.set_ylim(-1.6,1.6)
for yy,ang in [(1,(0,0)),(-.8,(90,0))]:
 ax.plot([0,2.1],[yy,yy],color='.65',ls='--',lw=.8)
 for xx,aa in zip([0,2.1],ang):ax.add_patch(Ellipse((xx,yy),1.05,.32,angle=aa,fc='#EAF1F8',ec='#233A57',lw=1.2))
 ax.text(1.05,yy+.22,r'$r$',ha='center')
ax.set_title('(a) Pair arrangements',loc='left');ax.text(1.05,.1,'Same separation',ha='center',fontsize=11)
ax=axs[1];ax.set_xlim(-.55,2.7);ax.set_ylim(-1.4,1.6)
ax.plot([0,2.1],[.55,.55],ls='--',color='.6')
for xx,ang in [(0,35),(2.1,65)]:
 ax.add_patch(Ellipse((xx,.55),1.05,.32,angle=ang,fc='#EAF1F8',ec='#233A57',lw=1.2))
 ax.plot([xx,xx+.7*np.cos(np.deg2rad(ang))],[.55,.55+.7*np.sin(np.deg2rad(ang))],color='#1647a0',lw=1)
ax.add_patch(Arc((0,.55),.85,.85,theta1=0,theta2=35,lw=.8,color='#1647A0'));ax.add_patch(Arc((2.1,.55),.6,.6,theta1=0,theta2=65,lw=.8,color='#1647A0'))
ax.text(.48,.65,r'$\alpha$');ax.text(2.52,.84,r'$\gamma$');ax.text(1.05,-.45,r'2D  $q=(\alpha,\gamma)$',ha='center',fontsize=11);ax.text(1.05,-.9,r'3D  $q=(x,y,\psi)$',ha='center',fontsize=11);ax.set_title('(b) Relative orientation',loc='left')
ax=axs[2];ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title('(c) Common-bin estimation',loc='left')
for y,txt in [(.80,'Configurations and total forces'),(.53,'Shared shells and angular cells'),(.19,'EDH counts\nEFS anchor + force integral')]:
 ax.add_patch(FancyBboxPatch((.035,y-.1),.93,.20 if y>.3 else .25,boxstyle='round,pad=.008',fc='#F4F4F4' if y>.3 else '#EAF1F8',ec='#677685' if y>.3 else '#1647A0',lw=.9))
 if y>.3:ax.text(.5,y,txt,ha='center',va='center',fontsize=10.5,color='black')
 else:
  ax.text(.5,.242,'EDH counts',ha='center',va='center',fontsize=10.5,color='black')
  ax.text(.5,.178,'EFS anchor + force integral',ha='center',va='center',fontsize=10.5,color='#1647A0')
ax.annotate('',xy=(.5,.64),xytext=(.5,.69),arrowprops={'arrowstyle':'->','color':'#1647A0'});ax.annotate('',xy=(.5,.345),xytext=(.5,.42),arrowprops={'arrowstyle':'->','color':'#1647A0'})
f.subplots_adjust(left=.015,right=.99,top=.85,bottom=.05,wspace=.3);save(f,1)

f,a=panels(2,2)
for row,(model,title) in enumerate([('dimer2d','Dimers'),('gb2d','Gay–Berne')]):
    z=load(f'data/blocks/{model}_demo.npz');rr=(z['edges'][1:]+z['edges'][:-1])/2
    for m,name,col,mark in [(0,'EDH','black','o'),(1,'EFS','#1647a0','s')]:
        rad=z['radial100'][:,m].reshape(20,5,-1).mean(1)
        cut=z['angular100'][:,m,row].reshape(20,5,12).mean(1)
        for j,(xx,yy) in enumerate([(rr,rad),(np.arange(12)*np.pi/12+np.pi/24,cut)]):
            sel=slice(None,None,3) if j==0 else slice(None)
            a[row,j].errorbar(xx[sel],yy.mean(0)[sel],yerr=yy.std(0,ddof=1)[sel]/np.sqrt(len(yy)),fmt=mark,color=col,mfc='none',ms=3.2,mew=.8,capsize=1.5,lw=.7,label=name)
    for j in range(2):a[row,j].set_title(title);label(a[row,j],f'({chr(97+row*2+j)})');a[row,j].legend(loc='upper right')
    a[row,0].legend(loc='lower right');a[row,0].set_xlim(.6,3.5 if row==0 else 5.5);a[row,0].set_xlabel(r'$r$');ylab(a[row,0],r'$g_d^{\mathrm{rad}}(r)$')
    a[row,1].set_xlabel(r'$\alpha$');ylab(a[row,1],r'$\overline{g}_{h,A}$');a[row,1].set_xticks([0,np.pi/2,np.pi],[r'$0$',r'$\pi/2$',r'$\pi$'])
a[1,1].legend(loc='lower center');save(f,2)
allb=[]
for seed in [410131,410132,410133]:
    with load(f'data/blocks/gb3d_{seed}_moments100.npz') as z:
        b=z['blocks'];edges=z['edges'];allb.append(b.reshape(160,5,*b.shape[1:]).mean(1))
rr=(edges[1:]+edges[:-1])/2
f,a=panels(1,2,(8.2,3.2))
for m,name,col,mark in [(0,'EDH','black','o'),(1,'EFS','#1647a0','s')]:
 for ax,ch in zip(a[0],[0,2]):
  vals=[b[:,m,:,ch] for b in allb];mean=np.mean([v.mean(0) for v in vals],axis=0);se=np.sqrt(sum(v.var(0,ddof=1)/len(v) for v in vals))/3
  ax.errorbar(rr[::2],mean[::2],yerr=se[::2],fmt=mark,color=col,mfc='none',ms=3.2,mew=.8,capsize=1.5,lw=.7,label=name)
for j,(ax,title,yl) in enumerate(zip(a[0],['Radial marginal','Angular moment'],[r'$g_d^{\mathrm{rad}}(r)$',r'$g_{P_2}(r)$'])):
 ax.set_xlim(.6,5.5);ax.set_xlabel(r'$r$');ylab(ax,yl);ax.set_title(title);label(ax,f'({chr(97+j)})');ax.legend()
f.subplots_adjust(bottom=.22);save(f,3)
# Figure 4. Planar efficiency curves.
f,a=panels(2,2)
for i,(path,title) in enumerate([('data/efficiency/dimer2d.json','Dimers'),('data/efficiency/gb2d.json','Gay–Berne')]):
 d=read(path);rows=d['summaries']
 if i==1:rows=rows+read('data/efficiency/gb2d_extended.json')['summaries']
 for j,q in enumerate(['radial','joint']):
  ax=a[i,j]
  for method,col,mark in [('DH','black','o'),('FS','#1647a0','s')]:
   for mode,fill,fac in [('dispersed_without_replacement',False,.975),('contiguous',True,1.025)]:
    sel=[x for x in rows if x['quantity']==q and x['method']==method and (x['protocol']==mode or (mode=='contiguous' and x['protocol'].startswith('contiguous')))]
    ms=sorted(set(x['M'] for x in sel));ys=[max(x['conditional_q95_CI95'][1] for x in sel if x['M']==m) for m in ms]
    ax.plot(np.array(ms)*fac,ys,linestyle='none',marker=mark,color=col,mfc=col if fill else 'none',ms=3.6,mew=.85,label=('EDH' if method=='DH' else 'EFS')+(' contiguous' if fill else ' dispersed'))
  ax.axhline(.005 if j==0 else .02,color='#ae2633',ls='--',lw=.9);ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel(r'$M$');ylab(ax,r'$U_{0.95}$');ax.set_title(title+' / '+('radial' if j==0 else 'joint'));label(ax,f'({chr(97+2*i+j)})');ax.legend(fontsize=8.4,loc='upper right')
save(f,4)
# Figure 5. Primary 3D RMS comparison.
d=read('data/efficiency/gb3d.json');ms=sorted(map(int,d['by_M']))
f,a=panels(2,2)
for i,q in enumerate(['radial','angular']):
 for j,mode in enumerate(['contiguous','dispersed']):
  ax=a[i,j]
  for name,col,mark in [('EDH','black','o'),('EFS','#1647a0','s')]:
   z=[d['by_M'][str(m)][mode][q][name] for m in ms];y=np.array([v['RMS'] for v in z]);ci=np.array([v['conditional_RMS_CI95'] for v in z]);ax.errorbar(ms,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt=mark,color=col,mfc='none',ms=3.6,mew=.85,capsize=1.5,lw=.8,label=name)
  ax.axhline(.005 if i==0 else .02,color='#ae2633',ls='--',lw=.9,label='Target');ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel(r'$M$');ylab(ax,r'$\mathrm{RMS}(\varepsilon)$');ax.set_title(('Radial' if i==0 else 'Joint')+' / '+mode);label(ax,f'({chr(97+2*i+j)})');ax.legend(loc='upper right')
save(f,5)
# Figure 6. Original, continued and combined fixed endpoint.
d=read('data/reported/primary_endpoint.json');bs=[100,200,500,1000,2000,4000]
f,a=panels(2,1,(8.2,5.5));f.subplots_adjust(left=.16,hspace=.55)
for j,(key,name,col,mark) in enumerate([('original20k','Original 20k','black','o'),('new60k','Added 60k','#1647a0','s'),('combined80k','Combined 80k','#ae2633','^')]):
 rows=[d['datasets'][key]['blocks'][str(b)] for b in bs];y=np.array([x['mean'] for x in rows])*1e3;ci=np.array([x['CI95_bootstrap_t'] for x in rows])*1e3
 a[0,0].errorbar(np.arange(6)+(j-1)*.14,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt=mark,color=col,mfc='none',ms=4,capsize=2,lw=.8,label=name)
a[0,0].set_xticks(range(6),bs);a[0,0].set_xlabel('Block length (configurations)');ylab(a[0,0],r'$10^3\overline{\mathit{\Delta}}$');a[0,0].axhline(.75,ls='--',color='.5',lw=.8);a[0,0].axhline(-.75,ls='--',color='.5',lw=.8);a[0,0].axhline(0,color='.75',lw=.6);a[0,0].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.24));label(a[0,0],'(a)')
x=load('data/blocks/gb3d_targets100.npz')['targets100']
for j,(s,col,mark) in enumerate([(410131,'black','o'),(410132,'#1647a0','s'),(410133,'#ae2633','^')]):
 z=x[j].reshape(8,20,5).mean(2);a[1,0].errorbar(np.arange(1,9)+(j-1)*.14,z.mean(1)*1e3,yerr=z.std(1,ddof=1)/np.sqrt(20)*1e3,fmt=mark,color=col,mfc='none',ms=4,capsize=2,lw=.8,label=str(s))
a[1,0].axhline(0,color='.6',lw=.7);a[1,0].set_xlabel('Successive 10 000-configuration segment');ylab(a[1,0],r'$10^3\overline{\mathit{\Delta}}$');a[1,0].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.24));label(a[1,0],'(b)');save(f,6)
# Supplementary kernel and reference rotations.
f,a=panels(1,2,(8.2,3.2));t=np.linspace(.55,6,1200)
for ax,dim in zip(a[0],[2,3]):
 for rr,col in [(1.2,'black'),(2.5,'#1647a0'),(4.8,'#ae2633')]:ax.plot(t,(rr>=t).astype(float)-(6**dim-t**dim)/(6**dim-.55**dim),color=col,lw=1.1,label=fr'$r={rr}$')
 ax.set_xlabel(r'$t$');ylab(ax,r'$K_d(r,t)$');ax.set_title(fr'$d={dim}$');ax.legend();ax.axhline(0,lw=.5,color='.7')
f.subplots_adjust(bottom=.22);save(f,'S1')
d=read('data/efficiency/gb3d_rotations.json');seeds=list(d['rotations']);ms=sorted(map(int,d['rotations'][seeds[0]]['by_M']))
f,a=panels(2,2)
for i,q in enumerate(['radial','angular']):
 for j,mode in enumerate(['contiguous','dispersed_blocks']):
  ax=a[i,j]
  for seed,col,mark,fac in zip(seeds,['black','#1647a0','#ae2633'],['o','s','^'],[.975,1,1.025]):
   z=[d['rotations'][seed]['by_M'][str(m)][mode][q] for m in ms];y=np.array([v['EFS_EDH_ratio'] for v in z]);ci=np.array([v['ratio_CI95'] for v in z]);ax.errorbar(np.array(ms)*fac,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt=mark,color=col,mfc='none',ms=3.5,capsize=1.5,lw=.7,label=seed)
  ax.axhline(1,color='.5',ls='--',lw=.8);ax.set_xscale('log');ax.set_xlabel(r'$M$');ylab(ax,'EFS/EDH');ax.set_ylim(.35,1.08);ax.set_title(('Radial' if i==0 else 'Joint')+' / '+('contiguous' if j==0 else 'dispersed blocks'));label(ax,f'({chr(97+2*i+j)})');ax.legend(loc='upper right',fontsize=8)
save(f,'S3')

z=load('data/blocks/gb2d_demo.npz');f,a=panels(1,1,(5.5,3.5));ax=a[0,0]
for m,name,col,mark in [(0,'EDH','black','o'),(1,'EFS','#1647a0','s')]:
    v=z['angular100'][:,m,0].reshape(20,5,12).mean(1)
    ax.errorbar(np.arange(12)*np.pi/12+np.pi/24,v.mean(0),yerr=v.std(0,ddof=1)/np.sqrt(len(v)),fmt=mark,color=col,mfc='none',ms=3.6,capsize=2,lw=.8,label=name)
ax.set_xlabel(r'$\alpha$');ylab(ax,r'$\overline{g}_{h,A}$');ax.set_xticks([0,np.pi/2,np.pi],[r'$0$',r'$\pi/2$',r'$\pi$']);ax.legend();ax.set_title(r'Gay–Berne core section, $r\approx1.25$');f.subplots_adjust(left=.23,bottom=.22);save(f,'S2')
print('Wrote all 9 figures as EPS, PDF and PNG to',O)
