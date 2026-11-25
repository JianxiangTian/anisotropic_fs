from pathlib import Path
import sys,json,hashlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,FancyBboxPatch,Arc
from matplotlib.font_manager import findfont,FontProperties
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'data'
O=ROOT/'outputs'/'figures'
O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold','mathtext.bfit':'Times New Roman:bold:italic','axes.linewidth':.85,'axes.spines.top':True,'axes.spines.right':True,'legend.frameon':False,'legend.fontsize':9,'xtick.labelsize':10,'ytick.labelsize':10,'axes.labelsize':12,'axes.titlesize':12,'savefig.facecolor':'white','ps.fonttype':42,'pdf.fonttype':42})
assert 'times' in findfont(FontProperties(family='Times New Roman')).lower()
sources=[]
def read(rel):
 p=R/rel;sources.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return json.loads(p.read_text())
def load(rel):
 p=R/rel;sources.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return np.load(p)
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
# Figure 1. Original vector schematic, with no numerical data.
f,axs=plt.subplots(1,3,figsize=(9.2,3.4),gridspec_kw={'width_ratios':[1,1,1.45]})
for ax in axs:ax.axis('off')
ax=axs[0];ax.set_xlim(-.7,3.1);ax.set_ylim(-1.6,1.6)
for yy,ang in [(1,(0,0)),(-.8,(90,0))]:
 ax.plot([0,2.1],[yy,yy],color='.65',ls='--',lw=.8)
 for xx,aa in zip([0,2.1],ang):ax.add_patch(Ellipse((xx,yy),1.05,.32,angle=aa,fc='white',ec='black',lw=1.2))
 ax.text(1.05,yy+.22,r'$r$',ha='center')
ax.set_title('(a) Pair arrangements',loc='left');ax.text(1.05,.1,'Same separation',ha='center',fontsize=11)
ax=axs[1];ax.set_xlim(-.55,2.7);ax.set_ylim(-1.4,1.6)
ax.plot([0,2.1],[.55,.55],ls='--',color='.6')
for xx,ang in [(0,35),(2.1,65)]:
 ax.add_patch(Ellipse((xx,.55),1.05,.32,angle=ang,fc='white',ec='black',lw=1.2))
 ax.plot([xx,xx+.7*np.cos(np.deg2rad(ang))],[.55,.55+.7*np.sin(np.deg2rad(ang))],color='#1647a0',lw=1)
ax.add_patch(Arc((0,.55),.85,.85,theta1=0,theta2=35,lw=.8));ax.add_patch(Arc((2.1,.55),.6,.6,theta1=0,theta2=65,lw=.8))
ax.text(.48,.65,r'$\alpha$');ax.text(2.52,.84,r'$\gamma$');ax.text(1.05,-.45,r'2D  $q=(\alpha,\gamma)$',ha='center',fontsize=11);ax.text(1.05,-.9,r'3D  $q=(x,y,\psi)$',ha='center',fontsize=11);ax.set_title('(b) Relative orientation',loc='left')
ax=axs[2]
ax.axis("on")
ax.set_position([.64,.21,.32,.52])
edges=np.linspace(.55,6,12)
centers=(edges[1:]+edges[:-1])/2
t=2.85
fraction=np.clip((edges[1:]**2-t*t)/np.diff(edges**2),0,1)
kernel=fraction-(6**2-t*t)/(6**2-.55**2)
ax.axhline(0,color=".55",lw=.8)
ax.bar(centers,kernel,width=np.diff(edges)*.87,color="#315B84")
ax.axvline(t,color="#903C31",ls="--",lw=1.2)
ax.set(ylim=(-.92,.38),xlabel=r"Target shell center $r$",title="(c) One pair at fixed separation")
ax.text(t+.10,-.64,r"$r_{ij}$",color="#903C31")
ax.set_ylabel(r"$K_h(r_{ij})$",rotation=0,labelpad=14)
f.subplots_adjust(left=.015,right=.98,top=.85,bottom=.17,wspace=.45);save(f,1)
# Figure 2. Same demonstration data and cells as v2.
f,a=panels(2,2)
for row,(path,title) in enumerate([('dimer_profiles.npz','Dimers'),('gb_profiles.npz','Gay–Berne')]):
 z=load(path);rr=(z['edges'][1:]+z['edges'][:-1])/2
 for name,col,mark in [('EDH','black','o'),('EFS','#1647a0','s')]:
  rad=z[name+'_radial'];ang=z[name+'_angle']
  for j,(xx,yy) in enumerate([(rr,rad),(np.arange(12)*np.pi/12+np.pi/24,ang)]):
   sel=slice(None,None,3) if j==0 else slice(None)
   a[row,j].errorbar(xx[sel],yy.mean(0)[sel],yerr=yy.std(0,ddof=1)[sel]/np.sqrt(len(yy)),fmt=mark,color=col,mfc='none',ms=3.2,mew=.8,capsize=1.5,lw=.7,label=name)
 for j in range(2):a[row,j].set_title(title);label(a[row,j],f'({chr(97+row*2+j)})');a[row,j].legend(loc='upper right')
 a[row,0].legend(loc='lower right');a[row,0].set_xlim(.6,3.5 if row==0 else 5.5);a[row,0].set_xlabel(r'$r$');ylab(a[row,0],r'$g_d^{\mathrm{rad}}(r)$')
 a[row,1].set_xlabel(r'$\alpha$');ylab(a[row,1],r'$\overline{g}_{h,A}$');a[row,1].set_xticks([0,np.pi/2,np.pi],[r'$0$',r'$\pi/2$',r'$\pi$'])
save(f,"S1")
# Figure 3. Equal-chain means and propagated within-chain standard errors.
allb=[]
for seed in [410131,410132,410133]:
 with load(f'gb3d_profiles_{seed}.npz') as z:
  allb.append(z['blocks']);edges=z['edges']
rr=(edges[1:]+edges[:-1])/2
f,a=panels(2,2,(8.2,5.8))
for m,name,col,mark in [(0,'EDH','black','o'),(1,'EFS','#1647a0','s')]:
 for ax,ch in zip(a[0],[0,1]):
  vals=[b[:,m,:,ch] for b in allb];mean=np.mean([v.mean(0) for v in vals],axis=0);se=np.sqrt(sum(v.var(0,ddof=1)/len(v) for v in vals))/3
  ax.errorbar(rr[::2],mean[::2],yerr=se[::2],fmt=mark,color=col,mfc='none',ms=3.2,mew=.8,capsize=1.5,lw=.7,label=name)
for j,(ax,title,yl) in enumerate(zip(a[0],['Radial marginal','Angular moment'],[r'$g_d^{\mathrm{rad}}(r)$',r'$g_{P_2}(r)$'])):
 ax.set_xlim(.6,5.5);ax.set_xlabel(r'$r$');ylab(ax,yl);ax.set_title(title);label(ax,f'({chr(97+j)})');ax.legend()
comparison=read('gb3d_eff.json')['by_M']
counts=sorted(int(k) for k in comparison)
for j,quantity in enumerate(('radial','angular')):
 ax=a[1,j]
 for design,marker,offset,color in [('contiguous','o',0.96,'black'),('dispersed','s',1.04,'#1647a0')]:
  rows=[comparison[str(m)][design][quantity] for m in counts]
  y=np.array([row['EFS_over_EDH_RMS_ratio'] for row in rows])
  ci=np.array([row['paired_ratio_CI95'] for row in rows])
  ax.vlines(np.array(counts)*offset,ci[:,0],ci[:,1],color=color,lw=.9)
  ax.plot(np.array(counts)*offset,y,marker,ms=4,mfc='white',color=color,label=design.capitalize())
 ax.set(xscale='log',xlabel=r'$M$',title=('Radial error' if j==0 else 'Joint error'))
 ylab(ax,'RMS ratio');ax.axhline(1,color='.5',lw=.7,ls='--')
 ax.legend(loc='upper left');label(ax,'(c)' if j==0 else '(d)')
save(f,4)
# Figure 4 includes the already computed GB extended grid.
f,a=panels(2,2)
for i,(path,title) in enumerate([('dimer_eff.json','Dimers'),('gb_eff.json','Gay–Berne')]):
 d=read(path);rows=d['summaries']
 if i==1:rows=rows+read('gb_large.json')['summaries']
 for j,q in enumerate(['radial','joint']):
  ax=a[i,j]
  for method,col,mark in [('DH','black','o'),('FS','#1647a0','s')]:
   for mode,fill,fac in [('dispersed_without_replacement',False,.975),('contiguous',True,1.025)]:
    sel=[x for x in rows if x['quantity']==q and x['method']==method and (x['protocol']==mode or (mode=='contiguous' and x['protocol'].startswith('contiguous')))]
    ms=sorted(set(x['M'] for x in sel));ys=[max(x['conditional_q95_CI95'][1] for x in sel if x['M']==m) for m in ms]
    ax.plot(np.array(ms)*fac,ys,linestyle='none',marker=mark,color=col,mfc=col if fill else 'none',ms=3.6,mew=.85,label=('EDH' if method=='DH' else 'EFS')+(' contiguous' if fill else ' dispersed'))
  ax.axhline(.005 if j==0 else .02,color='#ae2633',ls='--',lw=.9);ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel(r'$M$');ylab(ax,r'$U_{0.95}$');ax.set_title(title+' / '+('radial' if j==0 else 'joint'));label(ax,f'({chr(97+2*i+j)})');ax.legend(fontsize=8.4,loc='upper right')
save(f,"S2")
# Figure 5. Primary 3D RMS comparison.
d=read('gb3d_eff.json');ms=sorted(map(int,d['by_M']))
f,a=panels(2,2)
for i,q in enumerate(['radial','angular']):
 for j,mode in enumerate(['contiguous','dispersed']):
  ax=a[i,j]
  for name,col,mark in [('EDH','black','o'),('EFS','#1647a0','s')]:
   z=[d['by_M'][str(m)][mode][q][name] for m in ms];y=np.array([v['RMS'] for v in z]);ci=np.array([v['conditional_RMS_CI95'] for v in z]);ax.errorbar(ms,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt=mark,color=col,mfc='none',ms=3.6,mew=.85,capsize=1.5,lw=.8,label=name)
  ax.axhline(.005 if i==0 else .02,color='#ae2633',ls='--',lw=.9,label='Target');ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel(r'$M$');ylab(ax,r'$\mathrm{RMS}(\varepsilon)$');ax.set_title(('Radial' if i==0 else 'Joint')+' / '+mode);label(ax,f'({chr(97+2*i+j)})');ax.legend(loc='upper right')
save(f,"S3")
