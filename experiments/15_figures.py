"""Generate the six manuscript figures at the journal text width (5.42 in)."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,
                     'legend.fontsize':6.3,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'figure.dpi':400})
F='figures/'
W=5.42

img=np.load('data/cache/thermal.npy'); df=pd.read_parquet('data/cache/labels.parquet')
y=df['Porosity Label'].values; L=df['Layer'].values

def ent(a):
    h,_=np.histogram(a,bins=32,range=(0,2000)); p=h/h.sum(); return -(p[p>0]*np.log(p[p>0])).sum()
E=np.array([ent(img[i]) for i in range(len(img))])

# ---- Fig 1 ----
pos=np.where(y==1)[0]; hi=pos[np.argsort(E[pos])[:5]]
free=np.where(y==0)[0]; safe=free[np.argsort(E[free])[-30]]
fig=plt.figure(figsize=(W,3.35)); gs=GridSpec(2,2,height_ratios=[1,1.05],hspace=0.45,wspace=0.28)
for k,(i,t) in enumerate(((safe,'(a) Pore-free, high entropy'),(hi[0],'(b) Pore, low entropy'))):
    ax=fig.add_subplot(gs[0,k]); im=ax.imshow(img[i],cmap='inferno',vmin=600,vmax=2000); ax.set_xticks([]);ax.set_yticks([])
    ax.set_title(t)
    cb=plt.colorbar(im,ax=ax,fraction=0.05,pad=0.03); cb.set_label('$^\\circ$C'); cb.ax.tick_params(labelsize=5.5)
ax=fig.add_subplot(gs[1,:])
pl=pd.Series(y).groupby(L).agg(['sum','count'])
ax.bar(pl.index,pl['sum'],width=1.0,color='#c0392b',label='Pore frames')
ax.bar(pl.index,pl['count']-pl['sum'],width=1.0,bottom=pl['sum'],color='#aab7b8',label='Pore-free')
ax.set(xlabel='Layer index',ylabel='Frames',xlim=(0.5,60.5),ylim=(0,42),title='(c) 1564 frames, 71 pore frames (4.5%)')
ax.legend(loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(0.5,1.02))
ax.grid(axis='y',alpha=0.25)
plt.savefig(F+'fig1_dataset.png',bbox_inches='tight')

# ---- Fig 2 ----
bench=pd.read_csv('results/results_bench_suite.csv'); b=bench[bench.feat=='base']
models=['LogReg','RF','XGB','LGBM','CatBoost','TabPFN']
leak=pd.read_csv('results/results_leakage.csv'); leak=leak[leak.model=='XGB'].reset_index(drop=True)
fig,axs=plt.subplots(1,2,figsize=(W,2.15),gridspec_kw={'width_ratios':[1.5,1],'wspace':0.38})
ax=axs[0]; x=np.arange(len(models)); w=0.38
g5=[b[(b.model==m)&(b.protocol=='group5')].pr.values[0] for m in models]
fw=[b[(b.model==m)&(b.protocol=='forward')].pr.values[0] for m in models]
ax.bar(x-w/2,g5,w,label='Layer-grouped',color='#2471a3'); ax.bar(x+w/2,fw,w,label='Forward',color='#c0392b')
ax.set_xticks(x); ax.set_xticklabels(models,rotation=30,ha='right'); ax.set(ylim=(0.6,1.0),ylabel='PR-AUC',title='(a) Ranking depends on protocol')
ax.legend(loc='lower left'); ax.grid(axis='y',alpha=0.3)
ax=axs[1]; xs=np.arange(3)
ax.plot(xs,leak.acc.values,'o-',color='#7f8c8d',label='Accuracy')
ax.plot(xs,leak.pr.values,'s-',color='#16a085',label='PR-AUC')
ax.set_xticks(xs); ax.set_xticklabels(['Rand.','Grouped','Forward'],fontsize=6)
ax.set(ylim=(0.85,1.02),title='(b) Accuracy is blind'); ax.legend(loc='center left'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(F+'fig2_protocol.png',bbox_inches='tight')

# ---- Fig 3 ----
combo=pd.read_csv('results/results_combo.csv'); abl=pd.read_csv('results/results_ablation.csv')
fig,axs=plt.subplots(1,2,figsize=(W,2.2),gridspec_kw={'width_ratios':[1.35,1],'wspace':0.36})
ax=axs[0]; x=np.arange(len(combo)); w=0.38
ax.bar(x-w/2,combo.cat_forward,w,yerr=combo.cat_forward_sd,capsize=1.8,label='Forward',color='#c0392b')
ax.bar(x+w/2,combo.cat_group5,w,label='Grouped',color='#2471a3')
ax.set_xticks(x); ax.set_xticklabels(['Base','+Ent','+Grad','+E+G','+E+G\n+Area','Full 39'],fontsize=6)
ax.set(ylim=(0.8,1.0),ylabel='PR-AUC',title='(a) Feature combinations (CatBoost)'); ax.legend(loc='lower right'); ax.grid(axis='y',alpha=0.3)
ax=axs[1]; xx=np.arange(5)
ax.bar(xx-w/2,abl.forward-0.845,w,label='Forward',color='#c0392b'); ax.bar(xx+w/2,abl.group5-0.938,w,label='Grouped',color='#2471a3')
ax.axhline(0,color='k',lw=0.6); ax.set_xticks(xx); ax.set_xticklabels(['Ent','Grad','Topo','Area','Moran'],rotation=25,ha='right',fontsize=6)
ax.set(ylabel='$\\Delta$PR-AUC',title='(b) Which groups help'); ax.legend(); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(F+'fig3_ablation.png',bbox_inches='tight')

# ---- Fig 4 ----
ms=pd.read_csv('results/results_multiseed.csv'); ex=pd.read_csv('results/results_exclude_firstlayers.csv'); ll=pd.read_csv('results/results_layerlevel.csv')
fig=plt.figure(figsize=(W,3.3)); gs=GridSpec(2,2,height_ratios=[1,1],hspace=0.55,wspace=0.35)
ax=fig.add_subplot(gs[0,0]); a=ms[ms.protocol=='forward']; mm=['XGB','Cat','RF']; x=np.arange(3); w=0.36
b1=[a[(a.model==m)&(a.feat=='base')]['mean'].iloc[0] for m in mm]; b2=[a[(a.model==m)&(a.feat=='entropy+gradient')]['mean'].iloc[0] for m in mm]
e1=[a[(a.model==m)&(a.feat=='base')]['sd'].iloc[0] for m in mm]; e2=[a[(a.model==m)&(a.feat=='entropy+gradient')]['sd'].iloc[0] for m in mm]
ax.bar(x-w/2,b1,w,yerr=e1,capsize=1.8,label='Base',color='#aab7b8'); ax.bar(x+w/2,b2,w,yerr=e2,capsize=1.8,label='+E+G',color='#16a085')
ax.set_xticks(x); ax.set_xticklabels(mm); ax.set(ylim=(0.6,1.05),ylabel='PR-AUC',title='(a) 10 seeds, forward'); ax.legend(loc='lower right'); ax.grid(axis='y',alpha=0.3)
ax=fig.add_subplot(gs[0,1]); s=ex[ex.protocol=='forward']
b1=[s[(s.model==m)&(s.feat=='base')]['pr'].iloc[0] for m in mm]; b2=[s[(s.model==m)&(s.feat=='base+entropy+gradient')]['pr'].iloc[0] for m in mm]
ax.bar(x-w/2,b1,w,label='Base',color='#aab7b8'); ax.bar(x+w/2,b2,w,label='+E+G',color='#16a085')
ax.set_xticks(x); ax.set_xticklabels(mm); ax.set(ylim=(0.7,1.02),ylabel='PR-AUC',title='(b) Layers 1-2 excluded'); ax.legend(loc='lower right'); ax.grid(axis='y',alpha=0.3)
ax=fig.add_subplot(gs[1,:]); q=ll[ll['agg']=='max']
keys=list(dict.fromkeys(zip(q.model,q.feat)))
labs=[f'{m}\n{"base" if f == "base" else "+EG"}' for m,f in keys]
gv=[q[(q.model==m)&(q.feat==f)&(q.protocol=='group5')]['pr'].iloc[0] for m,f in keys]
fv=[q[(q.model==m)&(q.feat==f)&(q.protocol=='forward')]['pr'].iloc[0] for m,f in keys]
xx=np.arange(len(keys)); ax.bar(xx-w/2,gv,w,label='Grouped',color='#2471a3'); ax.bar(xx+w/2,fv,w,label='Forward',color='#c0392b')
ax.set_xticks(xx); ax.set_xticklabels(labs,fontsize=6); ax.set(ylim=(0.9,1.02),ylabel='Layer-level PR-AUC',title='(c) Layer-level decisions'); ax.legend(loc='lower right',ncol=2); ax.grid(axis='y',alpha=0.3)
plt.savefig(F+'fig4_robustness.png',bbox_inches='tight')

# ---- Fig 5 ----
imp=pd.read_csv('results/shap_importance.csv',index_col=0).iloc[:,0].sort_values(); bins=pd.read_csv('results/shap_bins.csv')
fig,axs=plt.subplots(1,2,figsize=(W,2.35),gridspec_kw={'width_ratios':[1,1],'wspace':0.42})
ax=axs[0]; top=imp.tail(9)
cols=['#16a085' if n in ('hist_entropy','grad_mean','grad_p99') else '#95a5a6' for n in top.index]
ax.barh(top.index,top.values,color=cols); ax.set(xlabel='mean |SHAP|',title='(a) Global importance'); ax.grid(axis='x',alpha=0.3)
ax=axs[1]
for f,c,lbl,off in (('hist_entropy','#16a085','Entropy',-0.09),('grad_mean','#c0392b','Gradient',0.09),('std','#2471a3','Std. dev.',0)):
    d=bins[bins.feature==f].reset_index(drop=True); ax.plot(np.arange(1,len(d)+1)+off,d.pos_rate*100,'o-',color=c,label=lbl,ms=3)
ax.set_xticks(range(1,6)); ax.set(xlabel='Quintile',ylabel='Pore rate (%)',title='(b) Risk at extremes'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(F+'fig5_shap.png',bbox_inches='tight')

# ---- Fig 6 ----
cost=pd.read_csv('results/results_cost.csv'); sel=pd.read_csv('results/results_selective.csv')
fig,axs=plt.subplots(1,2,figsize=(W,2.1),gridspec_kw={'wspace':0.38})
ax=axs[0]; ax.plot(cost.R,cost.recall,'o-',ms=3,color='#c0392b',label='Recall'); ax.plot(cost.R,cost.precision,'s--',ms=3,color='#2471a3',label='Precision')
ax.set_xscale('log'); ax.set(xlabel='Cost ratio (miss:FA)',ylabel='Score',ylim=(0.5,1.05),title='(a) Cost-sensitive'); ax.legend(); ax.grid(alpha=0.3)
ax=axs[1]
for f,c,lbl in (('base','#aab7b8','Base'),('entropy+gradient','#16a085','+E+G')):
    d=sel[sel.feat==f]; ax.plot(d.coverage,d.miss_rate,'o-',ms=3,color=c,label=lbl)
ax.axvline(0.9,ls=':',c='gray'); ax.set(xlabel='Automatic coverage',ylabel='Miss rate',ylim=(-0.02,0.3),title='(b) Abstention'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(F+'fig6_decision.png',bbox_inches='tight')
print('done')

import os, sys
