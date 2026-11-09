"""Build the six Phase-I CDM figures from saved summaries; no model fitting.

Run this script from any directory. Input snapshots and provenance are in
../tables/experiments/phase1_cdm_six_figures/. Outputs are vector PDFs.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd

THESIS = Path(__file__).resolve().parents[1]
DATA = THESIS / 'tables/experiments/phase1_cdm_six_figures'
OUT = THESIS / 'images/experiments/phase1_cdm_six_figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.titlesize':9.5,
 'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
 'grid.alpha':.18,'axes.axisbelow':True,'lines.linewidth':1.6,'lines.markersize':4,
 'pdf.fonttype':42,'savefig.bbox':'tight','savefig.pad_inches':.06})
BLUE,ORANGE,GREEN,PURPLE='#0072B2','#D55E00','#009E73','#8B5AA6'
def read(name): return pd.read_csv(DATA/name)
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
def xticks(ax,vals):
 ax.set_xticks(vals);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:g}k' if x>=1000 else f'{x:g}'))
 ax.minorticks_off()

# 1. Full-CDM recovery: keep observation and item scaling separate.
fig,axs=plt.subplots(1,2,figsize=(6.3,2.7),layout='constrained')
for ax,name,xcol,title,color in zip(axs,
 ['full_cdm_observation_summary.csv','full_cdm_item_summary.csv'],
 ['Number of observations','Number of items'],
 ['(a) More observations; 5 items','(b) More items; 50,000 observations'],[BLUE,ORANGE]):
 t=read(name);ax.errorbar(t[xcol],t.Mean_parameter_RMSE,yerr=t.RMSE_std,fmt='o-',capsize=3,color=color)
 ax.set(xscale='log',xlabel=xcol,ylabel='Interaction RMSE',title=title)
 xticks(ax,t[xcol])
save(fig,'01_full_cdm_recovery')

# 2. Cross-model misspecification, with independent-repeat SDs.
fig,axs=plt.subplots(1,2,figsize=(6.3,2.7),layout='constrained')
for ax,name,xcol,title in zip(axs,['cdm_btl_observations.csv','cdm_btl_strength.csv'],
 ['Observations','Scale'],['(a) Increasing observations','(b) Increasing interaction scale']):
 t=read(name)
 for n,color in [(5,BLUE),(10,ORANGE)]:
  g=t[t.Items==n];ax.errorbar(g[xcol],g.Gap,yerr=g['Gap SD'],fmt='o-',capsize=3,color=color,label=f'{n} items')
 ax.axhline(0,color='.45',lw=.8,ls='--');ax.set(xlabel='Observations' if xcol=='Observations' else 'Interaction scale',ylabel='BTL test NLL - true CDM NLL',title=title)
 if xcol=='Observations':ax.set_xscale('log');xticks(ax,sorted(t[xcol].unique()))
 ax.legend()
save(fig,'02_cdm_to_btl_misspecification')

# 3. Common test observations; selection performed on validation data.
t=read('unified_comparison.csv');fig,axs=plt.subplots(1,2,figsize=(6.3,2.8),layout='constrained',sharey=True)
for ax,(_,row) in zip(axs,t.iterrows()):
 vals=[row['BTL test NLL'],row['Full CDM test NLL'],row['Low-rank test NLL']]
 rank=int(row['Selected low-rank rank']);labels=['BTL','Full CDM',f'Rank {rank}'+('*' if row.Source=='CDM' else '')]
 for i,(value,color) in enumerate(zip(vals,[BLUE,ORANGE,GREEN])):
  ax.scatter(i,value,color=color,s=40,zorder=3);ax.annotate(f'{value:.5f}',(i,value),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
 ax.set(xticks=range(3),xticklabels=labels,xlim=(-.5,2.5),ylim=(.82,.9),title=f'{row.Source}-generated choices',ylabel='Held-out mean NLL')
 ax.grid(axis='x',visible=False)
fig.supxlabel('* Selected rank-5 fit reached the epoch limit',fontsize=8,color=ORANGE)
save(fig,'03_unified_model_comparison')

# 4. Resource cost: trainable entries, optimization time, sampled process RSS.
t=read('resource_scaling.csv');fig,axs=plt.subplots(1,3,figsize=(6.3,2.8),layout='constrained')
for model,color in zip(['Full','Rank 1','Rank 2','Rank 3'],[BLUE,ORANGE,GREEN,PURPLE]):
 g=t[t.model==model].sort_values('n')
 for ax,col,err in zip(axs,['parameters','seconds_mean','peak_rss_mean'],[None,'seconds_std','peak_rss_std']):
  ax.errorbar(g.n,g[col],yerr=None if err is None else g[err],fmt='o-',capsize=2,color=color,label=model)
for ax,title,ylabel in zip(axs,['(a) Trainable entries','(b) 1,000 Adam updates','(c) Sampled peak process RSS'],['Entries','Training time (seconds)','RSS (MiB)']):
 ax.set(xscale='log',xlabel='Items',title=title,ylabel=ylabel);xticks(ax,[250,1000,5000])
axs[0].set_yscale('log');axs[1].set_yscale('log')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=False)
save(fig,'04_computational_scaling')

# 5. Six-item paper-inspired synthetic recovery, using the chosen 100k grid.
t=read('paper_recovery.csv');fig,axs=plt.subplots(1,2,figsize=(6.3,2.7),layout='constrained')
x=t['Number of observations'].to_numpy();y=t.Mean_squared_parameter_error.to_numpy()
axs[0].errorbar(x,y,yerr=t.Squared_parameter_error_std,fmt='o-',capsize=2,color=BLUE,label='Mean, 10 runs')
axs[0].plot(x,y[0]*x[0]/x,'--',color='.5',label=r'$m^{-1}$ guide')
axs[0].set(xscale='log',yscale='log',xlabel='Observations',ylabel='Squared Frobenius error',title='(a) Interaction recovery');axs[0].legend()
axs[1].plot(x,t.Mean_CDM_runtime_seconds,'o-',color=ORANGE)
axs[1].set(xscale='log',xlabel='Observations',ylabel='Mean fitting time (seconds)',title='(b) Optimization cost')
for ax in axs:xticks(ax,[1000,10000,100000])
save(fig,'05_paper_synthetic_recovery')

# 6. IIA pilot: rates conditional on computed tests, accompanied by availability.
t=read('iia_rejections.csv');sep=read('iia_separation.csv')
fig,axs=plt.subplots(2,2,figsize=(6.3,4.7),layout='constrained')
for ax,model in zip(axs.flat,['MNL','CDM','General choice']):
 for test,color,label in [('MNL vs CDM',BLUE,'CDM alternative'),('MNL vs Universal',ORANGE,'Universal alternative')]:
  g=t[(t['Generating model']==model)&(t.Test==test)].sort_values('Observations')
  rate=g['Rejection rate among computed tests'];lo=g['95% CI lower'];hi=g['95% CI upper']
  ax.errorbar(g.Observations,rate,yerr=np.vstack([np.maximum(0,rate-lo),np.maximum(0,hi-rate)]),fmt='o-',capsize=2,color=color,label=label)
 ax.axhline(.05,color='.5',ls='--',lw=.9);ax.set(xscale='log',ylim=(-.04,1.04),xlabel='Observations',ylabel='Rejection proportion',title=f'{model}-generated data');xticks(ax,[100,400,1000,2000]);ax.yaxis.set_major_formatter(PercentFormatter(1))
ax=axs[1,1];ax.plot(sep.Observations,sep.Separation_detected/sep.Datasets,'o-',color=PURPLE,label='Separation detected')
g=t[(t['Generating model']=='CDM')&(t.Test=='MNL vs CDM')].sort_values('Observations');ax.plot(g.Observations,g['Computed tests']/g['Attempted datasets'],'s-',color=BLUE,label='CDM test available')
ax.set(xscale='log',ylim=(-.04,1.04),xlabel='Observations',ylabel='Fraction of 10 datasets',title='CDM-generated fit diagnostics');xticks(ax,[100,400,1000,2000]);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc='center right',fontsize=7)
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
save(fig,'06_iia_testing_pilot')
print('Created exactly six CDM summary figures from saved results.')
