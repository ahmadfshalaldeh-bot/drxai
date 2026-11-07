#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'results/ordinal_risk_decision/summary_metrics.csv'
out_dirs=[ROOT/'results/figures',Path('/home/ubuntu/dr_revision/mathematics_figures'),Path('/home/ubuntu/dr_revision/mathematics_submission_figures_600dpi')]
for d in out_dirs: d.mkdir(parents=True,exist_ok=True)

df=pd.read_csv(source).set_index('decision_rule').loc[['argmax','ordinal_squared_distance_bayes']]
labels=['Argmax\n(primary)','Squared-distance\n(risk sensitivity)']
colors=['#1f5a7a','#b55b3e']
fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),dpi=600)

left_metrics=[('accuracy','Accuracy'),('balanced_accuracy','Balanced accuracy'),('macro_f1','Macro-F1'),('qwk','QWK')]
right_metrics=[('ordinal_mae','Mean absolute\ngrade error'),('within_one_grade','Within-one-grade\naccuracy'),('severe_miss_rate_ge2','Large ordinal error\nrate (|error| ≥ 2)')]

for ax, metrics, title in [(axes[0],left_metrics,'Classification and ordinal agreement'),(axes[1],right_metrics,'Grade-distance behavior')]:
    x=np.arange(len(metrics)); width=0.34
    vals_a=[df.loc['argmax',k] for k,_ in metrics]
    vals_r=[df.loc['ordinal_squared_distance_bayes',k] for k,_ in metrics]
    ax.bar(x-width/2,vals_a,width,label=labels[0],color=colors[0])
    ax.bar(x+width/2,vals_r,width,label=labels[1],color=colors[1])
    ax.set_xticks(x,[lab for _,lab in metrics],fontsize=8)
    ax.set_ylim(0,max(max(vals_a),max(vals_r))*1.28)
    ax.set_title(title,fontsize=10,fontweight='bold')
    ax.grid(axis='y',alpha=.25)
    ax.spines[['top','right']].set_visible(False)
    for xpos,v in list(zip(x-width/2,vals_a))+list(zip(x+width/2,vals_r)):
        ax.text(xpos,v+max(max(vals_a),max(vals_r))*.025,f'{v:.3f}',ha='center',va='bottom',fontsize=7,rotation=90)
axes[0].set_ylabel('Metric value')
axes[1].legend(frameon=False,fontsize=8,loc='upper right')
fig.suptitle('Prediction-level ordinal decision sensitivity on fixed fold-assigned probabilities',fontsize=12,fontweight='bold')
fig.text(0.5,0.01,'The squared-distance rule reduced errors spanning at least two grades but worsened primary classification metrics; the argmax rule remains primary.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,0.06,1,0.92])

for d in out_dirs:
    name='Figure_9_Ordinal_Risk_Decision_Comparison.png' if d.name=='mathematics_submission_figures_600dpi' else 'ordinal_risk_decision_comparison.png'
    fig.savefig(d/name,dpi=600,facecolor='white',bbox_inches='tight')
plt.close(fig)
print('generated')
