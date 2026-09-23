"""Render every model/predictor grid from the exported predictions."""
from pathlib import Path
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'judicial_replication_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
COLORS = {'Contracts Expression':'#C45B28', 'Mixed Outcome':'#666666', 'Expands Expression':'#167BA3'}


def main():
    out=BASE/'results'
    figures=out/'figures'
    figures.mkdir(exist_ok=True)
    data=pd.read_csv(out/'predictions.csv')
    support=pd.read_csv(out/'observed_support.csv')
    validation=pd.read_csv(out/'validation_models.csv').set_index('model')
    checks=pd.read_csv(out/'prediction_validation.csv')
    assert checks.passed.all()
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.facecolor':'white'})
    index=[]
    for (model,predictor), group in data.groupby(['model','predictor'],sort=False):
        fig=plt.figure(figsize=(12,5.3))
        gs=fig.add_gridspec(2,3,height_ratios=[5,1],hspace=.12,wspace=.16)
        observed=support.loc[(support.model==model)&(support.predictor==predictor),'value'].to_numpy()
        for j,(label,color) in enumerate(COLORS.items()):
            ax=fig.add_subplot(gs[0,j])
            df=group[group.outcome==label].sort_values('grid_value')
            ax.fill_between(df.grid_value,df.lower,df.upper,color=color,alpha=.18,linewidth=0)
            ax.plot(df.grid_value,df.probability,color=color,lw=2)
            ax.set(title=label,ylim=(0,1),yticks=np.linspace(0,1,6))
            ax.grid(axis='y',alpha=.18)
            ax.tick_params(axis='x',labelbottom=False)
            if j==0:
                ax.set_ylabel('Predicted probability')
            dist=fig.add_subplot(gs[1,j],sharex=ax)
            dist.hist(observed,bins=25,color='#b8bdc3',edgecolor='white',linewidth=.3)
            dist.plot(observed,np.zeros(len(observed)), '|',color='#555555',alpha=.13,markersize=5)
            dist.set_ylabel('Count' if j==0 else '')
            dist.tick_params(labelsize=8)
            dist.set_xlim(df.grid_value.min(),df.grid_value.max())
        N=int(group.N.iloc[0])
        fig.suptitle(f'{model} | {predictor} | N = {N:,}',y=.98,fontsize=13)
        strict=bool(validation.loc[model,'all_printed_comparisons_pass'])
        status='Log comparisons pass' if strict else 'Printed-precision differences: see validation report'
        fig.text(.5,.905,'Average adjusted predictions; pointwise 95% confidence intervals.',ha='center',fontsize=10)
        fig.text(.5,.025,f'{predictor}  |  Observed support: histogram and rug  |  {status}',ha='center',fontsize=9)
        fig.subplots_adjust(top=.84,bottom=.14,left=.07,right=.98)
        stem=f'{model}__{predictor}'
        fig.savefig(figures/f'{stem}.png',dpi=180)
        plt.close(fig)
        index.append(dict(model=model,predictor=predictor,N=N,png=f'figures/{stem}.png',
                          printed_comparisons_pass=strict))
    assert len(index)==24
    pd.DataFrame(index).to_csv(out/'figure_index.csv',index=False)
    print(f'Created {len(index)} PNG figures.')


if __name__=='__main__':
    main()
