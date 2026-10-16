"""Create standalone plots directly from the persisted trial evidence."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('artifacts/verified')
    plot_dir=root/'plots'
    plot_dir.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    series={}
    for model in ['mf','neumf','simplex']:
        for dataset in ['ml-50k','yelp-200k']:
            run=root/f'{model}_{dataset}'
            with (run/'trials.csv').open() as source:
                rows=list(csv.DictReader(source))
            for row in rows:
                all_rows.append(dict(model=model,dataset=dataset,source=str(run/'trials.csv'),**row))
            completed=[row for row in rows if row['state']=='COMPLETE']
            x=np.array([int(row['trial'])+1 for row in completed])
            y=np.array([float(row['validation_ndcg']) for row in completed])
            if not len(y) or not np.isfinite(y).all():
                raise ValueError(f'Missing or non-finite scores for {model}/{dataset}')
            if any(int(row['ndcg_k'])!=10 for row in completed):
                raise ValueError('All curves must measure NDCG@10')
            series[model,dataset]=(x,y)
    fields=list(dict.fromkeys(key for row in all_rows for key in row))
    with (root/'tuning_trials.csv').open('w',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    ymax=max(y.max() for x,y in series.values())*1.25
    colors={'mf':'#245DA8','neumf':'#C96522','simplex':'#25835E'}
    labels={'mf':'MF','neumf':'NeuMF','simplex':'SimpleX'}
    for cumulative in [False,True]:
        fig,axes=plt.subplots(3,2,figsize=(10.5,9),sharey=True,layout='constrained')
        for i,model in enumerate(['mf','neumf','simplex']):
            for j,dataset in enumerate(['ml-50k','yelp-200k']):
                ax=axes[i,j]
                x,y=series[model,dataset]
                values=np.maximum.accumulate(y) if cumulative else y
                ax.plot(x,values,marker='o',color=colors[model],linewidth=1.8)
                best=int(np.argmax(y))
                ax.scatter([x[best]],[y[best]],s=85,facecolor='white',edgecolor=colors[model],linewidth=2,zorder=5)
                ax.set_title(f'{labels[model]} on {dataset} ({len(x)} completed trials)',fontsize=11)
                ax.set_ylim(0,ymax)
                ax.set_xticks(x)
                ax.grid(axis='y',alpha=.25)
                ax.spines[['top','right']].set_visible(False)
                ax.set_xlabel('Trial (one-based display)')
                if j==0:
                    ax.set_ylabel('Best validation NDCG@10' if cumulative else 'Validation NDCG@10')
                ax.text(.03,.94,f'Best: {y[best]:.4f} at trial {x[best]}',transform=ax.transAxes,va='top',fontsize=10)
        fig.suptitle('Cumulative best tuning scores' if cumulative else 'Validation scores for every tuning trial',fontsize=15)
        filename='cumulative_best' if cumulative else 'ndcg_vs_trial'
        fig.savefig(plot_dir/f'{filename}.png',dpi=180)
        fig.savefig(plot_dir/f'{filename}.svg')
        plt.close(fig)
    print(f'Plots and consolidated evidence saved to {root}')


if __name__=='__main__':
    main()
