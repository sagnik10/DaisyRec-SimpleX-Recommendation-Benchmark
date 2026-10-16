"""Write a source-backed Markdown experiment report after the final audit."""
import csv
import json
from datetime import datetime
from pathlib import Path


def table(headers,rows):
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    lines+=['| '+' | '.join(map(str,row))+' |' for row in rows]
    return '\n'.join(lines)


def main():
    root=Path('artifacts/verified')
    audit=json.loads((root/'audit.json').read_text())
    assert audit['status']=='passed' and len(audit['checks'])==6
    with (root/'final_summary.csv').open() as source:
        results=list(csv.DictReader(source))
    with (root/'selected_parameters.csv').open() as source:
        parameters=list(csv.DictReader(source))
    stats=json.loads(Path('artifacts/dataset_statistics.json').read_text())
    names={'mf':'MF','neumf':'NeuMF','simplex':'SimpleX'}
    lines=['# Verified DaisyRec recommendation experiments','',
           f"Prepared on {datetime.now().strftime('%d %B %Y')}. This report documents the corrected code, supplied datasets, complete tuning evidence and six final tests performed on Ubuntu Linux. Every final training recipe matches its selected configuration and passed the automated audit.",'',
           '## Purpose and implementation','',
           'MF and NeuMF are the baseline models. SimpleX combines a learned user embedding with a projected mean of that user\'s training-history item embeddings, with a fixed combination weight of 0.5. It scores by cosine similarity and trains with Cosine Contrastive Loss. Its history excludes held-out interactions during fitting. LightGCN retains DaisyRec\'s normalized bipartite graph propagation; graph construction occurs before model creation and is covered by behavioral checks. [Implementation and verification](IMPLEMENTATION.md) provides the equations, code changes and source references.','',
           '## Dataset origin and statistics','',
           'MovieLens is a seed-42 sample of [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/). A fresh source download reproduced the supplied 50,000-row subset exactly, including row order. Yelp comes from the [provided Kaggle Yelp source](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data), converted to review interactions, sampled to an initial 200,000 rows with seed 42, then filtered once using user/item counts of at least five in that initial sample. The exact historical Yelp release was not recorded. [Dataset documentation](DATASETS.md) distinguishes verified measurements from this recorded preparation procedure.','',
           table(['Dataset','Raw rows','Users','Items','Processed interactions','Sparsity'],
                 [[s['dataset'],f"{s['raw_interactions']:,}",f"{s['users']:,}",f"{s['items']:,}",f"{s['interactions']:,}",f"{s['sparsity_percent']:.6f}%"] for s in stats]),'',
           'Yelp has 914 duplicate user-item pairs. DaisyRec keeps the last occurrence in file order before sorting by timestamp. All ratings become implicit positives, with no rating threshold or further count filter. Sparsity is `100 * (1 - interactions / (users * items))` after duplicate removal.','',
           '## Protocol and environment','',
           'Global chronological splitting holds out the latest 20% of processed interactions for test and the latest 10% of the remaining training portion for validation. Tuning therefore fits on approximately 72%, validates on 8% and preserves 20% for test. Final training uses the full 80% after selection. Training histories and graphs are rebuilt from each actual fitting split.','',
           table(['Dataset','Fitting','Validation','Final training','Test','Test users without training history'],
                 [[s['dataset'],s['fitting_interactions'],s['validation_interactions'],s['final_train_interactions'],s['test_interactions'],s['test_cold_users']] for s in stats]),'',
           'Evaluation ranks 1,000 sampled candidates per user, includes ground-truth items, excludes training items from candidate negatives and retains DaisyRec\'s sampling with replacement. The global time split includes cold users. Results describe this sampled, partly cold-start protocol rather than full-catalog ranking.','',
           'The verified environment was Ubuntu 24.04 under WSL2 with Python 3.12.3 and CPU PyTorch 2.5.1, pandas 2.2.3, NumPy 1.26.4, SciPy 1.14.1 and Optuna 4.1.0. Pandas imported successfully on Linux, package checks passed and five behavioral regression checks passed. See [environment metadata](../artifacts/environment.json), [verification output](../artifacts/linux_verification.log) and the [package lock](../artifacts/requirements-linux.lock.txt). Colab setup is documented in the README; Colab was not the recorded execution platform.','',
           '## Tuning evidence','',
           'All six pairs were freshly tuned using validation NDCG@10, seed 2022, the historical search ranges, four completed trials per model on MovieLens and eight per model on Yelp: 36 completed trials in total. MF uses 20 epochs; NeuMF and SimpleX use 30. Four negatives per positive are retained. The common training settings, fixed parameters and full search spaces are available in `configs/tuning/` and the complete selected recipes.','',
           'The configured Optuna TPE sampler has ten startup trials by default. These four/eight-trial studies remain in the random startup phase; the experiments should not be described as completed adaptive Bayesian searches. Their short budgets and single seed limit the strength of model comparisons.','',
           'Every trial\'s validation NDCG, state, parameters and duration is saved in its `trials.csv`, alongside an SQLite study and a complete `trial_configs/` recipe. [Consolidated trial evidence](../artifacts/verified/tuning_trials.csv) and [selected parameters](../artifacts/verified/selected_parameters.csv) retain the numerical values at full precision. Plot trial labels are one-based, while saved Optuna trial IDs are zero-based.','',
           '![Validation NDCG for every tuning trial](../artifacts/verified/plots/ndcg_vs_trial.png)','',
           '![Cumulative best validation NDCG](../artifacts/verified/plots/cumulative_best.png)','',
           table(['Model','Dataset','Selected trial ID','Validation NDCG@10','Factors','Batch','Learning rate','L1','L2','Epochs'],
                 [[names[r['model']],r['dataset'],r['selected_trial'],f"{float(r['validation_ndcg']):.6f}",r['factors'],r['batch_size'],r['lr'],r['reg_1'],r['reg_2'],r['epochs']] for r in parameters]),'',
           table(['Model','Dataset','Layers','Dropout','Margin','Negative weight','Gamma'],
                 [[names[r['model']],r['dataset'],r.get('num_layers') or '-',r.get('dropout') or '-',r.get('margin') or '-',r.get('negative_weight') or '-',r.get('gamma') or '-'] for r in parameters if r['model']!='mf']),'',
           '## Final tests','',
           'The six final tests load their complete selected training recipes. Common reporting metrics are saved for all pairs. The audit checks selected settings, completed trial budgets, source/data hashes and Linux provenance. [Audit evidence](../artifacts/verified/audit.json) and [the full result table](../artifacts/verified/final_summary.csv) make these checks reviewable.']
    for dataset in ['ml-50k','yelp-200k']:
        rows=[row for row in results if row['dataset']==dataset]
        lines+=['',f'### {dataset}','',
                 table(['Model','Recall@10','NDCG@10','MRR@10','Hit@10','Precision@10','Coverage@10','Gini@10','Entropy@10'],
                       [[names[r['model']]]+[f"{float(r[key]):.6f}" for key in ['Recall','NDCG','MRR','Hit Ratio','Precision','Coverage','Gini','Entropy']] for r in rows]),'']
        ordered=sorted(rows,key=lambda r:float(r['NDCG']),reverse=True)
        lines.append(f"In this run, {names[ordered[0]['model']]} has the highest test NDCG@10 ({float(ordered[0]['NDCG']):.6f}), followed by {names[ordered[1]['model']]} ({float(ordered[1]['NDCG']):.6f}) and {names[ordered[2]['model']]} ({float(ordered[2]['NDCG']):.6f}). These are single-seed measurements; no statistical significance claim is supported.")
        lines+=['',table(['Model','Result file'],[[names[r['model']],f"[{r['model']}_kpi_results.csv](../artifacts/verified/results/{dataset}/{r['model']}_kpi_results.csv)"] for r in rows])]
    lines+=['','## Interpretation and limits','',
            'Coverage is the fraction of catalog items appearing in recommendations. Entropy is the exposure-distribution entropy in bits. The retained Gini implementation uses only items that received at least one recommendation, excluding zero-exposure catalog items. Coverage, Gini and entropy therefore measure different aspects of exposure; none alone establishes fairness, user satisfaction or within-list diversity. Higher Gini indicates more concentration within the covered item set.','',
            'No repeated-seed uncertainty interval or history ablation was run. The comparisons support the measured outcomes under this protocol, but cannot establish that a particular architecture component caused an improvement. The sparse datasets, cold users, sampled candidates, short startup-phase searches and limited negative budget all constrain generalization.','',
            'The original report and historical outputs contain inconsistencies: the Yelp label and final interaction count were inaccurate, no-filter wording omitted preparation filtering, SimpleX\'s actual objective was CCL, the legacy tuning cutoff was 50 rather than 10, and Yelp SimpleX testing used margin 0.9 rather than the selected 0.7. Later historical logs already contained an eight-trial completion, following earlier interrupted runs. Historical trial-level scores were not fully retained; fresh studies provide the requested complete evidence. The legacy files are preserved under `artifacts/legacy/`.','',
            '## Run commands','',
            'From the activated Linux environment in the project root:','',
            '```bash','python scripts/run_experiments.py --stage all --jobs 2','```','',
            'To repeat only the six final tests with their selected configurations:','',
            '```bash','python scripts/run_experiments.py --stage test --jobs 2','```','',
            'The [README](../README.md) provides setup, dataset preparation, Colab and PowerShell-to-WSL commands.']
    Path('docs/EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('Wrote docs/EXPERIMENT_REPORT.md from audited results.')


if __name__=='__main__':
    main()
