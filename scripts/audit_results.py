"""Audit six final result files against their selected training recipes."""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path


def main():
    # Resolve relative dataset and artifact paths from the project, even when
    # invoked by an absolute path from PowerShell through WSL.
    import os
    os.chdir(Path(__file__).resolve().parents[1])
    code_hash = hashlib.sha256()
    for source in [Path('tune.py'), Path('test.py')] + sorted(Path('daisy').rglob('*.py')):
        code_hash.update(source.as_posix().encode())
        code_hash.update(source.read_bytes())
    root=Path('artifacts/verified')
    checks=[]
    summaries=[]
    best_rows=[]
    for model in ['mf','neumf','simplex']:
        for dataset in ['ml-50k','yelp-200k']:
            run=root/f'{model}_{dataset}'
            best=json.loads((run/'best_config.json').read_text())
            selection=json.loads((run/'selection.json').read_text())
            assert selection['tuning_complete'],f'Incomplete tuning: {model}/{dataset}'
            trial_recipe=json.loads((run/'trial_configs'/f"{selection['trial']:03d}.json").read_text())
            assert best==trial_recipe,f'Selected recipe mismatch: {model}/{dataset}'
            with Path(f'tune_res/best_params_BPR_{model}_{dataset}_origin_tsbr.csv').open() as source:
                saved_rows=list(csv.DictReader(source))
            assert len(saved_rows)==1
            for key,value in saved_rows[0].items():
                expected=selection['validation_ndcg'] if key=='ndcg' else best[key]
                assert float(value)==float(expected),(model,dataset,'saved CSV',key)
            result=root/'results'/dataset/f'{model}_kpi_results.csv'
            manifest=json.loads(result.with_name(f'{model}_manifest.json').read_text())
            assert manifest['status']=='complete',(model,dataset,'incomplete final test')
            assert manifest['provenance']['code_sha256']==code_hash.hexdigest(),(model,dataset,'current code changed')
            data_file=Path(manifest['provenance']['dataset_file'])
            assert hashlib.sha256(data_file.read_bytes()).hexdigest()==manifest['provenance']['dataset_sha256'],(model,dataset,'current data changed')
            # The final recipe adds reporting metrics; all training and split settings are exact.
            for key,value in best.items():
                if key not in ['metrics','res_path']:
                    assert manifest['config'][key]==value,(model,dataset,key,value,manifest['config'][key])
            assert manifest['provenance']['platform'].startswith('Linux')
            provenance=json.loads((run/'tuning_provenance.json').read_text())
            assert provenance['dataset_sha256']==manifest['provenance']['dataset_sha256']
            assert provenance['code_sha256']==manifest['provenance']['code_sha256']
            for package in ['torch','numpy','pandas','scipy','optuna']:
                assert provenance[package]==manifest['provenance'][package],(model,dataset,package)
            with result.open() as source:
                rows=list(csv.DictReader(source))
            metrics={row['KPI@K']:float(row['10']) for row in rows}
            assert all(math.isfinite(value) for value in metrics.values())
            assert all(0<=value<=1+1e-8 for key,value in metrics.items() if key!='Entropy')
            assert metrics['Entropy']>=0
            conventional=Path('res')/dataset/best['prepro']/best['test_method']
            conventional.mkdir(parents=True,exist_ok=True)
            prefix=f"{best['loss_type']}_{model}_with_{best['sample_ratio']}{best['sample_method']}"
            shutil.copyfile(result,conventional/f'{prefix}_kpi_results.csv')
            shutil.copyfile(result.with_name(f'{model}_manifest.json'),conventional/f'{prefix}_manifest.json')
            summaries.append(dict(model=model,dataset=dataset,validation_ndcg=selection['validation_ndcg'],
                                  selected_trial=selection['trial'],**metrics))
            with (run/'trials.csv').open() as source:
                trials=list(csv.DictReader(source))
            assert len([trial for trial in trials if trial['state']=='COMPLETE'])==selection['requested_trials']
            best_rows.append(dict(model=model,dataset=dataset,selected_trial=selection['trial'],
                                  validation_ndcg=selection['validation_ndcg'],**{
                                      key:best[key] for key in ['num_ng','factors','batch_size','lr','reg_1','reg_2','epochs','optimizer','init_method','early_stop']},
                                  **{key:best[key] for key in ['num_layers','dropout','margin','negative_weight','gamma'] if key in best}))
            checks.append(dict(model=model,dataset=dataset,result=str(result),status='passed',
                              sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
                              matched_training_recipe=True,selected_trial_recipe_match=True,
                              current_code_match=True,current_data_match=True,
                              saved_csv_match=True,package_versions_match=True,
                              completed_trials=selection['complete_trials']))
    for filename,rows in [('final_summary.csv',summaries),('selected_parameters.csv',best_rows)]:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        with (root/filename).open('w',newline='') as target:
            writer=csv.DictWriter(target,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (root/'audit.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2)+'\n')
    print('All six final results passed recipe, provenance, Linux and trial-completion checks.')


if __name__=='__main__':
    main()
