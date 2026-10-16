import ast, csv, json, math, shutil
from pathlib import Path
root = Path('.')
legacy = root / 'artifacts/legacy'
legacy.mkdir(parents=True, exist_ok=True)
for directory in ['tune_res', 'res']:
    if not (legacy / directory).exists():
        shutil.copytree(root / directory, legacy / directory)
configs = root / 'configs/tuning'
configs.mkdir(parents=True, exist_ok=True)
summary = []
for algorithm in ['mf','neumf','simplex']:
    for dataset in ['ml-50k','yelp-200k']:
        filename = f'best_params_BPR_{algorithm}_{dataset}_origin_tsbr.csv'
        with (legacy / 'tune_res' / filename).open() as source:
            rows = list(csv.DictReader(source))
        if len(rows) != 1:
            raise ValueError(f'Missing best parameters: {filename}')
        best = rows[0]
        matches = []
        for logfile in (root / 'log' / algorithm).glob('*.log'):
            text = logfile.read_text(encoding='utf8', errors='replace')
            if 'get the best ndcg' not in text:
                continue
            first = text.splitlines()[0]
            try:
                recipe = ast.literal_eval(first[first.index('{'):])
                result = ast.literal_eval(text.rsplit('with params: ',1)[1].splitlines()[0])
            except (ValueError, SyntaxError):
                continue
            if recipe['dataset'] != dataset:
                continue
            candidate = {**recipe, **result}
            if all(math.isclose(float(candidate[name]), float(value), rel_tol=1e-10, abs_tol=1e-10)
                   for name, value in best.items() if name != 'ndcg'):
                matches.append((logfile.name, candidate))
        if not matches:
            raise ValueError(f'No complete source log for {filename}')
        filename_log, recipe = matches[-1]
        (legacy / f'{algorithm}_{dataset}_best_config.json').write_text(json.dumps(recipe, indent=2))
        recipe.update({'topk':10, 'optimization_metric':'ndcg', 'gpu':'',
                       'num_workers':0, 'torch_threads':2,
                       'run_dir':'artifacts/verified', 'study_name':None, 'study_storage':None})
        if algorithm == 'simplex':
            recipe['gamma'] = 0.5
        (configs / f'{algorithm}_{dataset}.json').write_text(json.dumps(recipe, indent=2) + '\n')
        summary.append({'model':algorithm, 'dataset':dataset, 'source_log':filename_log,
                        'trials':recipe['hyperopt_trail'], 'epochs':recipe['epochs'], 'search_space':json.loads(recipe['tune_pack'])})
(root / 'artifacts/tuning_plan.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary, indent=2))
