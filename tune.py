"""Resumable tuning with per-trial validation evidence and complete best recipes."""
import csv
import json
import time
from pathlib import Path
from logging import getLogger
import numpy as np
import optuna
import torch
from daisy.model.Models import RecommenderModel
from daisy.utils.loader import RawDataReader, Preprocessor
from daisy.utils.splitter import TestSplitter, ValidationSplitter
from daisy.utils.config import init_seed, init_config, init_logger, param_type_config
from daisy.utils.metrics import metrics_config
from daisy.utils.dataset import CandidatesDataset, get_dataloader
from daisy.utils.utils import (get_ur, build_candidates_set, fit_model, write_json,
                               serializable_config, experiment_provenance)


def export_study(study, run_path, base_config, names):
    rows = []
    for trial in study.trials:
        row = {'trial': trial.number, 'state': trial.state.name,
               'validation_ndcg': trial.value, 'ndcg_k': base_config['topk'],
               'duration_seconds': trial.duration.total_seconds() if trial.duration else None}
        recipe = trial.user_attrs.get('config', {})
        row.update({name: recipe.get(name) for name in names})
        rows.append(row)
    fields = ['trial', 'state', 'validation_ndcg', 'ndcg_k', 'duration_seconds'] + names
    with (run_path / 'trials.csv').open('w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if complete:
        best = study.best_trial
        write_json(run_path / 'best_config.json', best.user_attrs['config'])
        write_json(run_path / 'selection.json', {'trial': best.number,
                   'validation_ndcg': best.value, 'ndcg_k': base_config['topk'],
                   'complete_trials': len(complete), 'requested_trials': base_config['hyperopt_trail'],
                   'tuning_complete': len(complete) >= base_config['hyperopt_trail']})
        # Keep DaisyRec's conventional best-parameter CSV, at full precision.
        directory = Path('tune_res')
        directory.mkdir(exist_ok=True)
        filename = f"best_params_{base_config['loss_type']}_{base_config['algo_name']}_{base_config['dataset']}_{base_config['prepro']}_{base_config['val_method']}.csv"
        with (directory / filename).open('w', newline='', encoding='utf-8') as target:
            writer = csv.DictWriter(target, fieldnames=names + ['ndcg'])
            writer.writeheader()
            writer.writerow({**{name: best.user_attrs['config'][name] for name in names}, 'ndcg': best.value})


def main():
    config = init_config()
    if config['optimization_metric'] != 'ndcg':
        raise ValueError('This evidence export requires optimization_metric=ndcg')
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    config['logger'] = logger
    names = RecommenderModel(config['algo_name']).tunable_param_names
    space = json.loads(config['tune_pack'])
    unknown = set(space) - set(names)
    if unknown:
        raise ValueError(f'Unsupported tuning parameters: {sorted(unknown)}')
    run_path = Path(config['run_dir']) / f"{config['algo_name']}_{config['dataset']}"
    run_path.mkdir(parents=True, exist_ok=True)
    provenance = experiment_provenance(config)
    provenance['recipe'] = serializable_config(config)
    provenance['search_space'] = space
    provenance_path = run_path / 'tuning_provenance.json'
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        for key in ['dataset_sha256', 'code_sha256', 'recipe', 'search_space']:
            if previous[key] != provenance[key]:
                raise ValueError(f'Resume mismatch in {key}; use a new run_dir')
    else:
        write_json(provenance_path, provenance)
    init_seed(config['seed'], config['reproducibility'])
    processor = Preprocessor(config)
    frame = processor.process(RawDataReader(config).get_data())
    config['user_num'], config['item_num'] = processor.user_num, processor.item_num
    train_indices, _ = TestSplitter(config).split(frame)
    training = frame.iloc[train_indices].copy()
    folds = list(ValidationSplitter(config).split(training))

    def objective(trial):
        recipe = serializable_config(config)
        for name, options in space.items():
            if isinstance(options, list):
                recipe[name] = trial.suggest_categorical(name, options)
            elif param_type_config[name] == 'int':
                recipe[name] = trial.suggest_int(name, options['min'], options['max'], step=options.get('step') or 1)
            else:
                recipe[name] = trial.suggest_float(name, options['min'], options['max'], step=options.get('step'))
        trial.set_user_attr('config', recipe)
        write_json(run_path / 'trial_configs' / f'{trial.number:03d}.json', recipe)
        scores = []
        for fold, (train_indices, validation_indices) in enumerate(folds):
            # Same initialization, sampling seed and candidate policy for each trial.
            init_seed(config['seed'] + fold, config['reproducibility'])
            local = dict(recipe, logger=logger, user_num=processor.user_num, item_num=processor.item_num)
            train = training.iloc[train_indices].copy()
            validation = training.iloc[validation_indices].copy()
            model = fit_model(train, local)
            validation_ur = get_ur(validation)
            init_seed(config['seed'] + fold, config['reproducibility'])
            users, candidates = build_candidates_set(validation_ur, local['train_ur'], local)
            loader = get_dataloader(CandidatesDataset(candidates), 64, False, 0)
            with torch.no_grad():
                predictions = model.rank(loader)
            score = float(metrics_config['ndcg']['evaluator'](validation_ur, predictions, users))
            scores.append(score)
            logger.info(f'Trial {trial.number}, fold {fold}: validation NDCG@{recipe["topk"]}={score:.12f}')
            del model
        trial.set_user_attr('fold_ndcg', scores)
        return float(np.mean(scores))

    storage = config['study_storage'] or f"sqlite:///{(run_path / 'study.db').as_posix()}"
    study = optuna.create_study(direction='maximize', storage=storage,
               study_name=config['study_name'] or f"{config['algo_name']}_{config['dataset']}",
               load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=config['seed']))
    remaining = max(0, config['hyperopt_trail'] - sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials))
    try:
        study.optimize(objective, n_trials=remaining,
                       callbacks=[lambda study, trial: export_study(study, run_path, config, names)])
    finally:
        export_study(study, run_path, config, names)
    logger.info(f'Best trial {study.best_trial.number}: validation NDCG@{config["topk"]}={study.best_value:.12f}')


if __name__ == '__main__':
    main()
