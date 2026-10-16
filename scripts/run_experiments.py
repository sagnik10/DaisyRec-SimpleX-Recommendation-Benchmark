"""Run or resume all six tuning jobs, then test their exact selected recipes."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [('simplex','yelp-200k'), ('simplex','ml-50k'),
         ('mf','ml-50k'), ('mf','yelp-200k'),
         ('neumf','ml-50k'), ('neumf','yelp-200k')]


def execute(stage, model, dataset, wait_for_tuning=False):
    run = ROOT / 'artifacts/verified' / f'{model}_{dataset}'
    run.mkdir(parents=True, exist_ok=True)
    configuration = (ROOT / 'configs/tuning' / f'{model}_{dataset}.json'
                     if stage == 'tune' else run / 'best_config.json')
    if stage == 'test':
        announced = False
        while True:
            selection_path = run / 'selection.json'
            if selection_path.exists() and json.loads(selection_path.read_text())['tuning_complete']:
                break
            if not wait_for_tuning:
                raise ValueError(f'Tuning is incomplete for {model}/{dataset}')
            if not announced:
                print(f'Waiting for completed tuning: {model} / {dataset}', flush=True)
                announced = True
            time.sleep(10)
    if not configuration.exists():
        raise FileNotFoundError(f'No saved recipe: {configuration}')
    if stage == 'test':
        recipe = json.loads(configuration.read_text())
        recipe['metrics'] = ['recall','mrr','ndcg','hit','precision','coverage','gini','entropy']
        configuration = run / 'test_config.json'
        configuration.write_text(json.dumps(recipe, indent=2) + '\n')
    command = [sys.executable, f'{stage}.py', '--config_file', str(configuration)]
    print(f'{stage}: {model} / {dataset}', flush=True)
    environment = {**os.environ, 'DAISY_PROGRESS':'0', 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2'}
    with (run / f'{stage}_console.log').open('a', encoding='utf-8') as console:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=console, stderr=subprocess.STDOUT, check=True)
    print(f'Completed {stage}: {model} / {dataset}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all','tune','test','plots'], default='all')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--wait-for-tuning', action='store_true')
    parser.add_argument('--models', choices=['mf','neumf','simplex'], nargs='+',
                        default=['mf','neumf','simplex'])
    parser.add_argument('--datasets', choices=['ml-50k','yelp-200k'], nargs='+',
                        default=['ml-50k','yelp-200k'])
    arguments = parser.parse_args()
    os.chdir(ROOT)
    if sys.platform != 'linux':
        raise RuntimeError('Run the verified experiment workflow in Linux or Colab')
    if arguments.stage in ['all','test']:
        subprocess.run([sys.executable,'scripts/check_env.py'], cwd=ROOT, check=True)
        subprocess.run([sys.executable,'scripts/dataset_statistics.py'], cwd=ROOT, check=True)
    if arguments.stage in ['all','tune','test']:
        stages = ['tune','test'] if arguments.stage == 'all' else [arguments.stage]
        for stage in stages:
            with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.jobs) as executor:
                futures = [executor.submit(execute, stage, model, dataset, arguments.wait_for_tuning) for model,dataset in PAIRS
                           if model in arguments.models and dataset in arguments.datasets]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
    full_scope=set(arguments.models)=={'mf','neumf','simplex'} and set(arguments.datasets)=={'ml-50k','yelp-200k'}
    if arguments.stage=='plots' or (arguments.stage=='all' and full_scope):
        subprocess.run([sys.executable,'scripts/plot_tuning.py'], cwd=ROOT, check=True)
    if arguments.stage in ['all','test'] and full_scope:
        subprocess.run([sys.executable,'scripts/audit_results.py'], cwd=ROOT, check=True)
        subprocess.run([sys.executable,'scripts/write_experiment_report.py'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
