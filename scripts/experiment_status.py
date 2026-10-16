"""Show tuning and final-test progress without importing numeric packages."""
import json
import re
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'artifacts/verified'
for model in ['mf','neumf','simplex']:
 for dataset in ['ml-50k','yelp-200k']:
  run=root/f'{model}_{dataset}'
  selection=run/'selection.json'
  if selection.exists():
   value=json.loads(selection.read_text())
   message=f"{value['complete_trials']}/{value['requested_trials']} trials, best validation NDCG@10={value['validation_ndcg']:.6f}"
  else:
   message='first trial running' if run.exists() else 'queued'
  if selection.exists() and not value['tuning_complete']:
   log=run/'tune_console.log'
   if log.exists():
    epochs=re.findall(r'Epoch (\d+)/(\d+)',log.read_text(errors='replace'))
    if epochs:
     message+=f"; latest tuning epoch {epochs[-1][0]}/{epochs[-1][1]}"
  manifest=root/'results'/dataset/f'{model}_manifest.json'
  if manifest.exists():
   message+='; final test saved'
  else:
   log=run/'test_console.log'
   if log.exists():
    epochs=re.findall(r'Epoch (\d+)/(\d+)',log.read_text(errors='replace'))
    message+=f'; final training epoch {epochs[-1][0]}/{epochs[-1][1]}' if epochs else '; final test starting'
  print(f'{model:7} {dataset:9}: {message}')
