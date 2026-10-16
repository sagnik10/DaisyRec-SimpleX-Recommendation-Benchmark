import importlib, json, platform, sys
from pathlib import Path
packages = {}
for name in ['torch','pandas','numpy','optuna','scipy','sklearn','matplotlib']:
    module = importlib.import_module(name)
    packages[name] = module.__version__
record = {'platform':platform.platform(), 'python':platform.python_version(),
          'executable':sys.executable, 'packages':packages,
          'pandas_import':'passed', 'torch_import':'passed'}
print(json.dumps(record,indent=2))
if sys.platform == 'linux':
    Path('artifacts/environment.json').write_text(json.dumps(record,indent=2)+'\n')
