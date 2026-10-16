import json
from pathlib import Path
for path in Path('configs/tuning').glob('*.json'):
 recipe=json.loads(path.read_text())
 recipe['metrics']=['recall','mrr','ndcg','hit','precision','coverage','gini','entropy']
 path.write_text(json.dumps(recipe,indent=2)+'\n')
