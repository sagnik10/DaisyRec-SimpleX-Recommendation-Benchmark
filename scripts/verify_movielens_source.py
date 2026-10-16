import hashlib,json,zipfile,sys
from pathlib import Path
import pandas as pd
archive=Path('artifacts/sources/ml-1m.zip')
with zipfile.ZipFile(archive) as source:
 with source.open('ml-1m/ratings.dat') as ratings:
  original=pd.read_csv(ratings,sep='::',header=None,names=['user','item','rating','timestamp'],engine='python')
expected=original.sample(n=50000,random_state=42).reset_index(drop=True)
supplied=pd.read_csv('data/ml-50k/ratings.dat',sep='::',header=None,names=expected.columns,engine='python')
record={'source_url':'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'download_date':'2026-10-05','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'source_interactions':len(original),'seed':42,
        'exact_sample_and_row_order_match':expected.equals(supplied)}
Path('artifacts/movielens_source_verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
assert record['exact_sample_and_row_order_match']
