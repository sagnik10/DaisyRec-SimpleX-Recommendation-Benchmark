"""Reproduce the recorded seed-42 sampling procedure from local source files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pandas as pd


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--movielens',type=Path,help='Original ML-1M ratings.dat')
    parser.add_argument('--yelp-csv',type=Path,help='Complete review CSV in original JSON-line order')
    parser.add_argument('--yelp-json',type=Path,help='Original Yelp review JSON-lines; converts before sampling')
    parser.add_argument('--output-dir',type=Path,default=Path('data'))
    args=parser.parse_args()
    if not any([args.movielens,args.yelp_csv,args.yelp_json]):
        parser.error('Provide at least one source file')
    if args.yelp_csv and args.yelp_json:
        parser.error('Use either --yelp-csv or --yelp-json')
    if args.movielens:
        frame=pd.read_csv(args.movielens,sep='::',header=None,names=['user','item','rating','timestamp'],engine='python')
        sample=frame.sample(n=50000,random_state=42)
        destination=args.output_dir/'ml-50k'/'ratings.dat'
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('w') as target:
            for row in sample.itertuples(index=False,name=None):
                target.write('::'.join(map(str,row))+'\n')
        provenance=dict(source=str(args.movielens),source_sha256=digest(args.movielens),
                        source_rows=len(frame),seed=42,sampled_rows=50000,filter='none',
                        output_sha256=digest(destination))
        destination.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if args.yelp_csv or args.yelp_json:
        source=args.yelp_csv or args.yelp_json
        csv_source=source
        if args.yelp_json:
            csv_source=args.output_dir/'yelp'/'yelp_ratings.csv'
            csv_source.parent.mkdir(parents=True,exist_ok=True)
            with source.open(encoding='utf8') as reviews,csv_source.open('w',newline='',encoding='utf8') as target:
                writer=csv.writer(target)
                writer.writerow(['user','item','rating','timestamp'])
                for line in reviews:
                    review=json.loads(line)
                    writer.writerow([review['user_id'],review['business_id'],review['stars'],review['date']])
        frame=pd.read_csv(csv_source)
        sample=frame.sample(n=200000,random_state=42)
        # This is the original single-pass count filter, not an iterative 5-core.
        users=sample.user.value_counts()
        items=sample.item.value_counts()
        sample=sample[sample.user.isin(users[users>=5].index)&sample.item.isin(items[items>=5].index)]
        destination=args.output_dir/'yelp-200k'/'yelp_ratings.csv'
        destination.parent.mkdir(parents=True,exist_ok=True)
        sample.to_csv(destination,index=False)
        provenance=dict(source=str(source),source_sha256=digest(source),source_rows=len(frame),
                        seed=42,sampled_rows=200000,output_rows=len(sample),
                        filter='Single pass: user and item counts >=5 in the unfiltered sample',
                        output_sha256=digest(destination))
        destination.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        print(json.dumps(provenance,indent=2))


if __name__=='__main__':
    main()
