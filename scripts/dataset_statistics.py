"""Inspect the supplied datasets using the exact experiment preprocessing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import logging
import yaml
from daisy.utils.loader import RawDataReader, Preprocessor
from daisy.utils.splitter import TestSplitter, ValidationSplitter
from daisy.utils.utils import write_json


def main():
    records = []
    for dataset in ['ml-50k','yelp-200k']:
        config = yaml.safe_load(Path('daisy/assets/basic.yaml').read_text())
        config.update(dataset=dataset,prepro='origin',logger=logging.getLogger('statistics'))
        raw = RawDataReader(config).get_data()
        processor = Preprocessor(config)
        processed = processor.process(raw.copy())
        training, test = TestSplitter(config).split(processed)
        fitting, validation = next(iter(ValidationSplitter(config).split(processed.iloc[training])))
        filename = Path('data') / dataset / ('ratings.dat' if dataset=='ml-50k' else 'yelp_ratings.csv')
        users, items = processor.user_num, processor.item_num
        row = dict(dataset=dataset,source_file=str(filename),sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                   raw_interactions=len(raw),raw_users=int(raw.user.nunique()),raw_items=int(raw.item.nunique()),
                   duplicate_pairs=int(raw.duplicated(['user','item']).sum()),
                   missing_cells=int(raw.isna().sum().sum()),users=int(users),items=int(items),
                   interactions=len(processed),sparsity_percent=100*(1-len(processed)/(users*items)),
                   minimum_user_interactions=int(processed.groupby('user').size().min()),
                   minimum_item_interactions=int(processed.groupby('item').size().min()),
                   maximum_user_interactions=int(processed.groupby('user').size().max()),
                   start_timestamp=str(processed.timestamp.min()),end_timestamp=str(processed.timestamp.max()),
                   fitting_interactions=len(fitting),validation_interactions=len(validation),
                   final_train_interactions=len(training),test_interactions=len(test),
                   validation_cold_users=int(len(set(processed.iloc[training].iloc[validation].user)-set(processed.iloc[training].iloc[fitting].user))),
                   test_cold_users=int(len(set(processed.iloc[test].user)-set(processed.iloc[training].user))))
        records.append(row)
    write_json('artifacts/dataset_statistics.json',records)
    lines=['# Supplied dataset statistics','',
           'Sparsity is `100 * (1 - interactions / (users * items))` after DaisyRec preprocessing.','',
           '| Dataset | Raw rows | Duplicate pairs | Users | Items | Processed interactions | Sparsity |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['dataset']} | {row['raw_interactions']:,} | {row['duplicate_pairs']:,} | {row['users']:,} | {row['items']:,} | {row['interactions']:,} | {row['sparsity_percent']:.6f}% |")
    Path('docs/dataset_statistics.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
