"""Create a corrected report copy without changing experiment evidence."""
from pathlib import Path
import csv
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
doc = Document(ROOT / 'CA6124 Final Report.docx')
for p in doc.paragraphs:
    if p.text.startswith('This project extends DaisyRec [1]'):
        p.text = ('This project integrates the existing published SimpleX model [4] into DaisyRec [1] and compares it with Matrix Factorization (MF) and Neural Matrix Factorization (NeuMF) on MovieLens and Yelp subsets. The contribution is its implementation and integration, including training-history aggregation and cosine contrastive learning. Across six completed final experiments, NeuMF performed better on ml-50k by test NDCG@10 (0.120102), while SimpleX performed better on yelp-200k (0.082864). These outcomes apply to the sampled, single-seed protocol used here.')
    elif p.text.startswith('MF scores users and items'):
        p.text = p.text.replace('following the SimpleX design [4].', 'following the published SimpleX model [4]. The implementation is in daisy/model/SimpleXRecommender.py and is selected with --algo_name simplex.')
    elif p.text.startswith('The project integrates training-history-aware SimpleX'):
        p.text = p.text.replace('NeuMF performs best on the MovieLens subset and SimpleX on the Yelp subset in these runs.', 'NeuMF performed better than the other compared models on the MovieLens subset, and SimpleX performed better on the Yelp subset in these runs.')
out = ROOT / 'CA6124 Updated Report.docx'
doc.save(out)

# Verify every displayed final metric against the saved source of truth.
rows = list(csv.DictReader((ROOT / 'artifacts/verified/final_summary.csv').open()))
for r in rows:
    for key in ['Recall', 'NDCG', 'MRR', 'Hit Ratio', 'Precision', 'Coverage', 'Gini', 'Entropy']:
        value = f'{float(r[key]):.6f}'
        assert any(value in cell.text for table in doc.tables for row in table.rows for cell in row.cells), (r['model'], r['dataset'], key, value)
assert 'Yelp-50K' not in '\n'.join(p.text for p in doc.paragraphs)
print(f'Created {out.name}; all 48 final metric values match final_summary.csv.')
