# DaisyRec recommendation experiments

This project evaluates MF, NeuMF and the corrected mean-aggregator SimpleX on the supplied `ml-50k` and `yelp-200k` datasets. LightGCN retains DaisyRec's graph propagation and is covered by regression checks. Tuning selects validation NDCG@10; final tests load the complete selected configuration.

## Environment setup

The verified environment is Ubuntu 24.04 under WSL2, Python 3.12.3 and CPU PyTorch 2.5.1. Pandas and PyTorch imported successfully in Linux. See [environment evidence](artifacts/environment.json) and [change documentation](docs/IMPLEMENTATION.md).

From the project root in Ubuntu/Linux:

```bash
bash scripts/setup_linux.sh
source "$HOME/.venvs/daisyrec/bin/activate"
python tests/test_models.py
python scripts/dataset_statistics.py
```

The setup script installs the CPU PyTorch wheel before the remaining dependencies. CUDA is optional. `requirements.txt` pins the direct dependencies used by the verified environment; `artifacts/requirements-linux.lock.txt` records its full installed package set.

## Run the software

The supplied checkout has completed all 36 configured tuning trials and all six final tests. This was rechecked on 6 October 2026 in Ubuntu WSL2: all five model regression tests passed, and the six-run audit confirmed the current code/data hashes and selected training recipes. See [completion verification](docs/COMPLETION_VERIFICATION.md). No additional tuning is required for these configured budgets.

To inspect completion without training again, run from the project root:

```bash
python scripts/experiment_status.py
python scripts/audit_results.py
```

To repeat only the six final tests with the saved best settings:

```bash
python scripts/run_experiments.py --stage test --jobs 2
```

Final testing fits each model again before evaluating; this project saves results and recipes, not trained model checkpoints.

Run the complete workflow, including all six tuning jobs, six final tests, plots and an audit:

```bash
python scripts/run_experiments.py --stage all --jobs 2
```

The supplied datasets are already present, so preparation is unnecessary for this checkout. Completed studies are retained in SQLite; rerunning tuning resumes the same study rather than discarding trials. A code, data or recipe mismatch requires a new experiment directory. Final testing requires a completed tuning selection.

To run each stage separately:

```bash
python scripts/run_experiments.py --stage tune --jobs 2
python scripts/run_experiments.py --stage test --jobs 2
python scripts/run_experiments.py --stage plots
python scripts/audit_results.py
```

For one tuning run:

```bash
python tune.py --config_file configs/tuning/simplex_yelp-200k.json
```

All six individual tuning commands are:

```bash
python tune.py --config_file configs/tuning/mf_ml-50k.json
python tune.py --config_file configs/tuning/mf_yelp-200k.json
python tune.py --config_file configs/tuning/neumf_ml-50k.json
python tune.py --config_file configs/tuning/neumf_yelp-200k.json
python tune.py --config_file configs/tuning/simplex_ml-50k.json
python tune.py --config_file configs/tuning/simplex_yelp-200k.json
```

For its final test, using the saved selected configuration:

```bash
python test.py --config_file artifacts/verified/simplex_yelp-200k/best_config.json
```

The six-test runner writes `test_config.json` for each pair. Its training and split settings exactly match `best_config.json`; it adds the common reporting metrics Recall, MRR, NDCG, Hit Ratio, Precision, Coverage, Gini and Entropy. To reproduce that complete result file individually, use `test_config.json` instead of `best_config.json`.

All six individual final-test commands use the saved selected recipes:

```bash
python test.py --config_file artifacts/verified/mf_ml-50k/best_config.json
python test.py --config_file artifacts/verified/mf_yelp-200k/best_config.json
python test.py --config_file artifacts/verified/neumf_ml-50k/best_config.json
python test.py --config_file artifacts/verified/neumf_yelp-200k/best_config.json
python test.py --config_file artifacts/verified/simplex_ml-50k/best_config.json
python test.py --config_file artifacts/verified/simplex_yelp-200k/best_config.json
```

On this Windows computer, launch the verified Linux environment from PowerShell:

```powershell
wsl -d Ubuntu-24.04 -- /home/sagniksen/.venvs/daisyrec/bin/python /mnt/c/Users/Administrator/Downloads/DaisyRec-v2.0_Updated/DaisyRec-v2.0/scripts/run_experiments.py --stage all --jobs 2
```

## Dataset preparation

`ml-50k` is a seed-42 sample of 50,000 MovieLens 1M ratings from [GroupLens](https://grouplens.org/datasets/movielens/1m/). `yelp-200k` is named for the initial sample size: the original script sampled 200,000 reviews, then applied a single-pass user/item count filter. The supplied Yelp CSV has 88,692 rows, and experiment preprocessing removes 914 duplicate pairs. See [dataset provenance](docs/DATASETS.md) and [measured statistics](docs/dataset_statistics.md).

To prepare new files from downloaded originals:

```bash
curl -L https://files.grouplens.org/datasets/movielens/ml-1m.zip -o ml-1m.zip
unzip ml-1m.zip
python scripts/prepare_datasets.py --movielens ml-1m/ratings.dat --output-dir data
```

Download the Yelp review JSON from the [provided Kaggle dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data), extract it, then run:

```bash
python scripts/prepare_datasets.py --yelp-json /path/to/yelp_academic_dataset_review.json --output-dir data
python scripts/dataset_statistics.py
```

If a complete converted Yelp CSV is already available:

```bash
python scripts/prepare_datasets.py --yelp-csv /path/to/yelp_ratings.csv --output-dir data
```

The Yelp preparation procedure loads the full converted CSV before sampling, matching the original script. It requires more memory than running experiments on the prepared subset. Preparation writes source/output hashes and sampling provenance. The exact historical Yelp source release was not recorded; a current download may produce a different subset. Preserve the supplied files to reproduce these experiments.

## Colab

Upload this project as a ZIP to a Colab environment with Python 3.12, including `data/`, `configs/`, `daisy/`, `tune.py`, `test.py`, `scripts/` and `requirements.txt`. Unzip it under `/content`, open a cell in the extracted project directory, and run:

```python
%cd /content/DaisyRec-v2.0
import sys
assert sys.version_info[:2] == (3, 12), "These pinned dependencies were verified with Python 3.12"
!pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
!pip install -r requirements.txt
!python tests/test_models.py
!python scripts/run_experiments.py --stage all --jobs 2
```

These are preparation instructions for Colab. The actual recorded verification was performed on Ubuntu WSL2. Download `artifacts/verified/` before the Colab runtime expires.

## Results and documentation

- `artifacts/verified/results/<dataset>/<model>_kpi_results.csv`: one final result file per pair.
- `artifacts/verified/<model>_<dataset>/trials.csv`: every trial's score, status and parameters.
- `artifacts/verified/<model>_<dataset>/study.db`: resumable Optuna study.
- `artifacts/verified/<model>_<dataset>/best_config.json`: full selected recipe, including fixed settings.
- `artifacts/verified/<model>_<dataset>/selection.json`: selected trial, score and tuning-completion status.
- `artifacts/verified/plots/`: per-trial and cumulative-best figures for both datasets.
- `artifacts/verified/final_summary.csv` and `selected_parameters.csv`: consolidated final tables.
- `artifacts/verified/audit.json`: checks for all six completed runs and matching training recipes.
- After the six-run audit passes, result files and manifests are also copied to DaisyRec's conventional `res/<dataset>/origin/tsbr/` directories.
- [Experiment report](docs/EXPERIMENT_REPORT.md), [implementation notes](docs/IMPLEMENTATION.md) and [dataset documentation](docs/DATASETS.md).

Historical results and best-parameter files were copied to `artifacts/legacy/`. They should be interpreted using their original code and configurations. The corrected experiments use a separate output directory. Original DaisyRec project information is in [the upstream README](docs/UPSTREAM_README.md); the [original license](LICENSE) remains applicable.
