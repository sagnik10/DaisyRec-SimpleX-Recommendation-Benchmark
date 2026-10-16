set -euo pipefail
cd /mnt/c/Users/Administrator/Downloads/DaisyRec-v2.0_Updated/DaisyRec-v2.0
/home/sagniksen/.venvs/daisyrec/bin/python scripts/run_experiments.py --stage test --models simplex neumf --datasets yelp-200k --wait-for-tuning --jobs 2
/home/sagniksen/.venvs/daisyrec/bin/python scripts/plot_tuning.py
/home/sagniksen/.venvs/daisyrec/bin/python scripts/audit_results.py
/home/sagniksen/.venvs/daisyrec/bin/python scripts/write_experiment_report.py
