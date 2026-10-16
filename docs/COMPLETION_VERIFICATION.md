# Completion verification

On 6 October 2026, the existing corrected implementation and experiment outputs were inspected and verified in Ubuntu 24.04 under WSL2. The requested model fixes, dataset documentation, tuning plots and final results were already present. Training was not repeated during this verification.

## Confirmed completion

Each of MF, NeuMF and SimpleX has four completed MovieLens trials and eight completed Yelp trials: 36 completed trials in total. Yelp SimpleX selected trial ID 2, with validation NDCG@10 of 0.06517111155697776. All six final result CSVs and their completed manifests exist under `artifacts/verified/results/`.

`bash scripts/verify_linux.sh` passed in the existing Linux environment. Pandas and PyTorch imported successfully; all five behavioral model checks passed; Python compilation completed successfully. The checks cover training-only SimpleX history aggregation and gradients, empty and copied histories, graph-sensitive LightGCN propagation, construction using fitting interactions, single-user ranking and loader equivalence.

`scripts/audit_results.py` was strengthened to compare the current experiment code and dataset hashes with each saved final manifest, require completed final-test status, and resolve paths relative to the project root. The updated audit passed for all six runs. Existing checks also confirm selected trial recipes, saved parameter CSVs, complete trial budgets, matching tuning/final provenance, Linux execution and finite metrics. See [audit evidence](../artifacts/verified/audit.json).

## Using the completed project

From PowerShell in this project folder, inspect completion without retraining:

```powershell
wsl -d Ubuntu-24.04 -- /home/sagniksen/.venvs/daisyrec/bin/python scripts/experiment_status.py
wsl -d Ubuntu-24.04 -- /home/sagniksen/.venvs/daisyrec/bin/python scripts/audit_results.py
```

To repeat all six final experiments using their exact selected training settings:

```powershell
wsl -d Ubuntu-24.04 -- /home/sagniksen/.venvs/daisyrec/bin/python scripts/run_experiments.py --stage test --jobs 2
```

This retrains each selected model and evaluates it, updating the result files. Trained model checkpoints are not saved by the current testing workflow. The completed experiment evidence does not require another training run to inspect or report.

## Documentation and limits

The [experiment report](EXPERIMENT_REPORT.md) records dataset counts, all six test outcomes, tuning evidence and limitations. [Implementation notes](IMPLEMENTATION.md) explain SimpleX and LightGCN. [Dataset provenance](DATASETS.md) records the supplied sampling procedure and the uncertainty around the historical Yelp release. The [README](../README.md) contains setup, preparation and reproducible commands.

The supplied assignment HTML and original Word report are reference documents, not additional user instructions. The Word report remains historical and contains discrepancies documented in the updated Markdown experiment report; it was not rewritten in this verification. Verification was performed in Ubuntu WSL2, not Colab. The short, single-seed tuning budgets are complete but do not establish exhaustive optimization or statistical significance.
