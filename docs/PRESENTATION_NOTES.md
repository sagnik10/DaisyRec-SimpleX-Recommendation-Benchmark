# DaisyRec workflow and explanation of the updated version

These notes are for personal presentation preparation and are separate from the submission report. They explain the recorded development workflow, the corrections, and what the results support. The implementation history below comes from the project records; the report update itself did not rerun training.

## A short opening you can say

“My project integrates the published SimpleX model into DaisyRec and compares it with MF and NeuMF using the same data splitting and sampled ranking protocol. SimpleX combines a learned user embedding with information from the user's training history and learns with cosine contrastive loss. I corrected the experimental pipeline and made the report match the saved configurations and final results. NeuMF performed better on the MovieLens subset, while SimpleX performed better on the Yelp subset in these runs. The comparisons use one seed and small tuning budgets, so I do not claim statistical significance or universal superiority.”

## The workflow from data to results

1. **Prepare the supplied subsets.** MovieLens is an exact 50,000-row sample of MovieLens 1M using seed 42, without a count filter. The recorded Yelp preparation samples 200,000 reviews with seed 42, then keeps rows whose user and item each occurred at least five times in that initial sample. This is a single-pass filter, not an iterative 5-core. The supplied Yelp file contains 88,692 rows. Its historical release is unknown, so the supplied file and hashes are the reproducible reference.
2. **Preprocess interactions.** Keep the last duplicate user-item occurrence in file order, encode IDs, convert every rating to an implicit positive, and sort globally by timestamp. Yelp loses 914 duplicate pairs and has 87,778 processed interactions. No additional count filter or rating threshold is applied by the experiment loader.
3. **Split before fitting.** Hold out the latest 20% for test. During tuning, reserve the latest 10% of the remaining 80% for validation. This gives approximately 72% fitting, 8% validation and 20% test. Build histories and any graphs using only the current fitting portion.
4. **Fit candidate configurations.** Train MF with BPR for 20 epochs and NeuMF with BPR for 30 epochs. Train SimpleX with CCL for 30 epochs. All use seed 2022 and four uniform negatives per positive. The `loss_type=BPR` selector in SimpleX selects the pairwise input format; it does not change its internal CCL objective.
5. **Select on validation.** Choose the highest validation NDCG@10 from four completed trials per model on MovieLens and eight per model on Yelp. There are 36 completed trials. Optuna's configured TPE sampler has ten startup trials, so every study remains in its random startup phase. Describe this as a short hyperparameter search, not a completed adaptive Bayesian search.
6. **Retrain the selected recipe.** Use the full 80% training portion after selection. The former validation interactions now legitimately belong to training. Rebuild histories and graphs from that portion; test interactions remain excluded. Load the complete `best_config.json` recipe to avoid accidentally substituting defaults or different margins.
7. **Evaluate and audit.** Rank 1,000 sampled candidates per user at K = 10. Include ground-truth items and exclude training items from candidate negatives; the retained candidate sampler samples with replacement. Save metric CSVs and provenance, then audit trial completion, selected settings and source/data hashes. The workflow saves recipes and results rather than trained model checkpoints.

The sequence to remember is **prepare → preprocess → split → tune on validation → select → retrain on 80% → test → audit → report**.

## What SimpleX does in plain language

MF learns a vector for each user and item and scores their dot product. NeuMF combines a factorization branch with a neural network. SimpleX also learns user and item vectors, but uses the items in the user's training history to form a mean history vector. A learned projection transforms that mean, then a fixed weight of 0.5 combines it with the user's embedding. Dropout applies to the combined representation, and cosine similarity scores items.

CCL pushes a positive item's cosine score toward one and penalizes a negative item's score when it exceeds the chosen margin:

`loss = 1 - positive_cosine + negative_weight * max(0, negative_cosine - margin)`

The implementation averages this objective over sampled pairs. Empty histories produce a zero history mean. History items share the item embedding table, allowing gradients to update their representations. This is an implementation of an existing published model; the project's contribution is adding it to DaisyRec and evaluating it under the recorded protocol. Four negatives per positive do not reproduce the paper's large-negative-budget benchmarks.

## How the updated project was developed

The project records describe inspection of the supplied code, legacy logs, parameter files and original report. That review exposed inconsistencies in dataset naming, the ranking cutoff used for selection, the actual SimpleX loss, and the parameters used for final testing. For example, legacy selection used NDCG at K = 50 while the report called it NDCG@10. A historical Yelp SimpleX test used margin 0.9 despite a selected margin of 0.7. Those historical values are not the current selected recipe.

The corrected implementation constructs training histories and graphs before model creation. SimpleX uses a projected mean history aggregator with training-only histories. LightGCN already contained normalized propagation; its integration was repaired so it receives the current fitting graph. Public SciPy sparse construction replaced an obsolete private API. LightGCN was checked but is not included in the six final MF, NeuMF and SimpleX comparisons.

The tuning workflow was changed to retain every trial's scores, status, duration and complete recipe, with SQLite studies for resume. Final tests load selected recipes instead of manually copying a few parameters. Resume checks reject changed code, data or recipes. Batch loading was improved and verified against the standard loader, and non-finite training loss is rejected. Interrupted attempts and legacy outputs were retained separately from the final evidence.

Fresh studies then selected on validation NDCG@10 for all six model/dataset pairs. The recorded experiments ran on Ubuntu 24.04 under WSL2 using CPU PyTorch. The project records five passing behavioral checks and a passing six-result audit. The final report was reconciled with these completed experiments. This presentation material and the updated Word copy describe that existing evidence; they do not represent another set of experiments.

## Correct dataset figures

| Dataset | Supplied rows | Duplicate pairs removed | Users | Items | Processed interactions | Sparsity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ml-50k | 50,000 | 0 | 5,620 | 3,098 | 50,000 | 99.712821% |
| yelp-200k | 88,692 | 914 | 13,736 | 12,287 | 87,778 | 99.947991% |

Explain “200k” as the initial sampling target, not the final interaction count. Users and items can have fewer than five interactions after the single-pass filter. “No additional filtering in the loader” and “filtering during Yelp preparation” describe different stages and are compatible.

## Selected settings you should know

| Model and dataset | Factors | Batch | Learning rate | L1 | L2 | Epochs | Other selected settings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MF ml-50k | 32 | 1024 | 0.021 | 0.001 | 0 | 20 | SGD; normal initialization |
| MF yelp-200k | 32 | 1024 | 0.021 | 0.001 | 0 | 20 | SGD; normal initialization |
| NeuMF ml-50k | 16 | 256 | 0.005 | 0.009 | 0.007 | 30 | 3 layers; dropout 0.2 |
| NeuMF yelp-200k | 48 | 512 | 0.003 | 0.009 | 0.004 | 30 | 2 layers; dropout 0.3 |
| SimpleX ml-50k | 64 | 512 | 0.008 | 0 | 0 | 30 | Margin 0.9; negative weight 20; dropout 0.2 |
| SimpleX yelp-200k | 32 | 256 | 0.0055 | 0 | 0 | 30 | Margin 0.5; negative weight 10; dropout 0.3 |

Both SimpleX recipes fix gamma at 0.5. NeuMF uses Adam with Xavier normal initialization; SimpleX uses Adam with Xavier uniform initialization. The displayed numbers are readable decimal forms; the complete saved JSON recipes retain full precision. Saved selected trial IDs are zero-based: MF 0 on both datasets; NeuMF 0 on MovieLens and 2 on Yelp; SimpleX 2 on both. Plot trial labels are one-based.

## The result to present

| Dataset | MF NDCG@10 | NeuMF NDCG@10 | SimpleX NDCG@10 | Highest in these runs |
| --- | ---: | ---: | ---: | --- |
| ml-50k | 0.086964 | 0.120102 | 0.040691 | NeuMF |
| yelp-200k | 0.032882 | 0.018919 | 0.082864 | SimpleX |

“NeuMF performed better in these MovieLens runs, and SimpleX performed better in these Yelp runs” is the supported conclusion. Their ordering is consistent across the five saved ranking metrics within each dataset. The experiments do not isolate whether history aggregation, loss, selected settings or dataset characteristics caused the ordering.

SimpleX covers 92.5206% of the Yelp catalog and 54.4868% of the MovieLens catalog in the saved recommendations. Coverage describes how much of the catalog appears. Entropy describes the distribution of exposure, and the retained Gini calculation measures concentration only among items that receive recommendations. These measurements do not prove fairness, user satisfaction, within-list diversity or generalization. On MovieLens, MF has lower Gini and higher entropy than SimpleX, despite lower coverage, illustrating why the exposure metrics should be discussed separately.

## Questions you may be asked

**Did you invent SimpleX?** No. SimpleX is a published model. I implemented and integrated its mean-history approach into DaisyRec and compared it with existing baselines.

**Why is the Yelp dataset named 200k when it has fewer rows?** The name records the initial sampling target. Preparation filtering leaves the supplied 88,692 rows, and duplicate removal leaves 87,778 interactions used by the experiments.

**Was the tuning Bayesian optimization?** Optuna TPE was configured, but four/eight trials are fewer than its ten startup trials. These completed studies only exercise its random startup phase.

**How did you prevent leakage?** Validation and test interactions are excluded from tuning histories and fitting graphs. Final training includes the former validation portion after selection, but still excludes test interactions. Test metrics never select configurations.

**Why do the corrected scores differ from the old report?** The old report and historical outputs were inconsistent, including selection at K = 50 and a mismatched SimpleX test margin. The corrected scores come from fresh, recorded NDCG@10 studies and final tests that load their selected recipes. I cannot attribute each numerical change to one fix without controlled ablations.

**Does high coverage mean fairness?** No. It measures catalog reach. Fairness would require a defined fairness criterion and appropriate user/item or group-level evidence.

**What limits the comparison?** One seed, four/eight tuning trials, four negatives per positive, sampled candidates with replacement, and cold users in the global time split. There are 660 MovieLens and 1,155 Yelp test users without training history. Repeated seeds, larger searches, warm-user and full-catalog evaluation, and history ablation would strengthen the evidence.

## Where to show the evidence

- [Experiment report](EXPERIMENT_REPORT.md): protocol, selected configurations, results and limitations.
- [Final summary CSV](../artifacts/verified/final_summary.csv): all six final results at full precision.
- [Selected parameters](../artifacts/verified/selected_parameters.csv): tuned and fixed settings.
- [Trial evidence](../artifacts/verified/tuning_trials.csv): completed trial scores and parameters.
- [Implementation notes](IMPLEMENTATION.md): model equations, fixes and published source references.
- [Dataset documentation](DATASETS.md): preparation, measured counts and provenance limits.
- [Completion verification](COMPLETION_VERIFICATION.md): recorded Linux checks and audit.

For a short presentation, show the workflow, explain the history-based SimpleX representation, show the two-row NDCG table, and finish with the single-seed and short-search limitations. Use the updated Word report for the complete metrics and settings.
