# Implementation and verification

The project now constructs training histories and interaction graphs before creating a model. SimpleX uses the mean behavior aggregator from the paper, and each final experiment loads a complete saved training recipe. The original Word report was reviewed as historical context; the measured files, new trial records and final manifests govern the updated experimental documentation.

## SimpleX

`daisy/model/SimpleXRecommender.py` implements

`representation(u) = gamma * user_embedding(u) + (1 - gamma) * W_v(mean(item_embeddings(training_history(u))))`.

The mean is zero for an empty history. `W_v` is a learned projection without a bias. `gamma=0.5` is an explicit fixed setting in the experiment recipes. History shares the target-item embedding table, so gradients can update history-item embeddings. Ragged embedding bags aggregate each distinct user's history once per batch and avoid Python loops over users during training. Dropout applies to the combined representation; scoring uses cosine similarity.

The training objective is Cosine Contrastive Loss: `1 - positive_cosine + negative_weight * max(0, negative_cosine - margin)`, averaged over sampled pairs, plus the configured regularization. With a fixed negative count, averaging pairwise samples averages the negative penalties for each positive. The four-negative DaisyRec sampler is retained. It samples negatives once per user and reuses them across that user's positive interactions. This experiment is a mean-aggregator SimpleX implementation with a limited negative budget; it is not a reproduction of the paper's large-negative-budget benchmarks.

`loss_type=BPR` remains the DaisyRec command-line selector for pairwise sample format. SimpleX computes CCL internally, and final manifests explicitly record `actual_loss=CCL`. The report's earlier statement that all three models trained with BPR was inaccurate.

SimpleX requires `train_ur` at construction. `set_user_history` copies item IDs, which prevents subsequent mutation of the caller's dictionary from changing history. During tuning, history comes from the fitting portion of the current validation fold. During final testing, it comes from the complete 80% training portion. Validation and test interactions are excluded from fitting histories; after model selection, the former validation portion is legitimately included in final training.

The authoritative architecture references are the [SimpleX paper](https://arxiv.org/abs/2109.12613) and the authors' [mean aggregator implementation](https://github.com/66chenl/SimpleX/blob/adef0d8085036606ba5ce04aac18939fc29ffb51/src/SimpleX.py). A retrieved source copy is retained under `artifacts/SimpleX-upstream.py`.

## LightGCN

The inspected local file already contained normalized graph propagation. Its tuning integration constructed the model before assigning the interaction graph, so graph models could receive missing or stale split state. The shared `fit_model` helper in `daisy/utils/utils.py` now builds `inter_matrix` from the current fitting split before constructing LightGCN or NGCF.

LightGCN retains DaisyRec's `D^-1/2 A D^-1/2` bipartite adjacency, sparse propagation for every layer, and equal averaging of layer-zero and propagated embeddings. Its omitted tunable-parameter declaration has been restored. The obsolete private SciPy `dok_matrix._update` call was replaced by public sparse block-matrix construction. Duplicate edges are collapsed and converted to binary edges. The normalized adjacency is a registered PyTorch buffer, so it follows the model's device. A single-user candidate batch keeps its two-dimensional score shape, and recommendation IDs remain integers.

The source comparison used [DaisyRec's dev LightGCN implementation](https://github.com/recsys-benchmark/DaisyRec-v2.0/blob/dev/daisy/model/LightGCNRecommender.py), retrieved on 5 October 2026 and retained as `artifacts/LightGCN-upstream.py`. The regression check demonstrates that replacing the graph changes propagated embeddings, and that one-layer outputs equal the expected average of identity and neighbor embeddings.

## Tuning and final recipes

`tune.py` persists an Optuna SQLite study and exports every trial's status, full configuration, validation NDCG, fold scores and duration. A callback refreshes `trials.csv`, `selection.json`, `best_config.json` and DaisyRec's conventional `tune_res` CSV after completed trials. The conventional CSV stores full precision. The complete selected recipe includes fixed settings as well as tuned parameters, so missing values do not need to be inferred from defaults.

Each run's provenance records package versions, Linux platform, dataset SHA-256 and a hash of the experiment code. Resume rejects changed recipes, search spaces, data or code. Each trial uses the same seed for initialization and negative sampling; validation candidates are regenerated with a fixed seed. Trials cannot use the held-out test set for selection.

The explicit saved value `optimizer=default` resolves in the retained model code to SGD for MF and Adam for NeuMF and SimpleX. Likewise, `init_method=default` resolves to normal initialization for MF, Xavier normal for NeuMF and Xavier uniform for SimpleX. These are recorded choices with a matching code fingerprint, rather than inferred missing parameters.

The new experiments select NDCG@10. The legacy code selected NDCG at `topk=50`, while the original report called it NDCG@10. This discrepancy is corrected through fresh tuning for all six pairs. Trial budgets and search ranges come from the complete historical logs associated with the saved best-parameter files: four trials per model on MovieLens and eight per model on Yelp. They are short tuning budgets, not exhaustive searches. The new fixed mean-aggregation weight and common reporting metrics are explicitly recorded in the JSON recipes.

The configured TPE sampler has ten startup trials by default. These four/eight-trial searches stay in its random startup phase and should not be described as completed adaptive Bayesian searches.

`scripts/run_experiments.py` runs tuning, checks that each study reached its requested completed-trial count, and then loads the selected recipe for each final test. The generated `test_config.json` adds the common reporting metrics; all training and split settings match the selected recipe. `test.py` writes one metric CSV and one manifest for each pair. `scripts/audit_results.py` verifies the six files, completed tuning, exact training settings, matching source/data hashes and Linux provenance.

The earlier two-trial Yelp interruption is present in the historical logs. The supplied checkout also contained a later eight-trial completion and selected configuration, so a best configuration had subsequently been saved. Its final test used margin 0.9 despite a selected margin of 0.7. Historical files were copied into `artifacts/legacy/`; the new verified output directory contains the corrected experiments. Existing legacy logs do not retain every trial's validation score, so individual historical scores cannot be reconstructed honestly. Fresh studies provide complete trial evidence.

## Runtime and tests

The scientific runs execute on Ubuntu 24.04 under WSL2 using Python 3.12.3, CPU PyTorch 2.5.1, pandas 2.2.3, NumPy 1.26.4, SciPy 1.14.1 and Optuna 4.1.0. Pandas and PyTorch imports succeeded, and five behavioral regression checks passed. `artifacts/environment.json` and `artifacts/linux_verification.log` contain the environment and verification evidence. Colab instructions are supplied in the README; the recorded verification platform is Ubuntu Linux.

The checks cover the exact history mean and projection, gradients to history items, exclusion of a held-out item from history aggregation, empty histories, immutable copied history, required history input, graph-sensitive LightGCN propagation, model construction from fitting interactions, single-user ranking, and batch-order equivalence.

`BasicDataset` stores samples as a tensor, and its loader indexes a complete batch at once. A regression check confirms equality with standard PyTorch DataLoader batches under both sequential and seeded shuffled sampling. The interrupted attempt before this performance change is retained in `artifacts/pre_batching_attempt/`; final studies use one consistent code version. Training rejects any non-finite loss. MF and NeuMF ranking also retain the batch dimension for a final batch of one user.

A subsequent scheduler interruption stopped the two SimpleX workers. That partial attempt is retained in `artifacts/scheduling_attempt/`. Their final studies were restarted independently; completed baseline studies were retained. Neither interrupted attempt is mixed into the final trial evidence.

Run verification with `python tests/test_models.py`. Run the full experiments with `python scripts/run_experiments.py --stage all --jobs 2`. See the README for Linux setup, dataset preparation and individual commands.
