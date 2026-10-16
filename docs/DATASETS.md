# Dataset origin and preparation

These experiments use the two supplied subset files, whose SHA-256 fingerprints are recorded in `artifacts/dataset_statistics.json` and every experiment manifest. Dataset names refer to preparation targets rather than guaranteed final row counts.

## MovieLens

MovieLens 1M comes from [GroupLens Research](https://grouplens.org/datasets/movielens/1m/). The recorded project sampling script, `daisy/dataset_sample.txt`, loads the original `ratings.dat`, samples exactly 50,000 rows with `pandas.DataFrame.sample(n=50000, random_state=42)`, and retains the `user::item::rating::timestamp` format. It applies no count filter. The supplied file has 5,620 users, 3,098 items and 50,000 interactions, with no duplicate user-item pairs or missing cells.

The seed and source were independently verified against a fresh download of the GroupLens ML-1M archive on 5 October 2026. Sampling with seed 42 reproduces the supplied subset exactly, including row order. The source archive contains 1,000,209 ratings. `artifacts/movielens_source_verification.json` records the archive SHA-256 and successful exact match; the downloaded archive is retained under `artifacts/sources/`.

## Yelp

The stated source is the [Yelp dataset on Kaggle supplied by the user](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data). Each review is converted from `user_id`, `business_id`, `stars` and `date` to `user`, `item`, `rating` and `timestamp`. The recorded sampling script loads a complete converted CSV, draws 200,000 rows with seed 42, computes user and item counts within that initial sample, and retains rows whose user and item each had at least five interactions in that initial sample.

This is a single-pass count filter. It is not an iterative 5-core procedure, and surviving users or items can have fewer than five remaining interactions. The supplied file confirms minimum degrees of one after preprocessing. The historical full converted CSV and Kaggle release identifier were not retained, so the script documents the intended procedure but does not prove the precise original download version or all sampled review identities.

The supplied `yelp-200k/yelp_ratings.csv` contains 88,692 rows, 13,736 users and 12,287 items. There are 914 repeated user-item pairs. DaisyRec removes those duplicates, leaving 87,778 interactions with the same user and item counts. Calling this file 200,000 final interactions or labelling it Yelp-50K is inaccurate.

## Experiment preprocessing and splitting

Both experiments use `prepro=origin`, `binary_inter=true` and no positive-rating threshold. Existing pair duplicates are removed with `keep='last'` in file order before timestamp sorting; this keeps the last occurrence in the supplied file, not necessarily the chronologically latest review. Ratings of all star values become implicit positive interactions. User and item IDs are encoded, then rows are sorted globally by timestamp. No additional count filter is applied by the experiment loader.

Global time-aware split-by-ratio holds out the latest 20% of processed interactions for test. The latest 10% of the remaining training portion is validation during tuning. Thus model fitting uses approximately 72%, validation 8% and test 20% of the processed dataset. After selecting parameters, final training uses the full 80% training portion. Histories and graphs are built only from the appropriate training portion.

| Dataset | Users | Items | Processed interactions | Sparsity | Fitting | Validation | Final training | Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ml-50k | 5,620 | 3,098 | 50,000 | 99.712821% | 36,000 | 4,000 | 40,000 | 10,000 |
| yelp-200k | 13,736 | 12,287 | 87,778 | 99.947991% | 63,201 | 7,022 | 70,223 | 17,555 |

Sparsity is `100 * (1 - processed_unique_interactions / (users * items))`. Raw-row and processed-interaction counts are both retained in the measured statistics.

Global chronological splitting creates users with no fitting history: 420 in MovieLens validation and 824 in Yelp validation. Final test contains 660 MovieLens users and 1,155 Yelp users absent from final training. Evaluation includes these users. These are materially sparse, partly cold-start comparisons; the outcomes should not be generalized to a warm-user-only protocol.

Ranking uses DaisyRec's 1,000-candidate sampled evaluation. Ground-truth items are included and candidate negatives exclude the user's training items. The retained candidate sampler draws with replacement. This is sampled ranking, not full-catalog ranking. Validation and test candidate generation use a fixed seed in the corrected runs.

## Reproduction

`scripts/prepare_datasets.py` implements the recorded seed-42 procedures and writes source/output hashes. The README provides copy-pasteable commands for MovieLens and either the original Yelp review JSON or a complete converted CSV. Preparation can produce different Yelp files if the source release or source row ordering differs. Use the supplied subset files and their recorded hashes to reproduce the verified experiments exactly.

Run `python scripts/dataset_statistics.py` to regenerate [the measured table](dataset_statistics.md) and machine-readable statistics. The original Word report's dataset labels, interaction counts and no-filter description are superseded by these measurements and the retained preparation script.
