# fable5.md — Channel Imputation Pipeline v2

Change log for the productionisation of `prop_model_app.py` (channel imputation
for unconsented GA4 sessions). The monolithic ~1,060-line script was
restructured into the `channel_imputer/` package with `prop_model_app.py` kept
as the CLI entrypoint, so the existing `python prop_model_app.py` workflow
still works.

Verified end-to-end: `python tests/smoke_test.py` runs the whole pipeline
(features → train → evaluate → impute → export) on synthetic data with no
BigQuery access and asserts the outputs are well-formed. Both the normal
imputation path and the confidence-threshold path were run and pass.

---

## 1. New structure

| File | Responsibility |
|---|---|
| `prop_model_app.py` | Thin CLI entrypoint (argument parsing, logging setup, exit codes) |
| `channel_imputer/config.py` | `PipelineConfig` dataclass — every tunable in one validated, persisted object |
| `channel_imputer/bq_io.py` | BigQuery client, parameterised query, result streaming, BQ export |
| `channel_imputer/features.py` | Feature extraction, leakage protection, journey & session construction |
| `channel_imputer/training.py` | Session samples, time/user splits, label encoding, evaluation metrics |
| `channel_imputer/model.py` | Model architecture and the batched `tf.data` input pipeline |
| `channel_imputer/impute.py` | Confidence-aware session imputation |
| `channel_imputer/pipeline.py` | Orchestration and artifact persistence |
| `tests/smoke_test.py` | Synthetic-data end-to-end test (no cloud access needed) |

---

## 2. Logic and effectiveness improvements

### Session is now the unit of prediction (was: individual events)
The product goal is assigning channels to **sessions**, and GA4's
`session_traffic_source_last_click...primary_channel_group` is session-scoped —
every event in a session shares one channel. The old code trained and imputed
per event, which had three problems:

- **Inconsistent output**: two events in the same unattributed session could be
  imputed with *different* channels, which is incoherent for attribution.
- **Duplicated training samples**: every event in a session carried the same
  label and near-identical features, inflating the training set with correlated
  copies (and wasting compute in proportion).
- **Noisier evaluation**: metrics counted correlated events, not sessions.

Now events are grouped into sessions via `ga_session_id` (extracted as a
first-class column in the query); each session becomes exactly one training or
imputation sample, and an imputed channel is applied to all of the session's
events. Events with no `ga_session_id` (common for unconsented pings) become
singleton sessions rather than being dropped or merged.

### Fixed: `ga_session_id` leaked into the features
The param filter blocked `session_id` (exact match) but not `ga_session_id` or
`ga_session_number`, so a near-unique session identifier was being hashed into
the feature space — an invitation to memorise rather than learn behaviour.
These (plus `firebase_screen_id`) are now excluded as identifiers.

### Fixed: class weights were computed on validation data
`compute_class_weight` previously ran over **all** labels before the
train/validation split, leaking the validation distribution into training.
Weights are now computed from the training subset only (the label encoder is
still fitted on all labelled sessions so validation classes stay representable).

### Confidence-aware imputation
The model's softmax confidence is now used instead of discarded:

- Every imputed event carries an `imputation_confidence` column and a
  `was_imputed` flag in the export — clients can filter or weight downstream.
- `--confidence-threshold X` leaves sessions below the threshold unimputed
  (an honest gap beats a confident-sounding wrong label). Default `0.0`
  preserves the old impute-everything behaviour.
- The run report includes mean/median/min confidence over the imputation set.

### Real evaluation, not just accuracy
Channel distributions are heavily skewed, so accuracy alone can look great
while minority channels (often the paid ones that matter) are never predicted.
Each run now writes `evaluation.json` containing per-class precision / recall /
F1 (`classification_report`) and a confusion matrix, for both:

- **val-time**: sessions after the time cutoff (the production scenario), and
- **val-user**: the cold-start subset whose users were never seen in training.

### Feature hygiene
- Integer event-param values are now bucketed (`bucket_numeric`) instead of
  used raw — values like `engagement_time_msec` are near-unique and previously
  created one hashed feature per millisecond value.
- The click-id exclusion list was extended (`srsltid`, `msclkid`, `ttclid`,
  `twclid`, `li_fat_id`, `igshid`, `epik`, `sccid`) plus a `clid` substring
  rule that catches future ad-platform click-id variants.
- Deprecated `datetime.utcfromtimestamp` replaced with timezone-aware parsing.
- NaN-safe handling throughout (`is_missing`, device/geo fields, params) —
  pandas delivers `NaN` rather than `None` for missing BQ values, which the old
  string-formatting produced literal `"nan"` features from.
- Rows with a null `user_pseudo_id` (typical for unconsented traffic) are kept
  as singleton journeys instead of all collapsing into one giant fake "user"
  journey under the `None` key.

---

## 3. Efficiency improvements

### Removed an O(n²)-per-journey aggregation
`prepare_training_data` rebuilt the full-journey aggregate `Counter` from
scratch **for every event** in the journey, even though the result is identical
for all of them. The journey aggregate is now computed once per user and shared
(by reference) across all of that user's session samples. For a 100-event
journey this is a ~100× reduction in aggregation work.

### Features are extracted and hashed exactly once
The old flow extracted event features twice (once for training prep, once
again inside imputation) and kept raw feature dicts alive. Now each event is
extracted once, hashed immediately to a sparse row, and the dicts are
discarded — the imputation queue reuses the already-vectorised session
samples, so imputation does **no** feature re-extraction at all.

### Journey building: one sort instead of a process pool
`build_event_journeys` used joblib/loky workers running `DataFrame.iterrows()`
per user — the serialisation of per-user DataFrames to worker processes costs
more than the "work" being parallelised, and `iterrows` is one of the slowest
ways to consume pandas data. Replaced with a single vectorised sort plus one
linear pass over `to_dict('records')`. Simpler, faster, and no Windows
spawn-semantics caveats.

### Batched `tf.data` generator
The previous generator densified and yielded **one sample at a time**, paying
Python/TF handoff overhead per sample, then needed a large shuffle buffer
(~660 MB at the default feature dim). The new generator slices whole batches
from the CSR matrices with vectorised scipy ops and yields model-ready batches;
shuffling is a fresh full permutation per epoch, so the shuffle buffer is gone
entirely. Peak memory remains proportional to `batch_size × feature_dim`.

### Integer labels + sparse categorical cross-entropy
One-hot label materialisation (`to_categorical`) removed; labels stay integer
end-to-end.

### Cheaper BigQuery query
- Dropped the unused `campaign` / `medium` / `source` sub-selects and the
  unused `user_id`, `user_first_touch_timestamp`, `event_date` columns —
  BigQuery bills per column scanned, and the attribution columns were
  deliberately never used as features anyway.
- Removed `ORDER BY user_pseudo_id, event_timestamp`: a global sort of a large
  event export is a classic "resources exceeded" failure mode in BigQuery and
  costs slot time; the pipeline sorts locally in one vectorised pandas call.
- Fewer training samples overall (sessions instead of events) means every
  downstream stage — vectorisation, training, imputation — does roughly
  `events_per_session×` less work per epoch.

---

## 4. Production readiness

### Configuration and CLI
Everything that was a hard-coded constant is now a validated
`PipelineConfig` field with a CLI flag: table id, date range, credentials
path, model hyperparameters, thresholds, output destinations. Run
`python prop_model_app.py --help` for the full list. The config is embedded in
the persisted run report, so every artifact directory is self-describing.

### BigQuery cost and safety controls
- **`--max-gb-billed N`** sets `maximum_bytes_billed` — the job fails fast
  instead of silently billing an unbounded scan.
- **`--limit N`** enables cheap smoke runs against live data.
- The date range is passed as **query parameters** instead of f-string
  interpolation; the table id (the one thing BigQuery cannot parameterise) is
  validated against a strict pattern before being interpolated.
- Jobs are labelled (`app: channel-imputer`) for cost attribution in billing
  exports.

### Credentials
The service-account key file is used when present, with automatic fallback to
**Application Default Credentials** — the same code runs unchanged on Cloud
Run / Vertex / GCE with a workload identity and no key file to manage. (The
key file was already git-ignored; for client deployments prefer ADC and avoid
distributing key files at all.)

### Artifacts per run (`models/<run_id>/`)
- `model.h5` — trained model (HDF5 kept for the installed TF 2.9 runtime)
- `feature_hasher.joblib`, `label_encoder.joblib` — inference preprocessing
- `run_config.json` — full config + split sizes + accuracies + imputation
  summary + runtime (the complete, reproducible run report)
- `evaluation.json` — per-class metrics and confusion matrices
- `history.json` — training curves
- `imputed_events.csv` — the deliverable

### Output delivery
`--export-bq-table project.dataset.table` loads the imputed events straight
back into BigQuery (WRITE_TRUNCATE), so client dashboards / Looker Studio can
consume results without CSV handling. The export schema gained
`ga_session_id`, `was_imputed` and `imputation_confidence` columns.

### Testability
`pipeline.run(cfg, events_df=...)` accepts an injected DataFrame, decoupling
the ML pipeline from BigQuery. `tests/smoke_test.py` uses this to run the
entire pipeline on synthetic data in ~30 s and asserts, among other things:

- artifacts all exist and the export schema is exact,
- imputed sessions are internally consistent (one channel per session),
- known labels are never overwritten,
- confidences are valid probabilities and imputed channels come from the
  training label space,
- below-threshold sessions keep their missing channel (tested separately).

### Operational polish
- Proper exit codes (2 for invalid config), module-level loggers, clear
  progress logging at scale checkpoints.
- Deterministic seeding via `tf.keras.utils.set_random_seed`.
- The blocking BigQuery result call replaced the 5-second `reload()` polling
  loop (fewer API calls, same outcome).
- Compatible with the environment actually installed in `.venv`
  (Python 3.10, TensorFlow 2.9, scikit-learn 1.6) as well as the pinned
  TF 2.13. `scipy` (now a direct import) and optional `pyarrow` added to
  `requirements.txt`.

---

## 5. Behaviour changes to be aware of

1. **Output rows are per event, but imputation is per session** — counts in
   logs are now expressed in sessions. `run_config.json` reports both.
2. **Training set size drops** versus v1 (one sample per session instead of
   one per event). This is deliberate: v1's extra samples were near-duplicates
   within each session, not extra information.
3. **`final_channel` can now legitimately stay empty** when
   `--confidence-threshold > 0` and the model is unsure. With the default
   threshold of `0.0` every missing session is imputed, matching v1.
4. **The sliding context window is now the session's first N events**
   (`--max-seq-length`, default 20) rather than a window centred on each
   event; the journey-level aggregate still provides full-journey context.
5. `prop_model_app.py.bak` (the pre-refactor v1 script) was left untouched;
   git history also preserves v1.

## 6. Suggested next steps (not implemented)

- **Referrer signal decision**: `page_referrer` is currently excluded as
  attribution-adjacent. If it is reliably present on unconsented pings in your
  property, whitelisting it would likely be the single biggest accuracy win —
  worth an A/B evaluation run against `evaluation.json` metrics.
- **Calibration**: softmax confidences are typically over-confident; a
  temperature-scaling pass on the validation set would make
  `imputation_confidence` trustworthy as a probability before clients build
  rules on it.
- **Scheduled runs**: the CLI + ADC support means this can be containerised
  and run on a schedule (Cloud Run jobs) with `--export-bq-table` as the
  delivery mechanism.
- Pin the venv and `requirements.txt` to the same TensorFlow version (the
  venv currently has 2.9.0 while requirements pin 2.13.0; the code supports
  both).
