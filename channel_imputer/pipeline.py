"""End-to-end pipeline: query → features → train → evaluate → impute → export."""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import joblib
import pandas as pd
import tensorflow as tf

from . import __version__
from .bq_io import export_to_bigquery, make_client, query_event_data
from .config import PipelineConfig
from .features import build_user_journeys
from .impute import impute_missing_channels
from .model import build_model, default_callbacks, make_dataset
from .training import (
    build_session_samples, encode_labels, evaluate_predictions,
    stack_split, time_then_user_split,
)

logger = logging.getLogger(__name__)

EXPORT_COLUMNS = [
    'user_pseudo_id', 'ga_session_id', 'event_timestamp', 'event_name',
    'original_channel', 'final_channel', 'was_imputed',
    'imputation_confidence',
]


def _make_artifact_dir(root: str):
    run_id = uuid.uuid4().hex[:8]
    artifact_dir = os.path.join(root, run_id)
    os.makedirs(artifact_dir, exist_ok=True)
    return run_id, artifact_dir


def _save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _export_events(user_journeys, artifact_dir: str,
                   client, cfg: PipelineConfig) -> None:
    rows = [
        {c: e.get(c) for c in EXPORT_COLUMNS}
        for events in user_journeys.values()
        for e in events
    ]
    df = pd.DataFrame(rows, columns=EXPORT_COLUMNS)
    csv_path = os.path.join(artifact_dir, "imputed_events.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s events to '%s'", f"{len(df):,}", csv_path)

    if cfg.output_bq_table:
        if client is None:
            logger.warning("No BigQuery client available; skipping export "
                           "to %s", cfg.output_bq_table)
        else:
            export_to_bigquery(client, df, cfg.output_bq_table)


def run(cfg: PipelineConfig, events_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Execute the full pipeline.  `events_df` injects a pre-fetched event
    DataFrame (tests / offline replays); when omitted, data is queried
    from BigQuery per the config.

    Returns the run report (also persisted as run_config.json).
    """
    start_time = time.time()
    cfg.validate()

    tf.keras.utils.set_random_seed(cfg.random_state)

    run_id, artifact_dir = _make_artifact_dir(cfg.artifact_root)
    logger.info("channel_imputer v%s | Run ID: %s | Artifacts: %s/",
                __version__, run_id, artifact_dir)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    client = None
    if events_df is None:
        client = make_client(cfg)
        events_df = query_event_data(client, cfg)
    elif cfg.output_bq_table:
        client = make_client(cfg)

    user_journeys = build_user_journeys(events_df)
    del events_df

    # ------------------------------------------------------------------
    # 2. Session samples (features extracted and hashed exactly once)
    # ------------------------------------------------------------------
    hasher, labelled, unlabelled = build_session_samples(
        user_journeys, cfg.feature_dim, cfg.max_seq_length,
    )

    # ------------------------------------------------------------------
    # 3. Split, encode, build datasets
    # ------------------------------------------------------------------
    idx_train, idx_val_time, idx_val_user, cutoff_ts = time_then_user_split(
        labelled, cfg.time_split_fraction,
    )
    le, y, num_classes, class_weights = encode_labels(labelled, idx_train)

    agg_train, seq_train, y_train = stack_split(labelled, idx_train, y)
    agg_val, seq_val, y_val = stack_split(labelled, idx_val_time, y)
    agg_val_user, seq_val_user, y_val_user = stack_split(
        labelled, idx_val_user, y,
    )
    if len(y_train) == 0 or len(y_val) == 0:
        raise RuntimeError(
            "Empty train or validation split — not enough labelled sessions "
            "across the configured date range."
        )

    ds_kwargs = dict(max_seq_len=cfg.max_seq_length,
                     feature_dim=cfg.feature_dim,
                     batch_size=cfg.batch_size, seed=cfg.random_state)
    train_ds = make_dataset(agg_train, seq_train, y_train,
                            shuffle=True, **ds_kwargs)
    val_ds = make_dataset(agg_val, seq_val, y_val, shuffle=False, **ds_kwargs)

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    model = build_model(cfg.feature_dim, cfg.max_seq_length,
                        num_classes, cfg.learning_rate)
    logger.info("Training for up to %d epochs", cfg.epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=default_callbacks(),
        class_weight=class_weights,
    )

    model_path = os.path.join(artifact_dir, "model.h5")
    model.save(model_path)
    logger.info("Model saved to '%s'", model_path)

    # ------------------------------------------------------------------
    # 5. Evaluate (per-class metrics, not just accuracy)
    # ------------------------------------------------------------------
    class_names = [str(c) for c in le.classes_]
    val_probs = model.predict(val_ds, verbose=0)
    evaluation = {
        'val_time': evaluate_predictions(y_val, val_probs, class_names),
    }
    logger.info("Val-time accuracy: %.2f%% over %s sessions",
                evaluation['val_time']['accuracy'] * 100, f"{len(y_val):,}")

    if len(y_val_user) > 0:
        val_user_ds = make_dataset(agg_val_user, seq_val_user, y_val_user,
                                   shuffle=False, **ds_kwargs)
        val_user_probs = model.predict(val_user_ds, verbose=0)
        evaluation['val_user'] = evaluate_predictions(
            y_val_user, val_user_probs, class_names,
        )
        logger.info("Val-user (cold-start) accuracy: %.2f%% over %s sessions",
                    evaluation['val_user']['accuracy'] * 100,
                    f"{len(y_val_user):,}")

    _save_json(os.path.join(artifact_dir, "evaluation.json"), evaluation)

    # ------------------------------------------------------------------
    # 6. Impute + export
    # ------------------------------------------------------------------
    imputation_summary = impute_missing_channels(
        unlabelled, user_journeys, model, le, cfg,
    )
    _export_events(user_journeys, artifact_dir, client, cfg)

    # ------------------------------------------------------------------
    # 7. Persist artifacts + run report
    # ------------------------------------------------------------------
    joblib.dump(hasher, os.path.join(artifact_dir, "feature_hasher.joblib"))
    joblib.dump(le, os.path.join(artifact_dir, "label_encoder.joblib"))
    _save_json(os.path.join(artifact_dir, "history.json"),
               {k: [float(v) for v in vals]
                for k, vals in history.history.items()})

    report = {
        'run_id': run_id,
        'version': __version__,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'config': cfg.to_dict(),
        'epochs_trained': len(history.history['loss']),
        'num_classes': num_classes,
        'classes': class_names,
        'class_weights': class_weights,
        'time_split_cutoff_ts': int(cutoff_ts),
        'train_sessions': int(len(y_train)),
        'val_time_sessions': int(len(y_val)),
        'val_user_sessions': int(len(y_val_user)),
        'val_time_accuracy': evaluation['val_time']['accuracy'],
        'val_user_accuracy': (
            evaluation.get('val_user', {}).get('accuracy')
        ),
        'imputation': imputation_summary,
        'runtime_seconds': round(time.time() - start_time, 1),
    }
    _save_json(os.path.join(artifact_dir, "run_config.json"), report)

    logger.info("Pipeline complete. Run ID: %s | Total runtime: %.1fs",
                run_id, report['runtime_seconds'])
    return report
