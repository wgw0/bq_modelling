#!/usr/bin/env python

import os
import sys
import uuid
import time
import logging
import json
from datetime import datetime

from collections import defaultdict, Counter
import numpy as np
import pandas as pd

from google.oauth2 import service_account
from google.cloud import bigquery

from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM, Masking, Concatenate, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import joblib as jl
from joblib import Parallel, delayed

# -----------------------------------------------------------------------------
# Configuration and Hyperparameters
# -----------------------------------------------------------------------------
SERVICE_ACCOUNT_FILE = "service-account-key.json"
MAX_SEQ_LENGTH = 20
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
RANDOM_STATE = 12
BQ_DATE_RANGE = ("20250701", "20260131")
TIME_SPLIT_FRACTION = 0.8

# BigQuery chunked download page size (rows per page, keeps memory bounded)
BQ_PAGE_SIZE = 100_000

# Inference batch size for imputation (controls peak memory during predict)
IMPUTATION_BATCH_SIZE = 512

# Fixed feature dimension via hashing — bounds memory regardless of vocabulary.
# Higher = more discriminative power but more RAM.
FEATURE_DIM = 4096

# Define keys that directly reveal marketing attribution or act as unique
# identifiers which would cause the model to "cheat" rather than learn behavior.
LEAKY_KEYS = {
    'campaign', 'medium', 'source', 'term', 'content',
    'gclid', 'fbclid', 'dclid', 'gclsrc', 'wbraid', 'gbraid',
    'manual_campaign_id', 'manual_campaign_name',
    'traffic_source_name', 'traffic_source_medium', 'traffic_source_source',
    'click_id', 'session_id',
}

# Event names strongly correlated with specific channels — bucket these
# to prevent soft leakage while retaining behavioural signal.
CHANNEL_CORRELATED_EVENTS = {
    'ad_click', 'ad_impression', 'ad_reward', 'ad_query',
    'campaign_details', 'dynamic_link_app_open', 'dynamic_link_first_open',
    'firebase_campaign', 'os_update', 'app_store_refund',
    'app_store_subscription_cancel', 'app_store_subscription_convert',
    'app_store_subscription_renew',
}

# Configure logging early, before any other code runs.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _init_seeds():
    """Set random seeds for reproducibility."""
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


def _make_artifact_dir():
    """Create the artifacts directory and return a unique run ID."""
    run_id = uuid.uuid4().hex[:8]
    artifact_dir = os.path.join("models", run_id)
    os.makedirs(artifact_dir, exist_ok=True)
    return run_id, artifact_dir


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def is_missing(channel):
    """Return True if the channel value is absent or represents a missing state."""
    if channel is None:
        return True
    channel_str = str(channel).strip().lower()
    return channel_str in {
        "(none)", "", "unknown", "undefined", "null", "unassigned", "nan"
    }


def bucket_numeric(value, bins=(0, 1, 10, 100, 1000, 10000)):
    """Bucket a numeric value into a categorical string."""
    if value is None:
        return "unknown"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "unknown"
    for b in bins:
        if value <= b:
            return f"<={b}"
    return f">{bins[-1]}"


def _safe_params_list(raw_params):
    """
    Normalise the all_params field from BigQuery into a list of dicts.
    Handles None, list-of-Row, list-of-dict, and other representations
    that vary across google-cloud-bigquery / pyarrow versions.
    """
    if raw_params is None:
        return []
    result = []
    try:
        for p in raw_params:
            if isinstance(p, dict):
                result.append(p)
            else:
                # google.cloud.bigquery Row or StructScalar — convert via dict()
                result.append(dict(p))
    except (TypeError, ValueError):
        return []
    return result


def extract_event_features(e):
    """
    Extract features from a single event dictionary.

    Mitigations applied:
      - event_name is bucketed for channel-correlated names to prevent soft
        leakage while retaining general behavioural signal.
      - Raw event_date replaced with day_of_week and hour_of_day for
        generalisable temporal signal (avoids unique-per-day features).
      - Attribution-related event params filtered via LEAKY_KEYS.
    """
    feat = Counter()

    # --- Event name (bucketed to prevent soft leakage) ---
    event_name = (e.get('event_name') or 'unknown').lower()
    if event_name in CHANNEL_CORRELATED_EVENTS:
        feat["event:__channel_correlated__"] += 1
    else:
        feat[f"event:{event_name}"] += 1

    # --- Temporal features (generalisable, not unique per date) ---
    event_ts = e.get('event_timestamp')
    if event_ts is not None:
        try:
            ts_sec = int(event_ts) / 1_000_000  # GA4 timestamps are microseconds
            dt = datetime.utcfromtimestamp(ts_sec)
            feat[f"day_of_week:{dt.weekday()}"] += 1
            feat[f"hour_of_day:{dt.hour}"] += 1
        except (ValueError, TypeError, OSError):
            feat["day_of_week:unknown"] += 1
            feat["hour_of_day:unknown"] += 1
    else:
        feat["day_of_week:unknown"] += 1
        feat["hour_of_day:unknown"] += 1

    # --- Event value ---
    if e.get('event_value_in_usd') is not None:
        feat[f"event_value_bucket:{bucket_numeric(e['event_value_in_usd'])}"] += 1
    else:
        feat["event_value_bucket:unknown"] += 1

    # --- Device and Geo Context ---
    feat[f"device:{e.get('device_category') or 'unknown'}"] += 1
    feat[f"os:{e.get('device_os') or 'unknown'}"] += 1
    feat[f"mobile_brand:{e.get('mobile_brand_name') or 'unknown'}"] += 1
    feat[f"mobile_model:{e.get('mobile_model_name') or 'unknown'}"] += 1
    feat[f"os_version:{e.get('operating_system_version') or 'unknown'}"] += 1
    feat[f"language:{e.get('language') or 'unknown'}"] += 1
    feat[f"browser:{e.get('browser') or 'unknown'}"] += 1
    feat[f"browser_version:{e.get('browser_version') or 'unknown'}"] += 1

    feat[f"geo_country:{e.get('geo_country') or 'unknown'}"] += 1
    feat[f"geo_city:{e.get('geo_city') or 'unknown'}"] += 1
    feat[f"geo_continent:{e.get('geo_continent') or 'unknown'}"] += 1
    feat[f"geo_region:{e.get('geo_region') or 'unknown'}"] += 1

    feat[f"app_id:{e.get('app_id') or 'unknown'}"] += 1
    feat[f"app_version:{e.get('app_version') or 'unknown'}"] += 1

    # --- Platform ---
    feat[f"platform:{e.get('platform') or 'unknown'}"] += 1

    # --- E-commerce Context (behavioural, safe) ---
    if e.get('purchase_revenue_in_usd') is not None:
        feat[f"purchase_revenue_bucket:{bucket_numeric(e['purchase_revenue_in_usd'])}"] += 1
    else:
        feat["purchase_revenue_bucket:unknown"] += 1

    feat[f"active_user:{str(e.get('is_active_user'))}"] += 1

    # --- Custom event parameters (with leakage protection) ---
    all_params = _safe_params_list(e.get('all_params'))
    for param in all_params:
        key = str(param.get('param_key', '')).lower()

        # Skip attribution-related keys
        if key in LEAKY_KEYS or 'utm_' in key or 'click_id' in key:
            continue

        val_str = param.get('param_str_val')
        val_int = param.get('param_int_val')

        if val_str is not None:
            feat[f"param:{key}:{val_str}"] += 1
        elif val_int is not None:
            feat[f"param:{key}:{val_int}"] += 1

    return feat


# -----------------------------------------------------------------------------
# tf.data Pipeline (replaces Keras Sequence — compatible with all TF 2.x)
# -----------------------------------------------------------------------------
def _build_tf_dataset(X_agg_sparse, X_seq_sparse_list, y_categorical,
                      max_seq_len, feature_dim, batch_size, shuffle=True):
    """
    Build a tf.data.Dataset from pre-computed sparse matrices.

    Uses a generator that densifies one sample at a time so peak memory stays
    proportional to batch_size * feature_dim rather than dataset * feature_dim.
    """
    n_samples = y_categorical.shape[0]
    num_classes = y_categorical.shape[1]

    def _generator():
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            # Aggregated branch — single dense row
            agg_dense = np.asarray(
                X_agg_sparse[idx].todense()
            ).flatten().astype(np.float32)

            # Sequence branch — zero-padded dense matrix
            seq_dense = np.zeros((max_seq_len, feature_dim), dtype=np.float32)
            seq_sparse = X_seq_sparse_list[idx]
            if seq_sparse is not None:
                seq_mat = np.asarray(seq_sparse.todense()).astype(np.float32)
                length = min(seq_mat.shape[0], max_seq_len)
                seq_dense[:length, :] = seq_mat[:length, :]

            y = y_categorical[idx].astype(np.float32)
            yield (agg_dense, seq_dense), y

    output_signature = (
        (
            tf.TensorSpec(shape=(feature_dim,), dtype=tf.float32),
            tf.TensorSpec(shape=(max_seq_len, feature_dim), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(
        _generator, output_signature=output_signature
    )

    if shuffle:
        # Keep buffer small relative to feature_dim to stay within RAM.
        # At FEATURE_DIM=4096, each sample ≈ 330 KB, so 2048 buffer ≈ 660 MB.
        ds = ds.shuffle(
            buffer_size=min(n_samples, 2048), seed=RANDOM_STATE
        )

    ds = (
        ds
        .batch(batch_size, drop_remainder=False)
        .map(
            lambda x, y: (
                {'aggregated_features': x[0], 'sequence_features': x[1]},
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


# -----------------------------------------------------------------------------
# BigQuery and Data Processing Functions
# -----------------------------------------------------------------------------
def setup_bigquery_client():
    """Set up and return a BigQuery client using the service account."""
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE
    )
    client = bigquery.Client(
        credentials=credentials, project=credentials.project_id
    )
    logger.info("=" * 50)
    logger.info(
        "STEP 1: BigQuery client created for project: %s",
        credentials.project_id,
    )
    logger.info("=" * 50)
    return client


def query_event_data(client):
    """
    Query event-level data from BigQuery.

    Downloads results in pages to bound peak memory instead of loading
    the entire result set in one call.
    """
    query = f"""
    SELECT
      event_date,
      event_timestamp,
      event_name,
      (SELECT value.string_value FROM UNNEST(event_params)
       WHERE key = 'campaign' LIMIT 1) AS campaign,
      (SELECT value.string_value FROM UNNEST(event_params)
       WHERE key = 'medium' LIMIT 1) AS campaign_medium,
      (SELECT value.string_value FROM UNNEST(event_params)
       WHERE key = 'source' LIMIT 1) AS campaign_source,
      event_value_in_usd,
      user_id,
      user_pseudo_id,
      user_first_touch_timestamp,
      device.category AS device_category,
      device.operating_system AS device_os,
      device.mobile_brand_name,
      device.mobile_model_name,
      device.operating_system_version,
      device.language AS language,
      device.browser AS browser,
      device.browser_version AS browser_version,
      geo.country AS geo_country,
      geo.city AS geo_city,
      geo.continent AS geo_continent,
      geo.region AS geo_region,
      app_info.id AS app_id,
      app_info.version AS app_version,
      platform,
      ecommerce.purchase_revenue_in_usd,
      is_active_user,
      session_traffic_source_last_click.cross_channel_campaign.primary_channel_group AS channel,
      (
        SELECT ARRAY_AGG(
          STRUCT(
            ep.key AS param_key,
            ep.value.string_value AS param_str_val,
            ep.value.int_value AS param_int_val
          )
        )
        FROM UNNEST(event_params) ep
      ) AS all_params
    FROM `gtm-5vcmpn9-ntzmm.analytics_345697125.events_*`
    WHERE
      _TABLE_SUFFIX BETWEEN '{BQ_DATE_RANGE[0]}' AND '{BQ_DATE_RANGE[1]}'
      AND event_timestamp IS NOT NULL
    ORDER BY user_pseudo_id, event_timestamp
    """
    logger.info("STEP 2: Querying event-level data from BigQuery")
    logger.info("Initiating query job...")

    query_job = client.query(query)

    # Poll until the query finishes executing on BigQuery
    while not query_job.done():
        query_job.reload()
        bp = query_job.total_bytes_processed
        if bp:
            logger.info(
                "  Query running... %.1f MB processed so far.",
                bp / 1024 / 1024,
            )
        else:
            logger.info("  Query running... (preparing)")
        time.sleep(5)

    # Stream results in pages to bound memory
    logger.info("Query complete. Streaming results in pages...")
    frames = []
    page_count = 0
    rows_so_far = 0
    for page in query_job.result(page_size=BQ_PAGE_SIZE).pages:
        page_rows = [dict(row) for row in page]
        if page_rows:
            frames.append(pd.DataFrame(page_rows))
            page_count += 1
            rows_so_far += len(page_rows)
            if page_count % 5 == 0:
                logger.info(
                    "  Downloaded %s rows (%d pages)...",
                    f"{rows_so_far:,}", page_count,
                )

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    del frames  # free intermediate page memory

    logger.info("Data retrieved: %s rows across %d pages.", f"{len(df):,}", page_count)
    logger.info("=" * 50)

    if df.empty:
        raise RuntimeError(
            "BigQuery returned 0 rows — check date range and table path."
        )

    return df


def _process_user_group(user, group):
    """Process a single user's events: sort and build the journey."""
    group = group.sort_values('event_timestamp')
    journey = []
    for _, row in group.iterrows():
        journey.append({
            'user_pseudo_id': row['user_pseudo_id'],
            'event_date': row.get('event_date'),
            'event_timestamp': row['event_timestamp'],
            'event_name': row['event_name'],
            'original_channel': row['channel'],
            'final_channel': row['channel'],
            'campaign': row.get('campaign'),
            'campaign_medium': row.get('campaign_medium'),
            'campaign_source': row.get('campaign_source'),
            'event_value_in_usd': row.get('event_value_in_usd'),
            'device_category': row.get('device_category'),
            'device_os': row.get('device_os'),
            'mobile_brand_name': row.get('mobile_brand_name'),
            'mobile_model_name': row.get('mobile_model_name'),
            'operating_system_version': row.get('operating_system_version'),
            'language': row.get('language'),
            'browser': row.get('browser'),
            'browser_version': row.get('browser_version'),
            'geo_country': row.get('geo_country'),
            'geo_city': row.get('geo_city'),
            'geo_continent': row.get('geo_continent'),
            'geo_region': row.get('geo_region'),
            'app_id': row.get('app_id'),
            'app_version': row.get('app_version'),
            'platform': row.get('platform'),
            'purchase_revenue_in_usd': row.get('purchase_revenue_in_usd'),
            'is_active_user': row.get('is_active_user'),
            'all_params': row.get('all_params'),
        })
    return user, journey


def build_event_journeys(df):
    """
    Group events by user to form journeys using parallel processing.
    Uses 'loky' backend for Windows compatibility (avoids spawn re-import
    issues inherent to the 'multiprocessing' backend on Windows).
    Caps workers to avoid excessive memory duplication.
    """
    logger.info("STEP 3: Building event-level journeys for each user")
    n_workers = min(os.cpu_count() or 1, 8)
    results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_process_user_group)(user, group)
        for user, group in df.groupby('user_pseudo_id')
    )
    user_journeys = {}
    for user, journey in results:
        user_journeys[user] = journey
    logger.info("Constructed journeys for %s users.", f"{len(user_journeys):,}")
    logger.info("=" * 50)
    return user_journeys


def prepare_training_data(user_journeys):
    """
    Prepare per-event training samples from user journeys.

    Key changes from original:
      - Uses ALL events in the journey (no arbitrary first/last trimming).
      - Each event with a known channel becomes its OWN training sample,
        with a sliding context window for the sequence branch.
      - Journeys with a mix of known and missing channels contribute their
        known-channel events to training (no full-journey discard), avoiding
        selection bias from non-random missingness.
    """
    logger.info("STEP 4: Preparing per-event training data")
    X_agg_dicts = []
    X_seq_dicts = []
    y_labels = []
    sample_meta = []
    total_events_used = 0

    for user, events in user_journeys.items():
        if len(events) < 1:
            continue

        # Pre-extract features for the whole journey once
        event_features = [extract_event_features(e) for e in events]

        for idx, e in enumerate(events):
            channel = e.get('original_channel')
            if is_missing(channel):
                continue  # skip this event, but keep others in this journey

            # Aggregated: full journey context
            agg_counter = Counter()
            for feat in event_features:
                agg_counter.update(feat)

            # Sequence: sliding window centred on this event
            half_window = MAX_SEQ_LENGTH // 2
            start = max(0, idx - half_window)
            end = min(len(events), idx + half_window + 1)
            seq_list = event_features[start:end]

            X_agg_dicts.append(dict(agg_counter))
            X_seq_dicts.append(seq_list)
            y_labels.append(channel)
            sample_meta.append({
                'user_pseudo_id': user,
                'event_timestamp': e.get('event_timestamp'),
            })
            total_events_used += 1

    logger.info(
        "Training samples (events with known channels): %s",
        f"{total_events_used:,}",
    )
    if total_events_used == 0:
        raise RuntimeError(
            "No training samples found — all channels may be missing."
        )
    logger.info("=" * 50)
    return X_agg_dicts, X_seq_dicts, y_labels, sample_meta


def split_time_then_user_holdout(sample_meta, time_fraction):
    """
    Time-based split first, then a user-holdout validation subset.

    - Train: events with timestamps <= cutoff
    - Time-based validation: events with timestamps > cutoff
    - User-holdout validation: time-based validation events where the user
      was never seen in training
    """
    timestamps = np.array(
        [m.get('event_timestamp') for m in sample_meta], dtype=np.float64
    )
    cutoff = np.quantile(timestamps, time_fraction)

    idx_train = np.where(timestamps <= cutoff)[0]
    idx_val_time = np.where(timestamps > cutoff)[0]

    train_users = {
        sample_meta[i].get('user_pseudo_id') for i in idx_train
    }
    idx_val_user = [
        i for i in idx_val_time
        if sample_meta[i].get('user_pseudo_id') not in train_users
    ]

    return idx_train, idx_val_time, np.array(idx_val_user, dtype=int), cutoff


def vectorize_data_sparse(X_agg_dicts, X_seq_dicts):
    """
    Hash feature dicts into a fixed-width sparse representation using
    FeatureHasher.  Unlike DictVectorizer the output dimension is bounded
    by FEATURE_DIM regardless of how many unique features appear in the
    data, keeping memory usage predictable at scale.
    """
    logger.info("STEP 5: Hashing features to %d-dim sparse matrices", FEATURE_DIM)
    vec = FeatureHasher(n_features=FEATURE_DIM, input_type='dict')
    # FeatureHasher is stateless — no fit() step required.

    # Transform aggregated data → single CSR matrix
    X_agg_sparse = vec.transform(X_agg_dicts)

    # Transform sequence data → list of small CSR matrices
    X_seq_sparse_list = []
    for seq in X_seq_dicts:
        if not seq:
            X_seq_sparse_list.append(None)
        else:
            X_seq_sparse_list.append(vec.transform(seq))

    logger.info("Feature dimension (hashed): %s", f"{FEATURE_DIM:,}")
    logger.info("=" * 50)
    return vec, X_agg_sparse, X_seq_sparse_list


def encode_labels(y_labels):
    """
    Encode string labels into one-hot vectors and compute balanced
    per-class weights to handle skewed channel distributions.
    """
    logger.info("STEP 6: Encoding labels and computing class weights")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    num_classes = len(le.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    # Balanced class weights to handle channel distribution skew
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=y_encoded,
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}

    logger.info("Classes (%d): %s", num_classes, list(le.classes_))
    logger.info("Class weights: %s", class_weight_dict)
    logger.info("=" * 50)
    return le, y_categorical, num_classes, class_weight_dict


def build_model(input_dim, seq_length, seq_features, num_classes, learning_rate):
    """
    Build a dual-input model with aggregated (Dense) and sequential (LSTM)
    branches. Includes BatchNormalization for stable training on high-volume data.
    """
    logger.info("STEP 7: Building dual-input deep learning model")

    # Aggregated features branch
    input_agg = Input(shape=(input_dim,), name='aggregated_features')
    x_agg = Dense(256, activation='relu')(input_agg)
    x_agg = BatchNormalization()(x_agg)
    x_agg = Dropout(0.3)(x_agg)
    x_agg = Dense(128, activation='relu')(x_agg)
    x_agg = Dropout(0.2)(x_agg)

    # Sequence branch
    input_seq = Input(shape=(seq_length, seq_features), name='sequence_features')
    x_seq = Masking(mask_value=0.0)(input_seq)
    x_seq = LSTM(128, return_sequences=False)(x_seq)
    x_seq = Dropout(0.2)(x_seq)

    # Merge
    merged = Concatenate()([x_agg, x_seq])
    x_merged = Dense(128, activation='relu')(merged)
    x_merged = BatchNormalization()(x_merged)
    x_merged = Dropout(0.3)(x_merged)
    x_merged = Dense(64, activation='relu')(x_merged)
    output = Dense(num_classes, activation='softmax')(x_merged)

    model = Model(inputs=[input_agg, input_seq], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary(print_fn=lambda line: logger.info(line))
    logger.info("=" * 50)
    return model


def impute_missing_channels(user_journeys, vec, model, le, max_seq_length):
    """
    Predict and impute missing channels PER EVENT (not per user).

    For each event with a missing channel, builds a context window from the
    surrounding events and runs the model to get the most likely channel
    for that specific position in the journey.
    """
    logger.info("STEP 10: Imputing missing channels (per-event)")
    feature_dim = vec.n_features
    imputed_event_count = 0

    # Collect all (user, event_idx) pairs that need imputation
    imputation_queue = []
    for user, events in user_journeys.items():
        for idx, e in enumerate(events):
            if is_missing(e.get('original_channel')):
                imputation_queue.append((user, idx))

    total_to_impute = len(imputation_queue)
    logger.info("Events needing imputation: %s", f"{total_to_impute:,}")

    if total_to_impute == 0:
        logger.info("Nothing to impute.")
        logger.info("=" * 50)
        return user_journeys

    # Cache per-user feature vectors to avoid redundant extraction
    _user_features_cache = {}

    def _get_user_features(user):
        if user not in _user_features_cache:
            _user_features_cache[user] = [
                extract_event_features(ev) for ev in user_journeys[user]
            ]
        return _user_features_cache[user]

    # Process in batches
    users_updated = set()
    for batch_start in range(0, total_to_impute, IMPUTATION_BATCH_SIZE):
        batch = imputation_queue[batch_start:batch_start + IMPUTATION_BATCH_SIZE]
        agg_dicts = []
        seq_dicts = []

        for user, event_idx in batch:
            event_features = _get_user_features(user)

            # Aggregated: full journey
            agg_counter = Counter()
            for feat in event_features:
                agg_counter.update(feat)
            agg_dicts.append(dict(agg_counter))

            # Sequence: window centred on this event
            half_window = max_seq_length // 2
            start = max(0, event_idx - half_window)
            end = min(len(event_features), event_idx + half_window + 1)
            seq_dicts.append(event_features[start:end])

        # Vectorize this batch
        X_agg_chunk = vec.transform(agg_dicts).toarray().astype(np.float32)
        X_seq_chunk = np.zeros(
            (len(batch), max_seq_length, feature_dim), dtype=np.float32
        )
        for i, seq in enumerate(seq_dicts):
            seq_mat = vec.transform(seq).toarray().astype(np.float32)
            length = min(seq_mat.shape[0], max_seq_length)
            X_seq_chunk[i, :length, :] = seq_mat[:length, :]

        # Predict
        probs = model.predict(
            {
                'aggregated_features': X_agg_chunk,
                'sequence_features': X_seq_chunk,
            },
            batch_size=IMPUTATION_BATCH_SIZE,
            verbose=0,
        )
        preds = np.argmax(probs, axis=1)
        predicted_channels = le.inverse_transform(preds)

        # Apply predictions back to the journey data
        for j, (user, event_idx) in enumerate(batch):
            user_journeys[user][event_idx]['final_channel'] = predicted_channels[j]
            users_updated.add(user)
            imputed_event_count += 1

        batch_num = batch_start // IMPUTATION_BATCH_SIZE
        if batch_num % 20 == 0:
            logger.info(
                "  Imputed %s / %s events...",
                f"{min(batch_start + len(batch), total_to_impute):,}",
                f"{total_to_impute:,}",
            )

    del _user_features_cache  # free cache
    logger.info(
        "Imputation complete: %s events across %s users.",
        f"{imputed_event_count:,}", f"{len(users_updated):,}",
    )
    logger.info("=" * 50)
    return user_journeys


def export_imputed_events(user_journeys, artifact_dir):
    """Export imputed event-level data to a CSV inside the artifact directory."""
    logger.info("STEP 11: Exporting imputed event-level data")
    cols = [
        'user_pseudo_id', 'event_timestamp', 'event_name',
        'original_channel', 'final_channel',
    ]

    rows = []
    for events in user_journeys.values():
        for e in events:
            rows.append({c: e.get(c) for c in cols})

    if not rows:
        logger.info("No data to export.")
        logger.info("=" * 50)
        return

    imputed_df = pd.DataFrame(rows, columns=cols)
    filename = os.path.join(artifact_dir, "imputed_events.csv")

    try:
        imputed_df.to_csv(filename, index=False)
        logger.info("Saved %s events to '%s'", f"{len(imputed_df):,}", filename)
    except OSError as exc:
        logger.error("Failed to write CSV: %s", exc)
        raise

    logger.info("=" * 50)


def save_artifacts(artifact_dir, vec, le, class_weight_dict, run_config):
    """
    Persist the DictVectorizer, LabelEncoder, class weights and run config
    alongside the model so it can be loaded for inference without retraining.
    """
    logger.info("STEP 9: Saving vectorizer, encoder and run config")
    jl.dump(vec, os.path.join(artifact_dir, "vectorizer.joblib"))
    jl.dump(le, os.path.join(artifact_dir, "label_encoder.joblib"))

    with open(os.path.join(artifact_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    logger.info("Artifacts saved to '%s/'", artifact_dir)
    logger.info("=" * 50)


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
def main():
    start_time = time.time()
    _init_seeds()
    run_id, artifact_dir = _make_artifact_dir()
    model_path = os.path.join(artifact_dir, "model.h5")

    logger.info("Run ID: %s  |  Artifacts: %s/", run_id, artifact_dir)
    logger.info("=" * 50)

    # ------------------------------------------------------------------
    # STEP 1–2: BigQuery
    # ------------------------------------------------------------------
    client = setup_bigquery_client()
    df = query_event_data(client)

    # ------------------------------------------------------------------
    # STEP 3: Build journeys
    # ------------------------------------------------------------------
    user_journeys = build_event_journeys(df)
    del df  # free the raw DataFrame

    # ------------------------------------------------------------------
    # STEP 4: Prepare per-event training data
    # ------------------------------------------------------------------
    X_agg_dicts, X_seq_dicts, y_labels, sample_meta = prepare_training_data(
        user_journeys
    )

    # ------------------------------------------------------------------
    # STEP 5: Vectorize
    # ------------------------------------------------------------------
    vec, X_agg_sparse, X_seq_sparse_list = vectorize_data_sparse(
        X_agg_dicts, X_seq_dicts
    )
    del X_agg_dicts, X_seq_dicts  # free dicts, sparse copies remain

    # ------------------------------------------------------------------
    # STEP 6: Encode labels + class weights
    # ------------------------------------------------------------------
    le, y_categorical, num_classes, class_weight_dict = encode_labels(y_labels)

    # ------------------------------------------------------------------
    # Time-based split with user-holdout validation subset
    # ------------------------------------------------------------------
    logger.info("STEP 6b: Time-based split + user-holdout validation")
    idx_train, idx_val_time, idx_val_user, cutoff_ts = (
        split_time_then_user_holdout(sample_meta, TIME_SPLIT_FRACTION)
    )
    train_users = {
        sample_meta[i].get('user_pseudo_id') for i in idx_train
    }
    val_time_users = {
        sample_meta[i].get('user_pseudo_id') for i in idx_val_time
    }
    overlap_users = train_users.intersection(val_time_users)
    logger.info(
        "Time split cutoff (unix micros): %s", f"{int(cutoff_ts):,}"
    )
    logger.info(
        "Train users: %s  |  Val-time users: %s  |  Overlap users: %s",
        f"{len(train_users):,}", f"{len(val_time_users):,}",
        f"{len(overlap_users):,}",
    )

    def _subset_sparse(matrix, idxs):
        return matrix[idxs]

    def _subset_list(lst, idxs):
        return [lst[i] for i in idxs]

    X_agg_train = _subset_sparse(X_agg_sparse, idx_train)
    X_seq_train = _subset_list(X_seq_sparse_list, idx_train)
    y_train = y_categorical[idx_train]

    X_agg_val_time = _subset_sparse(X_agg_sparse, idx_val_time)
    X_seq_val_time = _subset_list(X_seq_sparse_list, idx_val_time)
    y_val_time = y_categorical[idx_val_time]

    X_agg_val_user = _subset_sparse(X_agg_sparse, idx_val_user)
    X_seq_val_user = _subset_list(X_seq_sparse_list, idx_val_user)
    y_val_user = y_categorical[idx_val_user]

    del X_agg_sparse, X_seq_sparse_list  # free full copies
    logger.info(
        "Train samples: %s  |  Val-time samples: %s  |  Val-user samples: %s",
        f"{len(y_train):,}", f"{len(y_val_time):,}",
        f"{len(y_val_user):,}",
    )
    if len(y_val_user) == 0:
        logger.warning(
            "Val-user set is empty; no users are exclusive to validation."
        )
    logger.info("=" * 50)

    # ------------------------------------------------------------------
    # Build tf.data datasets
    # ------------------------------------------------------------------
    feature_dim = vec.n_features

    train_ds = _build_tf_dataset(
        X_agg_train, X_seq_train, y_train,
        max_seq_len=MAX_SEQ_LENGTH,
        feature_dim=feature_dim,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_time_ds = _build_tf_dataset(
        X_agg_val_time, X_seq_val_time, y_val_time,
        max_seq_len=MAX_SEQ_LENGTH,
        feature_dim=feature_dim,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    val_user_ds = None
    if len(y_val_user) > 0:
        val_user_ds = _build_tf_dataset(
            X_agg_val_user, X_seq_val_user, y_val_user,
            max_seq_len=MAX_SEQ_LENGTH,
            feature_dim=feature_dim,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

    # ------------------------------------------------------------------
    # STEP 7: Build model
    # ------------------------------------------------------------------
    model = build_model(
        input_dim=feature_dim,
        seq_length=MAX_SEQ_LENGTH,
        seq_features=feature_dim,
        num_classes=num_classes,
        learning_rate=LEARNING_RATE,
    )

    # ------------------------------------------------------------------
    # STEP 8: Train
    # ------------------------------------------------------------------
    logger.info("STEP 8: Training the model")
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_time_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )
    logger.info("=" * 50)

    # Save model
    model.save(model_path)
    logger.info("Model saved to '%s'", model_path)
    logger.info("=" * 50)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    loss, accuracy = model.evaluate(val_time_ds)
    logger.info(
        "Val-time Loss: %.4f  |  Val-time Accuracy: %.2f%%",
        loss, accuracy * 100,
    )
    val_user_loss = None
    val_user_accuracy = None
    if val_user_ds is not None:
        val_user_loss, val_user_accuracy = model.evaluate(val_user_ds)
        logger.info(
            "Val-user Loss: %.4f  |  Val-user Accuracy: %.2f%%",
            val_user_loss, val_user_accuracy * 100,
        )
    logger.info("=" * 50)

    # ------------------------------------------------------------------
    # STEP 9: Save supporting artifacts
    # ------------------------------------------------------------------
    run_config = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "bq_date_range": BQ_DATE_RANGE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "epochs_configured": EPOCHS,
        "epochs_trained": len(history.history['loss']),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "random_state": RANDOM_STATE,
        "num_classes": num_classes,
        "classes": list(le.classes_),
        "class_weights": class_weight_dict,
        "feature_dim": feature_dim,
        "train_samples": int(len(y_train)),
        "val_time_samples": int(len(y_val_time)),
        "val_time_loss": float(loss),
        "val_time_accuracy": float(accuracy),
        "val_user_samples": int(len(y_val_user)),
        "val_user_loss": (
            float(val_user_loss) if val_user_loss is not None else None
        ),
        "val_user_accuracy": (
            float(val_user_accuracy) if val_user_accuracy is not None else None
        ),
        "time_split_fraction": TIME_SPLIT_FRACTION,
        "time_split_cutoff_ts": int(cutoff_ts),
    }
    save_artifacts(artifact_dir, vec, le, class_weight_dict, run_config)

    # ------------------------------------------------------------------
    # STEP 10: Impute missing channels (per-event)
    # ------------------------------------------------------------------
    user_journeys = impute_missing_channels(
        user_journeys, vec, model, le, MAX_SEQ_LENGTH
    )

    # ------------------------------------------------------------------
    # STEP 11: Export
    # ------------------------------------------------------------------
    export_imputed_events(user_journeys, artifact_dir)

    elapsed = time.time() - start_time
    logger.info(
        "Pipeline complete.  Run ID: %s  |  Total runtime: %.1fs",
        run_id, elapsed,
    )


if __name__ == "__main__":
    main()
