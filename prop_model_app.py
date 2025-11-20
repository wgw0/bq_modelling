#!/usr/bin/env python
import os
import uuid
import time
import logging
import json

from collections import defaultdict, Counter
import numpy as np
import pandas as pd

from google.oauth2 import service_account
from google.cloud import bigquery

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Masking, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping

from joblib import Parallel, delayed

# -----------------------------------------------------------------------------
# Configuration and Hyperparameters
# -----------------------------------------------------------------------------
SERVICE_ACCOUNT_FILE = "service-account-key.json"
MAX_SEQ_LENGTH = 20
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RANDOM_STATE = 12
BQ_DATE_RANGE = ("20241001", "20250228")

# --- [AMENDMENT: LEAKAGE PROTECTION] ---
# Define keys that directly reveal marketing attribution or act as unique identifiers
# which would cause the model to "cheat" rather than learn behavior.
LEAKY_KEYS = {
    'campaign', 'medium', 'source', 'term', 'content',
    'gclid', 'fbclid', 'dclid', 'gclsrc', 'wbraid', 'gbraid',
    'manual_campaign_id', 'manual_campaign_name', 
    'traffic_source_name', 'traffic_source_medium', 'traffic_source_source',
    'click_id', 'session_id'
}

# random seeds for reproducibility.
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Generate a unique model ID and model checkpoint path.
unique_model_id = uuid.uuid4().hex[:8]
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
MODEL_CHECKPOINT_PATH = os.path.join(model_dir, f"model_{unique_model_id}.h5")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def is_missing(channel):
    if channel is None:
        return True
    channel_str = str(channel).strip().lower()
    return channel_str in ["(none)", "", "unknown", "undefined", "null", "unassigned"]

def bucket_numeric(value, bins=[0, 1, 10, 100, 1000, 10000]):
    if value is None:
        return "unknown"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "unknown"
    for b in bins:
        if value <= b:
            return f"<= {b}"
    return f"> {bins[-1]}"

def extract_event_features(e):
    """
    Extracts features from an event dictionary.
    AMENDED: Aggressively filters out attribution-related keys to prevent label leakage.
    """
    feat = Counter()
    
    # Contextual features
    feat[f"event:{e.get('event_name') or 'unknown'}"] += 1
    feat[f"event_date:{e.get('event_date') or 'unknown'}"] += 1

    if e.get('event_value_in_usd') is not None:
        bucket = bucket_numeric(e['event_value_in_usd'])
        feat[f"event_value_bucket:{bucket}"] += 1
    else:
        feat["event_value_bucket:unknown"] += 1

    # Device and Geo Context
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
    
    # Platform
    feat[f"platform:{e.get('platform') or 'unknown'}"] += 1

    # E-commerce Context (Safe, behavioral)
    if e.get('purchase_revenue_in_usd') is not None:
        bucket = bucket_numeric(e['purchase_revenue_in_usd'])
        feat[f"purchase_revenue_bucket:{bucket}"] += 1
    else:
        feat["purchase_revenue_bucket:unknown"] += 1

    feat[f"active_user:{str(e.get('is_active_user'))}"] += 1

    # Process custom event parameters with LEAKAGE PROTECTION
    all_params = e.get('all_params', [])
    if all_params is None:
        all_params = []
        
    for param in all_params:
        key = param.get('param_key', '').lower()
        
        # --- AMENDED: Filter Blacklisted Keys ---
        # Check if key is in blacklist or contains 'utm_'
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
# Data Generator (Memory Optimization)
# -----------------------------------------------------------------------------
class AttributionDataGenerator(Sequence):
    """
    Keras Data Generator to handle data in batches.
    Prevents OOM errors by converting sparse dictionaries to dense arrays 
    only when needed for the specific batch.
    """
    def __init__(self, X_agg_dicts, X_seq_dicts, y_categorical, vectorizer, batch_size=32, max_seq_len=20, shuffle=True):
        self.X_agg_dicts = X_agg_dicts
        self.X_seq_dicts = X_seq_dicts
        self.y = y_categorical
        self.vec = vectorizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.y))
        self.feature_dim = len(self.vec.feature_names_)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Select data for this batch
        batch_agg_dicts = [self.X_agg_dicts[k] for k in indexes]
        batch_seq_dicts = [self.X_seq_dicts[k] for k in indexes]
        batch_y = self.y[indexes]

        # 1. Transform Aggregated Branch (Sparse -> Dense for this batch only)
        X_agg_dense = self.vec.transform(batch_agg_dicts).toarray()
        
        # 2. Transform Sequence Branch
        # Initialize batch tensor
        X_seq_dense = np.zeros((self.batch_size, self.max_seq_len, self.feature_dim))
        
        for i, seq in enumerate(batch_seq_dicts):
            if not seq: continue
            # Transform list of dicts to dense matrix
            seq_mat = self.vec.transform(seq).toarray()
            
            # Truncate or Pad
            length = min(len(seq_mat), self.max_seq_len)
            X_seq_dense[i, :length, :] = seq_mat[:length, :]

        return {'aggregated_features': X_agg_dense, 'sequence_features': X_seq_dense}, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# -----------------------------------------------------------------------------
# BigQuery and Data Processing Functions
# -----------------------------------------------------------------------------
def setup_bigquery_client():
    """Set up and return a BigQuery client using the service account."""
    try:
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        logging.info("=" * 30)
        logging.info(f"STEP 0: BigQuery client created for project: {credentials.project_id}")
        logging.info("=" * 30 + "\n")
        return client
    except Exception as e:
        logging.error("Error setting up BigQuery client.", exc_info=True)
        raise e

def query_event_data(client):
    """Query event-level data from BigQuery using the specified date range."""
    query = f"""
    SELECT
      event_date,
      event_timestamp,
      event_name,
      (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'campaign' LIMIT 1) AS campaign,
      (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'medium' LIMIT 1) AS campaign_medium,
      (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'source' LIMIT 1) AS campaign_source,
      event_value_in_usd,
      user_id,
      user_pseudo_id,
      user_first_touch_timestamp,
      device.category AS device_category,
      device.operating_system AS device_os,
      device.mobile_brand_name,
      device.mobile_model_name,
      device.operating_system_version,
      device.language,
      device.browser,
      device.browser_version,
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
    try:
        logging.info("STEP 1: Querying event-level data from BigQuery")
        logging.info("Running query...")
        df = client.query(query).to_dataframe()
        logging.info(f"Data retrieved: {len(df)} rows.")
        logging.info("=" * 30 + "\n")
        return df
    except Exception as e:
        logging.error("Error executing BigQuery query.", exc_info=True)
        raise e

def process_user_group(user, group):
    """Process a single user's events: sort and build the journey for that user."""
    group = group.sort_values('event_timestamp')
    journey = []
    for _, row in group.iterrows():
        journey.append({
            'user_pseudo_id': row['user_pseudo_id'],
            'event_date': row['event_date'],
            'event_timestamp': row['event_timestamp'],
            'event_name': row['event_name'],
            'original_channel': row['channel'],
            'final_channel': row['channel'],
            'campaign': row['campaign'],
            'campaign_medium': row['campaign_medium'],
            'campaign_source': row['campaign_source'],
            'event_value_in_usd': row['event_value_in_usd'],
            'device_category': row['device_category'],
            'device_os': row['device_os'],
            'mobile_brand_name': row['mobile_brand_name'],
            'mobile_model_name': row['mobile_model_name'],
            'operating_system_version': row['operating_system_version'],
            'language': row['device.language'] if 'device.language' in row else row['language'],
            'browser': row['browser'],
            'browser_version': row['browser_version'],
            'geo_country': row['geo_country'],
            'geo_city': row['geo_city'],
            'geo_continent': row['geo_continent'],
            'geo_region': row['geo_region'],
            'app_id': row['app_id'],
            'app_version': row['app_version'],
            'platform': row['platform'],
            'purchase_revenue_in_usd': row['purchase_revenue_in_usd'],
            'is_active_user': row['is_active_user'],
            'all_params': row.get('all_params')
        })
    return user, journey

def build_event_journeys(df):
    """Group events by user to form journeys using parallel processing."""
    logging.info("STEP 2: Building event-level journeys for each user")
    results = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(process_user_group)(user, group) for user, group in df.groupby('user_pseudo_id')
    )
    user_journeys = defaultdict(list)
    for user, journey in results:
        user_journeys[user] = journey
    logging.info(f"Constructed journeys for {len(user_journeys)} users.")
    logging.info("=" * 30 + "\n")
    return user_journeys

def prepare_training_data(user_journeys):
    """Prepare aggregated and sequential features along with labels for training."""
    logging.info("STEP 3: Preparing training data for proprietary model")
    X_agg_dicts = []
    X_seq_dicts = []
    y_labels = []
    complete_journey_count = 0

    for user, events in user_journeys.items():
        if len(events) < 2:
            continue
        if any(is_missing(e.get('original_channel')) for e in events):
            continue
        
        journey_middle = events[1:-1] if len(events) > 2 else events
        if not journey_middle:
            continue
        
        agg_counter = Counter()
        seq_list = []
        for e in journey_middle:
            features = extract_event_features(e)
            agg_counter.update(features)
            seq_list.append(features)
        
        channels = [e.get('original_channel') for e in journey_middle]
        dominant_channel = Counter(channels).most_common(1)[0][0]
        
        X_agg_dicts.append(dict(agg_counter))
        X_seq_dicts.append(seq_list)
        y_labels.append(dominant_channel)
        complete_journey_count += 1

    logging.info(f"Number of complete journeys for training: {complete_journey_count}")
    logging.info("=" * 30 + "\n")
    return X_agg_dicts, X_seq_dicts, y_labels

def fit_vectorizer_only(X_agg_dicts, X_seq_dicts):
    logging.info("STEP 4: Fitting vectorizer (establishing vocabulary)")
    vec = DictVectorizer(sparse=True)
    all_event_dicts = []
    for seq in X_seq_dicts:
        all_event_dicts.extend(seq)
    all_event_dicts.extend(X_agg_dicts)
    
    vec.fit(all_event_dicts)
    logging.info(f"Vocabulary size established: {len(vec.feature_names_)} features.")
    logging.info("=" * 30 + "\n")
    return vec

def encode_labels(y_labels):
    """Encode string labels into integers and then into one-hot vectors."""
    logging.info("STEP 5: Encoding labels")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    num_classes = len(le.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)
    logging.info(f"Classes: {le.classes_}")
    logging.info("=" * 30 + "\n")
    return le, y_categorical, num_classes

def build_model(input_dim, seq_length, seq_features, num_classes, learning_rate):
    logging.info("STEP 6: Building dual-input deep learning model")
    
    # Aggregated features branch.
    input_agg = Input(shape=(input_dim,), name='aggregated_features')
    x_agg = Dense(128, activation='relu')(input_agg)
    x_agg = Dropout(0.2)(x_agg)
    
    # Sequence branch.
    input_seq = Input(shape=(seq_length, seq_features), name='sequence_features')
    x_seq = Masking(mask_value=0.0)(input_seq)
    x_seq = LSTM(128)(x_seq)
    
    # Combine both branches.
    merged = Concatenate()([x_agg, x_seq])
    x_merged = Dense(64, activation='relu')(merged)
    output = Dense(num_classes, activation='softmax')(x_merged)
    
    model = Model(inputs=[input_agg, input_seq], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary(print_fn=lambda x: logging.info(x))
    logging.info("=" * 30 + "\n")
    return model

def impute_missing_channels(user_journeys, vec, model, le, max_seq_length):
    """
    Predict and impute missing channels.
    """
    logging.info("STEP 7: Imputing missing channels for journeys with missing values")
    imputed_journeys_count = 0
    
    # Collect data for batch prediction to speed up inference
    prediction_batch_agg = []
    prediction_batch_seq = []
    prediction_user_ptrs = [] # Track which user belongs to which batch item
    
    users_needing_imputation = []
    
    # Identify users needing imputation
    for user, events in user_journeys.items():
        if not any(is_missing(e.get('original_channel')) for e in events):
            continue
        journey_middle = events[1:-1] if len(events) > 2 else events
        if not journey_middle:
            continue
        users_needing_imputation.append((user, journey_middle))

    logging.info(f"Found {len(users_needing_imputation)} users needing imputation.")

    # Process in chunks to manage memory during inference
    batch_size = 1000
    for i in range(0, len(users_needing_imputation), batch_size):
        chunk = users_needing_imputation[i:i+batch_size]
        
        agg_dicts = []
        seq_dicts = []
        
        for user, journey_middle in chunk:
            agg_counter = Counter()
            seq_list = []
            for e in journey_middle:
                features = extract_event_features(e)
                agg_counter.update(features)
                seq_list.append(features)
            agg_dicts.append(dict(agg_counter))
            seq_dicts.append(seq_list)
        
        # Vectorize this chunk
        X_agg_chunk = vec.transform(agg_dicts).toarray()
        
        feature_dim = len(vec.feature_names_)
        X_seq_chunk = np.zeros((len(chunk), max_seq_length, feature_dim))
        
        for idx, seq in enumerate(seq_dicts):
            seq_mat = vec.transform(seq).toarray()
            length = min(len(seq_mat), max_seq_length)
            X_seq_chunk[idx, :length, :] = seq_mat[:length, :]
            
        # Predict
        probs = model.predict({'aggregated_features': X_agg_chunk, 'sequence_features': X_seq_chunk}, verbose=0)
        preds = np.argmax(probs, axis=1)
        predicted_channels = le.inverse_transform(preds)
        
        # Apply back to data
        for j, (user, _) in enumerate(chunk):
            predicted_channel = predicted_channels[j]
            events = user_journeys[user]
            updated = False
            for e in events:
                if is_missing(e.get('original_channel')):
                    e['final_channel'] = predicted_channel
                    updated = True
            if updated:
                imputed_journeys_count += 1

    logging.info(f"Number of journeys with missing channels updated: {imputed_journeys_count}")
    logging.info("=" * 30 + "\n")
    return user_journeys

def export_imputed_events(user_journeys):
    """Export imputed event-level data to a CSV file."""
    logging.info("STEP 8: Preparing event-level data for export")
    imputed_events = []
    for user, events in user_journeys.items():
        # Only export if we actually did something (optional, but cleaner)
        # Or export everything. Here we export everything to show full picture.
        for e in events:
            imputed_events.append(e)
    
    imputed_df = pd.DataFrame(imputed_events)
    
    # Columns to export
    cols = ['user_pseudo_id', 'event_timestamp', 'event_name', 'original_channel', 'final_channel']
    if not imputed_df.empty:
        imputed_df = imputed_df[cols]
        unique_id = uuid.uuid4().hex[:8]
        filename = f"imputed_events_{unique_id}.csv"
        imputed_df.to_csv(filename, index=False)
        logging.info(f"Imputed event-level data saved to '{filename}'")
    else:
        logging.info("No data to export.")
    logging.info("=" * 30 + "\n")

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
def main():
    start_time = time.time()
    
    # STEP 0: Set up BigQuery client.
    client = setup_bigquery_client()

    # STEP 1: Query event-level data.
    df = query_event_data(client)

    # STEP 2: Build event-level journeys using parallel processing.
    user_journeys = build_event_journeys(df)

    # STEP 3: Prepare training data (List of Dictionaries).
    X_agg_dicts, X_seq_dicts, y_labels = prepare_training_data(user_journeys)

    # STEP 4: Fit Vectorizer Only (No Matrix Creation).
    vec = fit_vectorizer_only(X_agg_dicts, X_seq_dicts)

    # STEP 5: Encode labels.
    le, y_categorical, num_classes = encode_labels(y_labels)

    # STEP 6: Split data (Indices/Lists).
    indices = np.arange(len(y_categorical))
    idx_train, idx_val = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE)

    def subset_list(data_list, idxs): return [data_list[i] for i in idxs]

    X_agg_train = subset_list(X_agg_dicts, idx_train)
    X_seq_train = subset_list(X_seq_dicts, idx_train)
    y_train = y_categorical[idx_train]
    
    X_agg_val = subset_list(X_agg_dicts, idx_val)
    X_seq_val = subset_list(X_seq_dicts, idx_val)
    y_val = y_categorical[idx_val]

    logging.info("Training and validation data lists prepared.")
    logging.info("=" * 30 + "\n")

    # Initialize Generators
    train_gen = AttributionDataGenerator(X_agg_train, X_seq_train, y_train, vec, BATCH_SIZE, MAX_SEQ_LENGTH)
    val_gen = AttributionDataGenerator(X_agg_val, X_seq_val, y_val, vec, BATCH_SIZE, MAX_SEQ_LENGTH, shuffle=False)

    # STEP 7: Build the model.
    feature_dim = len(vec.feature_names_)
    model = build_model(
        input_dim=feature_dim,
        seq_length=MAX_SEQ_LENGTH,
        seq_features=feature_dim,
        num_classes=num_classes,
        learning_rate=LEARNING_RATE
    )

    # STEP 8: Train the model using Generators.
    logging.info("STEP 8: Training the model (using Data Generators)")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    logging.info("=" * 30 + "\n")

    # Save the model checkpoint.
    model.save(MODEL_CHECKPOINT_PATH)
    logging.info(f"Model saved to '{MODEL_CHECKPOINT_PATH}'")
    logging.info("=" * 30 + "\n")

    # STEP 9: Evaluate the model.
    loss, accuracy = model.evaluate(val_gen)
    logging.info(f"STEP 9: Validation Accuracy: {accuracy * 100:.2f}%")
    logging.info("=" * 30 + "\n")

    # STEP 10: Impute missing channels (Batched Inference).
    user_journeys = impute_missing_channels(user_journeys, vec, model, le, MAX_SEQ_LENGTH)

    # STEP 11: Export event-level data.
    export_imputed_events(user_journeys)
    
    end_time = time.time()
    logging.info(f"Total Runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()