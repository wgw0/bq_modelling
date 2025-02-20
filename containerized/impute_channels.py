#!/usr/bin/env python
import os
import uuid
import time
import logging
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from google.oauth2 import service_account
from google.cloud import bigquery

import tensorflow as tf
from tensorflow.keras.models import load_model

from joblib import load

# -----------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# -----------------------------------------------------------------------------
SERVICE_ACCOUNT_FILE = os.environ.get("SERVICE_ACCOUNT_FILE", "service-account-key.json")
BQ_DATE_RANGE = (
    os.environ.get("BQ_DATE_RANGE_START", "20250301"),
    os.environ.get("BQ_DATE_RANGE_END", "20250310")
)
MAX_SEQ_LENGTH = 20

# Paths to pre-trained artifacts (update these paths if necessary)
MODEL_CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", "models/model_<ID>.h5")
VECTORIZER_PATH = os.environ.get("VECTORIZER_PATH", "models/vectorizer_<ID>.pkl")
LABEL_ENCODER_PATH = os.environ.get("LABEL_ENCODER_PATH", "models/label_encoder_<ID>.pkl")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# -----------------------------------------------------------------------------
# Helper Functions (same as in training)
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
    feat = Counter()
    feat[f"event:{e.get('event_name') or 'unknown'}"] += 1
    feat[f"event_date:{e.get('event_date') or 'unknown'}"] += 1
    feat[f"campaign:{e.get('campaign') or 'unknown'}"] += 1
    feat[f"campaign_medium:{e.get('campaign_medium') or 'unknown'}"] += 1
    feat[f"campaign_source:{e.get('campaign_source') or 'unknown'}"] += 1

    if e.get('event_value_in_usd') is not None:
        bucket = bucket_numeric(e['event_value_in_usd'])
        feat[f"event_value_bucket:{bucket}"] += 1
    else:
        feat["event_value_bucket:unknown"] += 1

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
    feat[f"geo_sub_continent:{e.get('geo_sub_continent') or 'unknown'}"] += 1
    feat[f"geo_metro:{e.get('geo_metro') or 'unknown'}"] += 1

    feat[f"app_id:{e.get('app_id') or 'unknown'}"] += 1
    feat[f"app_version:{e.get('app_version') or 'unknown'}"] += 1
    feat[f"install_store:{e.get('install_store') or 'unknown'}"] += 1
    feat[f"firebase_app_id:{e.get('firebase_app_id') or 'unknown'}"] += 1
    feat[f"install_source:{e.get('install_source') or 'unknown'}"] += 1

    feat[f"traffic_source_name:{e.get('traffic_source_name') or 'unknown'}"] += 1
    feat[f"traffic_source_medium:{e.get('traffic_source_medium') or 'unknown'}"] += 1
    feat[f"traffic_source_source:{e.get('traffic_source_source') or 'unknown'}"] += 1

    feat[f"platform:{e.get('platform') or 'unknown'}"] += 1

    if e.get('purchase_revenue_in_usd') is not None:
        bucket = bucket_numeric(e['purchase_revenue_in_usd'])
        feat[f"purchase_revenue_bucket:{bucket}"] += 1
    else:
        feat["purchase_revenue_bucket:unknown"] += 1

    feat[f"manual_campaign_id:{e.get('manual_campaign_id') or 'unknown'}"] += 1
    feat[f"manual_campaign_name:{e.get('manual_campaign_name') or 'unknown'}"] += 1
    feat[f"manual_source:{e.get('manual_source') or 'unknown'}"] += 1
    feat[f"manual_medium:{e.get('manual_medium') or 'unknown'}"] += 1

    feat[f"active_user:{str(e.get('is_active_user'))}"] += 1

    if e.get('ad_revenue_in_usd') is not None:
        bucket = bucket_numeric(e['ad_revenue_in_usd'])
        feat[f"ad_revenue_bucket:{bucket}"] += 1
    else:
        feat["ad_revenue_bucket:unknown"] += 1
    feat[f"ad_format:{e.get('ad_format') or 'unknown'}"] += 1
    feat[f"ad_source_name:{e.get('ad_source_name') or 'unknown'}"] += 1
    feat[f"ad_unit_id:{e.get('ad_unit_id') or 'unknown'}"] += 1

    all_params = e.get('all_params', [])
    if all_params is None:
        all_params = []
    for param in all_params:
        if param.get('param_str_val') is not None:
            feat[f"param:{param.get('param_key')}:{param.get('param_str_val')}"] += 1
        elif param.get('param_int_val') is not None:
            feat[f"param:{param.get('param_key')}:{param.get('param_int_val')}"] += 1

    return feat

# -----------------------------------------------------------------------------
# BigQuery and Data Processing Functions
# -----------------------------------------------------------------------------
def setup_bigquery_client():
    """Set up and return a BigQuery client using the service account."""
    try:
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        logging.info("=" * 30)
        logging.info(f"BigQuery client created for project: {credentials.project_id}")
        logging.info("=" * 30 + "\n")
        return client
    except Exception as e:
        logging.error("Error setting up BigQuery client.", exc_info=True)
        raise e

def query_event_data(client):
    """Query new event-level data from BigQuery using the specified date range."""
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
      user_ltv.revenue AS user_ltv_revenue,
      user_ltv.currency AS user_ltv_currency,
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
      geo.sub_continent AS geo_sub_continent,
      geo.metro AS geo_metro,
      app_info.id AS app_id,
      app_info.version AS app_version,
      app_info.install_store,
      app_info.firebase_app_id,
      app_info.install_source,
      traffic_source.name AS traffic_source_name,
      traffic_source.medium AS traffic_source_medium,
      traffic_source.source AS traffic_source_source,
      platform,
      ecommerce.purchase_revenue_in_usd,
      collected_traffic_source.manual_campaign_id,
      collected_traffic_source.manual_campaign_name,
      collected_traffic_source.manual_source,
      collected_traffic_source.manual_medium,
      is_active_user,
      session_traffic_source_last_click.cross_channel_campaign.primary_channel_group AS channel,
      publisher.ad_revenue_in_usd,
      publisher.ad_format,
      publisher.ad_source_name,
      publisher.ad_unit_id,
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
        logging.info("Querying new event-level data from BigQuery")
        df = client.query(query).to_dataframe()
        logging.info(f"Data retrieved: {len(df)} rows.")
        logging.info("=" * 30 + "\n")
        return df
    except Exception as e:
        logging.error("Error executing BigQuery query.", exc_info=True)
        raise e

def process_user_group(user, group):
    """Process a single user's events to build the journey."""
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
            'user_id': row['user_id'],
            'user_first_touch_timestamp': row['user_first_touch_timestamp'],
            'user_ltv_revenue': row['user_ltv_revenue'],
            'user_ltv_currency': row['user_ltv_currency'],
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
            'geo_sub_continent': row['geo_sub_continent'],
            'geo_metro': row['geo_metro'],
            'app_id': row['app_id'],
            'app_version': row['app_version'],
            'install_store': row['install_store'],
            'firebase_app_id': row['firebase_app_id'],
            'install_source': row['install_source'],
            'traffic_source_name': row['traffic_source_name'],
            'traffic_source_medium': row['traffic_source_medium'],
            'traffic_source_source': row['traffic_source_source'],
            'platform': row['platform'],
            'purchase_revenue_in_usd': row['purchase_revenue_in_usd'],
            'manual_campaign_id': row['manual_campaign_id'],
            'manual_campaign_name': row['manual_campaign_name'],
            'manual_source': row['manual_source'],
            'manual_medium': row['manual_medium'],
            'is_active_user': row['is_active_user'],
            'ad_revenue_in_usd': row['ad_revenue_in_usd'],
            'ad_format': row['ad_format'],
            'ad_source_name': row['ad_source_name'],
            'ad_unit_id': row['ad_unit_id'],
            'all_params': row.get('all_params')
        })
    return user, journey

def build_event_journeys(df):
    """Group events by user to form journeys."""
    from collections import defaultdict
    from joblib import Parallel, delayed
    logging.info("Building event-level journeys for new events")
    results = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(process_user_group)(user, group) for user, group in df.groupby('user_pseudo_id')
    )
    user_journeys = defaultdict(list)
    for user, journey in results:
        user_journeys[user] = journey
    logging.info(f"Constructed journeys for {len(user_journeys)} users.")
    logging.info("=" * 30 + "\n")
    return user_journeys

def impute_missing_channels(user_journeys, vec, model, le, max_seq_length):
    """Predict and impute missing channels for journeys with missing channel values."""
    logging.info("Imputing missing channels for journeys with missing values")
    imputed_journeys_count = 0
    for user, events in user_journeys.items():
        if not any(is_missing(e.get('original_channel')) for e in events):
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
        agg_features = vec.transform([dict(agg_counter)]).toarray()
        seq_sparse = vec.transform(seq_list)
        seq_vec = seq_sparse.toarray()
        if seq_vec.shape[0] < max_seq_length:
            pad = np.zeros((max_seq_length - seq_vec.shape[0], seq_vec.shape[1]))
            seq_vec = np.vstack([seq_vec, pad])
        else:
            seq_vec = seq_vec[:max_seq_length, :]
        seq_features = np.expand_dims(seq_vec, axis=0)
        
        predicted_proba = model.predict({'aggregated_features': agg_features, 'sequence_features': seq_features}, verbose=0)
        predicted_index = np.argmax(predicted_proba, axis=1)[0]
        predicted_channel = le.inverse_transform([predicted_index])[0]
        
        journey_updated = False
        for e in events:
            if is_missing(e.get('original_channel')):
                e['final_channel'] = predicted_channel
                journey_updated = True
        if journey_updated:
            imputed_journeys_count += 1

    logging.info(f"Number of journeys with missing channels updated: {imputed_journeys_count}")
    logging.info("=" * 30 + "\n")
    return user_journeys

def export_imputed_events(user_journeys):
    """Export imputed event-level data to a CSV file."""
    logging.info("Preparing event-level data for export")
    imputed_events = []
    for user, events in user_journeys.items():
        for e in events:
            imputed_events.append(e)
    imputed_df = pd.DataFrame(imputed_events)
    imputed_df = imputed_df[['user_pseudo_id', 'event_timestamp', 'event_name', 'original_channel', 'final_channel']]
    unique_id = uuid.uuid4().hex[:8]
    filename = f"imputed_events_{unique_id}.csv"
    imputed_df.to_csv(filename, index=False)
    logging.info(f"Imputed event-level data saved to '{filename}'")
    logging.info("=" * 30 + "\n")

# -----------------------------------------------------------------------------
# Main Inference Pipeline
# -----------------------------------------------------------------------------
def main():
    start_time = time.time()
    
    # Load the pre-trained model, vectorizer, and label encoder.
    logging.info("Loading pre-trained artifacts...")
    model = load_model(MODEL_CHECKPOINT_PATH)
    vec = load(VECTORIZER_PATH)
    le = load(LABEL_ENCODER_PATH)
    logging.info("Artifacts loaded successfully.")
    logging.info("=" * 30 + "\n")
    
    # Set up BigQuery client and query new event-level data.
    client = setup_bigquery_client()
    df = query_event_data(client)
    
    # Build event journeys.
    user_journeys = build_event_journeys(df)
    
    # Impute missing channels.
    user_journeys = impute_missing_channels(user_journeys, vec, model, le, MAX_SEQ_LENGTH)
    
    # Export the imputed events.
    export_imputed_events(user_journeys)
    
    end_time = time.time()
    logging.info(f"Total inference time (seconds): {end_time - start_time:.2f}")

if __name__ == "__main__":
    main()
