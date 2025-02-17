import os
import uuid
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
from tensorflow.keras.utils import to_categorical

# =============================================================================
# Helper Functions
# =============================================================================
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
    Extract features from a single event dictionary using a Counter.
    """
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

    return feat

# =============================================================================
# STEP 0: Set up BigQuery Client
# =============================================================================
print("=" * 30)
print("STEP 0: Setting up BigQuery client using service-account-key.json")
credentials = service_account.Credentials.from_service_account_file("service-account-key.json")
client = bigquery.Client(credentials=credentials, project=credentials.project_id)
print("Client created for project:", credentials.project_id)
print("=" * 30, "\n")

# =============================================================================
# STEP 1: Query Event-Level Data from BigQuery
# =============================================================================
print("=" * 30)
print("STEP 1: Querying event-level data from BigQuery")
query = """
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
  publisher.ad_unit_id
FROM `gtm-5vcmpn9-ntzmm.analytics_345697125.events_*`
WHERE
  _TABLE_SUFFIX BETWEEN '20241001' AND '20250228'
  AND event_timestamp IS NOT NULL
ORDER BY user_pseudo_id, event_timestamp
"""
print("Running query...")
df = client.query(query).to_dataframe()
print("Data retrieved: {} rows.".format(len(df)))
print("=" * 30, "\n")

# =============================================================================
# STEP 2: Mark Conversion Events
# =============================================================================
print("=" * 30)
print("STEP 2: Marking conversion events")
conversion_events = ['form_submit', 'form_submit_contact']
df['is_conversion'] = np.where(df['event_name'].isin(conversion_events), 1, 0)
print("Total conversion events flagged:", df['is_conversion'].sum())
print("=" * 30, "\n")

# =============================================================================
# STEP 3: Build Event-Level Journeys
# =============================================================================
print("=" * 30)
print("STEP 3: Building event-level journeys for each user")
journey_events = []
for user, group in df.groupby('user_pseudo_id'):
    group = group.sort_values('event_timestamp')
    # Loop through each event and store necessary fields.
    for idx, row in group.iterrows():
        journey_events.append({
            'user_pseudo_id': row['user_pseudo_id'],
            'event_date': row['event_date'],
            'event_timestamp': row['event_timestamp'],
            'event_name': row['event_name'],
            'original_channel': row['channel'],
            'final_channel': row['channel'],  # initially the same
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
            'language': row['language'],
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
            'ad_unit_id': row['ad_unit_id']
        })
# Group events by user to form journeys.
user_journeys = defaultdict(list)
for event in journey_events:
    user_journeys[event['user_pseudo_id']].append(event)
print("Constructed journeys for {} users.".format(len(user_journeys)))
print("=" * 30, "\n")

# =============================================================================
# STEP 4: Prepare Training Data for the Proprietary Model
# =============================================================================
print("=" * 30)
print("STEP 4: Preparing training data for proprietary model")
X_agg_dicts = []  # Aggregated features per journey
X_seq_dicts = []  # Sequence (list) of per-event feature dictionaries per journey
y_labels = []     # Dominant channel label per journey
complete_journey_count = 0

for user, events in user_journeys.items():
    # Skip journeys that are too short.
    if len(events) < 2:
        continue
    # Only use journeys with complete channel info for training.
    if any(is_missing(e.get('original_channel')) for e in events):
        continue
    # Use the "middle" events (if available) for feature aggregation.
    if len(events) > 2:
        journey_middle = events[1:-1]
    else:
        journey_middle = events
    if not journey_middle:
        continue
    agg_counter = Counter()
    seq_list = []
    for e in journey_middle:
        features = extract_event_features(e)
        agg_counter.update(features)
        seq_list.append(features)
    # Use the dominant channel from the journey middle as the label.
    channels = [e.get('original_channel') for e in journey_middle]
    dominant_channel = Counter(channels).most_common(1)[0][0]
    X_agg_dicts.append(dict(agg_counter))
    X_seq_dicts.append(seq_list)
    y_labels.append(dominant_channel)
    complete_journey_count += 1

print("Number of complete journeys for training:", complete_journey_count)
print("=" * 30, "\n")

# =============================================================================
# STEP 5: Vectorize Feature Dictionaries
# =============================================================================
print("=" * 30)
print("STEP 5: Vectorizing feature dictionaries")
# Use one DictVectorizer for both aggregated and per-event features.
vec = DictVectorizer(sparse=False)
all_event_dicts = []
for seq in X_seq_dicts:
    all_event_dicts.extend(seq)
all_event_dicts.extend(X_agg_dicts)
vec.fit(all_event_dicts)

# Transform aggregated features.
X_agg = vec.transform(X_agg_dicts)

# Transform each event in the sequence and pad/truncate sequences to a fixed length.
max_seq_length = 20  # Covers 90% of journeys as found from 'visualize_journey_events.py'
X_seq = []
for seq in X_seq_dicts:
    seq_vec = vec.transform(seq)
    if seq_vec.shape[0] < max_seq_length:
        pad = np.zeros((max_seq_length - seq_vec.shape[0], seq_vec.shape[1]))
        seq_vec = np.vstack([seq_vec, pad])
    else:
        seq_vec = seq_vec[:max_seq_length, :]
    X_seq.append(seq_vec)
X_seq = np.array(X_seq)
print("Aggregated feature shape:", X_agg.shape)
print("Sequence feature shape:", X_seq.shape)
print("=" * 30, "\n")

# =============================================================================
# STEP 6: Encode Labels
# =============================================================================
print("=" * 30)
print("STEP 6: Encoding labels")
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
num_classes = len(le.classes_)
y_categorical = to_categorical(y_encoded, num_classes=num_classes)
print("Classes:", le.classes_)
print("=" * 30, "\n")

# =============================================================================
# STEP 7: Split Data into Training and Validation Sets
# =============================================================================
print("=" * 30)
print("STEP 7: Splitting training data")
X_agg_train, X_agg_val, X_seq_train, X_seq_val, y_train, y_val = train_test_split(
    X_agg, X_seq, y_categorical, test_size=0.2, random_state=42
)
print("Training and validation data prepared.")
print("=" * 30, "\n")

# =============================================================================
# STEP 8: Build the Dual-Input Deep Learning Model
# =============================================================================
print("=" * 30)
print("STEP 8: Building dual-input deep learning model")
# Aggregated features branch.
input_agg = Input(shape=(X_agg_train.shape[1],), name='aggregated_features')
x_agg = Dense(64, activation='relu')(input_agg)
x_agg = Dropout(0.5)(x_agg)

# Sequence branch.
input_seq = Input(shape=(max_seq_length, X_seq_train.shape[2]), name='sequence_features')
x_seq = Masking(mask_value=0.0)(input_seq)
x_seq = LSTM(64)(x_seq)
x_seq = Dropout(0.5)(x_seq)

# Combine both branches.
combined = Concatenate()([x_agg, x_seq])
x = Dense(64, activation='relu')(combined)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=[input_agg, input_seq], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print("=" * 30, "\n")

# =============================================================================
# STEP 9: Train the Model
# =============================================================================
print("=" * 30)
print("STEP 9: Training the model")
history = model.fit(
    {'aggregated_features': X_agg_train, 'sequence_features': X_seq_train},
    y_train,
    epochs=40,
    batch_size=32,
    validation_data=({'aggregated_features': X_agg_val, 'sequence_features': X_seq_val}, y_val)
)
print("=" * 30, "\n")

# =============================================================================
# STEP 10: Evaluate the Model
# =============================================================================
print("=" * 30)
print("STEP 10: Evaluating the model")
loss, accuracy = model.evaluate(
    {'aggregated_features': X_agg_val, 'sequence_features': X_seq_val},
    y_val
)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))
print("=" * 30, "\n")

# =============================================================================
# STEP 11: Predict and Impute Missing Channels for Incomplete Journeys
# =============================================================================
print("=" * 30)
print("STEP 11: Imputing missing channels for journeys with missing values")
imputed_journeys_count = 0
for user, events in user_journeys.items():
    # Process only journeys with any missing original channel.
    if not any(is_missing(e.get('original_channel')) for e in events):
        continue
    # Use middle events for prediction.
    if len(events) > 2:
        journey_middle = events[1:-1]
    else:
        journey_middle = events
    if not journey_middle:
        continue
    agg_counter = Counter()
    seq_list = []
    for e in journey_middle:
        features = extract_event_features(e)
        agg_counter.update(features)
        seq_list.append(features)
    # Prepare aggregated feature vector.
    agg_features = vec.transform([dict(agg_counter)])
    # Prepare sequence feature vector (pad/truncate).
    seq_vec = vec.transform(seq_list)
    if seq_vec.shape[0] < max_seq_length:
        pad = np.zeros((max_seq_length - seq_vec.shape[0], seq_vec.shape[1]))
        seq_vec = np.vstack([seq_vec, pad])
    else:
        seq_vec = seq_vec[:max_seq_length, :]
    seq_features = np.expand_dims(seq_vec, axis=0)  # Shape: (1, max_seq_length, num_features)
    
    # Predict the dominant channel.
    predicted_proba = model.predict({'aggregated_features': agg_features, 'sequence_features': seq_features})
    predicted_index = np.argmax(predicted_proba, axis=1)[0]
    predicted_channel = le.inverse_transform([predicted_index])[0]
    
    # Update events with missing original_channel.
    journey_updated = False
    for e in events:
        if is_missing(e.get('original_channel')):
            e['final_channel'] = predicted_channel
            journey_updated = True
    if journey_updated:
        imputed_journeys_count += 1

print("Number of journeys with missing channels updated:", imputed_journeys_count)
print("=" * 30, "\n")

# =============================================================================
# STEP 12: Prepare Event-Level Data for Export
# =============================================================================
print("=" * 30)
print("STEP 12: Preparing event-level data for export")
imputed_events = []
for user, events in user_journeys.items():
    for e in events:
        imputed_events.append(e)
imputed_df = pd.DataFrame(imputed_events)
# Retain only the key columns.
imputed_df = imputed_df[['user_pseudo_id', 'event_timestamp', 'event_name', 'original_channel', 'final_channel']]
unique_id = uuid.uuid4().hex[:8]
filename = f"imputed_events_{unique_id}.csv"
imputed_df.to_csv(filename, index=False)
print(f"Imputed event-level data saved to '{filename}'")
print("=" * 30, "\n")
