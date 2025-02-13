import os
import uuid
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# ------------------------------------------------------------------------------
# Helper function: Check if a channel value is missing.
# Returns True if the channel is None, empty, or one of the specified missing values.
# ------------------------------------------------------------------------------
def is_missing(channel):
    if channel is None:
        return True
    channel_str = str(channel).strip().lower()
    return channel_str in ["(none)", "", "unknown", "undefined", "null", "unassigned"]

# ------------------------------------------------------------------------------
# Helper function: Bucket a numeric value into a categorical bucket.
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# STEP 0: Set up BigQuery client using the service account key file.
# ------------------------------------------------------------------------------
print("=" * 30)
print("STEP 0: Setting up BigQuery client using service-account-key.json")
credentials = service_account.Credentials.from_service_account_file("service-account-key.json")
client = bigquery.Client(credentials=credentials, project=credentials.project_id)
print("Client successfully created for project:", credentials.project_id)
print("=" * 30, "\n")

# ------------------------------------------------------------------------------
# STEP 1: Query event-level data from BigQuery.
# The query now extracts many additional fields from the table, including selected 
# event_params keys and extra device, geo, app, traffic, ecommerce, and publisher info.
# ------------------------------------------------------------------------------
print("=" * 30)
print("STEP 1: Querying event-level data from BigQuery")
query = """
SELECT
  event_date,
  event_timestamp,
  event_name,
  -- Extract selected keys from event_params (if present)
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
FROM `gtm-5vcmpn9-ntzmm.analytics_345697125.events_202502*`
WHERE event_timestamp IS NOT NULL
ORDER BY user_pseudo_id, event_timestamp
"""
print("Running query...")
df = client.query(query).to_dataframe()
print("Data retrieved: {} rows.".format(len(df)))
print("=" * 30, "\n")

# ------------------------------------------------------------------------------
# STEP 2: Mark conversion events (using key event names).
# ------------------------------------------------------------------------------
print("=" * 30)
print("STEP 2: Marking conversion events")
conversion_events = ['form_submit', 'form_submit_contact']
df['is_conversion'] = np.where(df['event_name'].isin(conversion_events), 1, 0)
num_conversions = df['is_conversion'].sum()
print(f"Total conversion events flagged: {num_conversions}")
print("=" * 30, "\n")

# ------------------------------------------------------------------------------
# STEP 3: Build event-level journeys.
# Each event record now includes additional fields from the query.
# ------------------------------------------------------------------------------
print("=" * 30)
print("STEP 3: Building event-level journeys for each user")
journey_events = []
for user, group in df.groupby('user_pseudo_id'):
    group = group.sort_values('event_timestamp')
    channels = group['channel'].tolist()
    conversion = 1 if group['is_conversion'].max() == 1 else 0
    if conversion == 1:
        journey = ['Start'] + channels + ['Conversion']
    else:
        journey = ['Start'] + channels + ['Null']
    for idx, row in group.iterrows():
        journey_events.append({
            'user_pseudo_id': row['user_pseudo_id'],
            'event_date': row['event_date'],
            'event_timestamp': row['event_timestamp'],
            'original_channel': row['channel'],
            'event_name': row['event_name'],
            'final_channel': row['channel'],  # initially same as original
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

user_journeys = defaultdict(list)
for event in journey_events:
    user_journeys[event['user_pseudo_id']].append(event)

print("Constructed journeys for {} users.".format(len(user_journeys)))
print("=" * 30, "\n")

# ------------------------------------------------------------------------------
# Helper function to aggregate features from an event.
# This function extracts features from a single event dictionary.
# ------------------------------------------------------------------------------
def extract_event_features(e):
    feat = Counter()
    # Basic event information
    feat[f"event:{e['event_name'] if e['event_name'] else 'unknown'}"] += 1
    feat[f"event_date:{e['event_date'] if e['event_date'] else 'unknown'}"] += 1
    # Campaign parameters from event_params
    feat[f"campaign:{e['campaign'] if e['campaign'] else 'unknown'}"] += 1
    feat[f"campaign_medium:{e['campaign_medium'] if e['campaign_medium'] else 'unknown'}"] += 1
    feat[f"campaign_source:{e['campaign_source'] if e['campaign_source'] else 'unknown'}"] += 1
    # Bucket numeric event value
    if e['event_value_in_usd'] is not None:
        bucket = bucket_numeric(e['event_value_in_usd'])
        feat[f"event_value_bucket:{bucket}"] += 1
    else:
        feat["event_value_bucket:unknown"] += 1

    # Device features
    feat[f"device:{e['device_category'] if e['device_category'] else 'unknown'}"] += 1
    feat[f"os:{e['device_os'] if e['device_os'] else 'unknown'}"] += 1
    feat[f"mobile_brand:{e['mobile_brand_name'] if e['mobile_brand_name'] else 'unknown'}"] += 1
    feat[f"mobile_model:{e['mobile_model_name'] if e['mobile_model_name'] else 'unknown'}"] += 1
    feat[f"os_version:{e['operating_system_version'] if e['operating_system_version'] else 'unknown'}"] += 1
    feat[f"language:{e['language'] if e['language'] else 'unknown'}"] += 1
    feat[f"browser:{e['browser'] if e['browser'] else 'unknown'}"] += 1
    feat[f"browser_version:{e['browser_version'] if e['browser_version'] else 'unknown'}"] += 1

    # Geo features
    feat[f"geo_country:{e['geo_country'] if e['geo_country'] else 'unknown'}"] += 1
    feat[f"geo_city:{e['geo_city'] if e['geo_city'] else 'unknown'}"] += 1
    feat[f"geo_continent:{e['geo_continent'] if e['geo_continent'] else 'unknown'}"] += 1
    feat[f"geo_region:{e['geo_region'] if e['geo_region'] else 'unknown'}"] += 1
    feat[f"geo_sub_continent:{e['geo_sub_continent'] if e['geo_sub_continent'] else 'unknown'}"] += 1
    feat[f"geo_metro:{e['geo_metro'] if e['geo_metro'] else 'unknown'}"] += 1

    # App information
    feat[f"app_id:{e['app_id'] if e['app_id'] else 'unknown'}"] += 1
    feat[f"app_version:{e['app_version'] if e['app_version'] else 'unknown'}"] += 1
    feat[f"install_store:{e['install_store'] if e['install_store'] else 'unknown'}"] += 1
    feat[f"firebase_app_id:{e['firebase_app_id'] if e['firebase_app_id'] else 'unknown'}"] += 1
    feat[f"install_source:{e['install_source'] if e['install_source'] else 'unknown'}"] += 1

    # Traffic source information
    feat[f"traffic_source_name:{e['traffic_source_name'] if e['traffic_source_name'] else 'unknown'}"] += 1
    feat[f"traffic_source_medium:{e['traffic_source_medium'] if e['traffic_source_medium'] else 'unknown'}"] += 1
    feat[f"traffic_source_source:{e['traffic_source_source'] if e['traffic_source_source'] else 'unknown'}"] += 1

    # Platform
    feat[f"platform:{e['platform'] if e['platform'] else 'unknown'}"] += 1

    # Ecommerce purchase revenue bucketing
    if e['purchase_revenue_in_usd'] is not None:
        bucket = bucket_numeric(e['purchase_revenue_in_usd'])
        feat[f"purchase_revenue_bucket:{bucket}"] += 1
    else:
        feat["purchase_revenue_bucket:unknown"] += 1

    # Collected traffic source parameters
    feat[f"manual_campaign_id:{e['manual_campaign_id'] if e['manual_campaign_id'] else 'unknown'}"] += 1
    feat[f"manual_campaign_name:{e['manual_campaign_name'] if e['manual_campaign_name'] else 'unknown'}"] += 1
    feat[f"manual_source:{e['manual_source'] if e['manual_source'] else 'unknown'}"] += 1
    feat[f"manual_medium:{e['manual_medium'] if e['manual_medium'] else 'unknown'}"] += 1

    # Active user flag (converted to string)
    feat[f"active_user:{str(e['is_active_user']) if e['is_active_user'] is not None else 'unknown'}"] += 1

    # Publisher information (bucketing ad revenue)
    if e['ad_revenue_in_usd'] is not None:
        bucket = bucket_numeric(e['ad_revenue_in_usd'])
        feat[f"ad_revenue_bucket:{bucket}"] += 1
    else:
        feat["ad_revenue_bucket:unknown"] += 1
    feat[f"ad_format:{e['ad_format'] if e['ad_format'] else 'unknown'}"] += 1
    feat[f"ad_source_name:{e['ad_source_name'] if e['ad_source_name'] else 'unknown'}"] += 1
    feat[f"ad_unit_id:{e['ad_unit_id'] if e['ad_unit_id'] else 'unknown'}"] += 1

    return feat

# ------------------------------------------------------------------------------
# STEP 4: Build a journey-level classifier from complete journeys.
# Here we aggregate features from all “middle” events in each journey.
# ------------------------------------------------------------------------------
print("=" * 30)
print("STEP 4: Building journey-level classifier for channel imputation")
X_train_dicts = []
y_train = []
for user, events in user_journeys.items():
    # Only use journeys with complete channel information for training.
    if any(is_missing(e['original_channel']) for e in events):
        continue
    journey_middle = events[1:-1]  # Exclude 'Start' and the final marker
    if not journey_middle:
        continue
    feat = Counter()
    # Aggregate features across events
    for e in journey_middle:
        feat.update(extract_event_features(e))
    # Use the dominant (most common) channel from the journey middle as the label.
    channels = [e['original_channel'] for e in journey_middle]
    dominant_channel = Counter(channels).most_common(1)[0][0]
    X_train_dicts.append(feat)
    y_train.append(dominant_channel)

print("Number of complete journeys for training:", len(X_train_dicts))
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train_dicts)
clf = MultinomialNB()
clf.fit(X_train, y_train)
print("Classifier trained.")

# ------------------------------------------------------------------------------
# STEP 4.5: Evaluate Classifier Accuracy via Cross-Validation.
# ------------------------------------------------------------------------------
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Classifier cross-validation accuracy: {np.mean(cv_scores)*100:.2f}%")
print("=" * 30, "\n")

# ------------------------------------------------------------------------------
# STEP 5: Predict and Impute Missing Channels for Incomplete Journeys.
# For journeys with missing channel info, we aggregate features and predict the dominant channel.
# ------------------------------------------------------------------------------
print("=" * 30)
print("STEP 5: Predicting dominant channel for journeys with missing values and imputing them")
imputed_journeys_count = 0  # counter for journeys updated
for user, events in user_journeys.items():
    if not any(is_missing(e['original_channel']) for e in events):
        continue
    journey_middle = events[1:-1]
    if not journey_middle:
        continue
    feat = Counter()
    for e in journey_middle:
        feat.update(extract_event_features(e))
    X_test = vec.transform([feat])
    predicted_channel = clf.predict(X_test)[0]
    journey_updated = False
    for e in events:
        if is_missing(e['original_channel']):
            e['final_channel'] = predicted_channel
            journey_updated = True
    if journey_updated:
        imputed_journeys_count += 1

print(f"Number of journeys with missing channels updated: {imputed_journeys_count}")
print("=" * 30, "\n")

# ------------------------------------------------------------------------------
# STEP 6: Prepare Event-level Data for Export.
# ------------------------------------------------------------------------------
print("=" * 30)
print("STEP 6: Saving imputed event-level data to CSV for joining back to BigQuery")
imputed_events = []
for user, events in user_journeys.items():
    for e in events:
        imputed_events.append(e)
imputed_df = pd.DataFrame(imputed_events)
imputed_df = imputed_df[['user_pseudo_id', 'event_timestamp', 'event_name', 'original_channel', 'final_channel']]
unique_id = uuid.uuid4().hex[:8]
filename = f"imputed_events_{unique_id}.csv"
imputed_df.to_csv(filename, index=False)
print(f"Imputed event-level data saved to '{filename}'")
print("=" * 30, "\n")
