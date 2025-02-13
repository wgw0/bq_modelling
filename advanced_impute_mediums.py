import os
import uuid
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# ------------------------------------------------------------------------------
# Helper function to check if a channel value is missing.
# Returns True if the channel is None, empty, or one of the specified missing values.
# ------------------------------------------------------------------------------
def is_missing(channel):
    if channel is None:
        return True
    channel_str = str(channel).strip().lower()
    return channel_str in ["(none)", "", "unknown", "undefined", "null", "unassigned"]

# ------------------------------------------------------------------------------
# Advanced Channel Imputation through Python Modelling
# ------------------------------------------------------------------------------

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
# ------------------------------------------------------------------------------
print("=" * 30)
print("STEP 1: Querying event-level data from BigQuery")
query = """
SELECT
  user_pseudo_id,
  event_timestamp,
  IFNULL(session_traffic_source_last_click.cross_channel_campaign.primary_channel_group, 'direct') AS channel,
  event_name,
  device.category AS device_category,
  device.operating_system AS device_os,
  geo.country AS geo_country,
  geo.city AS geo_city
FROM `gtm-5vcmpn9-ntzmm.analytics_345697125.events_202412*`
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
conversion_events = ['purchase', 'conversion']  # Example conversion events
df['is_conversion'] = np.where(df['event_name'].isin(conversion_events), 1, 0)
num_conversions = df['is_conversion'].sum()
print(f"Total conversion events flagged: {num_conversions}")
print("=" * 30, "\n")

# ------------------------------------------------------------------------------
# STEP 3: Build event-level journeys.
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
            'event_timestamp': row['event_timestamp'],
            'original_channel': row['channel'],
            'event_name': row['event_name'],
            'final_channel': row['channel'],  # initially same as original
            'device_category': row['device_category'],
            'device_os': row['device_os'],
            'geo_country': row['geo_country'],
            'geo_city': row['geo_city']
        })

user_journeys = defaultdict(list)
for event in journey_events:
    user_journeys[event['user_pseudo_id']].append(event)

print("Constructed journeys for {} users.".format(len(user_journeys)))
print("=" * 30, "\n")

# ------------------------------------------------------------------------------
# STEP 4: Build a journey-level classifier from complete journeys.
# ------------------------------------------------------------------------------
print("=" * 30)
print("STEP 4: Building journey-level classifier for channel imputation")
X_train_dicts = []
y_train = []
for user, events in user_journeys.items():
    if any(is_missing(e['original_channel']) for e in events):
        continue  # Skip journeys with missing values for training
    journey_middle = events[1:-1]  # Exclude the artificial markers
    if not journey_middle:
        continue
    feat = Counter()
    # Aggregate multiple features per event:
    for e in journey_middle:
        feat[f"event:{e['event_name']}"] += 1
        feat[f"device:{e['device_category'] if e['device_category'] else 'unknown'}"] += 1
        feat[f"os:{e['device_os'] if e['device_os'] else 'unknown'}"] += 1
        feat[f"geo_country:{e['geo_country'] if e['geo_country'] else 'unknown'}"] += 1
        feat[f"geo_city:{e['geo_city'] if e['geo_city'] else 'unknown'}"] += 1
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
        feat[f"event:{e['event_name']}"] += 1
        feat[f"device:{e['device_category'] if e['device_category'] else 'unknown'}"] += 1
        feat[f"os:{e['device_os'] if e['device_os'] else 'unknown'}"] += 1
        feat[f"geo_country:{e['geo_country'] if e['geo_country'] else 'unknown'}"] += 1
        feat[f"geo_city:{e['geo_city'] if e['geo_city'] else 'unknown'}"] += 1
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
