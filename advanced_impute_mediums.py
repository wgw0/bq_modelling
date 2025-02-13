import os
import uuid
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

# ------------------------------------------------------------------------------
# Helper function to check if a channel value is missing.
# It returns True if the channel is None, empty, or one of the specified missing values.
# ------------------------------------------------------------------------------
def is_missing(channel):
    if channel is None:
        return True
    channel_str = str(channel).strip().lower()
    return channel_str in ["(none)", "", "unknown", "undefined", "null", "Unassigned"]

# ------------------------------------------------------------------------------
# Advanced Channel Imputation through Python Modelling
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# STEP 0: Set up BigQuery client using the service account key file.
# ------------------------------------------------------------------------------
print("=" * 60)
print("STEP 0: Setting up BigQuery client using service-account-key.json")
credentials = service_account.Credentials.from_service_account_file("service-account-key.json")
client = bigquery.Client(credentials=credentials, project=credentials.project_id)
print("Client successfully created for project:", credentials.project_id)
print("=" * 60, "\n")

# ------------------------------------------------------------------------------
# STEP 1: Query event-level data from BigQuery.
# ------------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Querying event-level data from BigQuery")
# Select the primary channel group as "channel"
query = """
SELECT
  user_pseudo_id,
  event_timestamp,
  IFNULL(session_traffic_source_last_click.cross_channel_campaign.primary_channel_group, 'direct') AS channel,
  event_name
FROM `gtm-5vcmpn9-ntzmm.analytics_345697125.events_20250212`
WHERE event_timestamp IS NOT NULL
ORDER BY user_pseudo_id, event_timestamp
"""
print("Running query...")
df = client.query(query).to_dataframe()
print("Data retrieved: {} rows.".format(len(df)))
print("=" * 60, "\n")

# ------------------------------------------------------------------------------
# STEP 2: Mark conversion events (using key event names).
# ------------------------------------------------------------------------------
print("=" * 60)
print("STEP 2: Marking conversion events")
# Instead of using revenue, define conversion events based on key event names.
# Adjust this list as needed for your GA4 configuration.
conversion_events = ['purchase', 'conversion']  # Example conversion events
df['is_conversion'] = np.where(df['event_name'].isin(conversion_events), 1, 0)
num_conversions = df['is_conversion'].sum()
print(f"Total conversion events flagged: {num_conversions}")
print("=" * 60, "\n")

# ------------------------------------------------------------------------------
# STEP 3: Build event-level journeys.
# ------------------------------------------------------------------------------
print("=" * 60)
print("STEP 3: Building event-level journeys for each user")
# For each user, sort events by timestamp.
# We'll create a journey for each user and add special markers:
# "Start" at the beginning and "Conversion" if any event is a conversion; otherwise "Null".
# Each event record will include user_pseudo_id, event_timestamp, original_channel, event_name, and final_channel (initially same as original).
journey_events = []
for user, group in df.groupby('user_pseudo_id'):
    group = group.sort_values('event_timestamp')
    channels = group['channel'].tolist()
    # Mark conversion if any event in the group is flagged.
    conversion = 1 if group['is_conversion'].max() == 1 else 0
    # Build a journey: add "Start" at beginning, then all channel values, then outcome.
    if conversion == 1:
        journey = ['Start'] + channels + ['Conversion']
    else:
        journey = ['Start'] + channels + ['Null']
    # Also keep the original event-level records for later join.
    for idx, row in group.iterrows():
        journey_events.append({
            'user_pseudo_id': row['user_pseudo_id'],
            'event_timestamp': row['event_timestamp'],
            'original_channel': row['channel'],
            'event_name': row['event_name'],
            'final_channel': row['channel']  # initially same as original
        })
# Build per-user journeys (as lists of event records) for imputation.
user_journeys = defaultdict(list)
for event in journey_events:
    user_journeys[event['user_pseudo_id']].append(event)

print("Constructed journeys for {} users.".format(len(user_journeys)))
print("Example journey for one user:")
example_user = next(iter(user_journeys))
for e in sorted(user_journeys[example_user], key=lambda x: x['event_timestamp']):
    print("  ", e)
print("=" * 60, "\n")

# ------------------------------------------------------------------------------
# STEP 4: Build a journey-level classifier from complete journeys.
# ------------------------------------------------------------------------------
print("=" * 60)
print("STEP 4: Building journey-level classifier for channel imputation")
# We'll build training data from journeys that have complete channel info (i.e. no missing channels).
# We assume missing channels are indicated by is_missing(original_channel)==True.
X_train_dicts = []
y_train = []
for user, events in user_journeys.items():
    # Check if any event has missing channel.
    if any(is_missing(e['original_channel']) for e in events):
        continue  # Skip journeys with missing values for training
    # We use events in the "middle" of the journey (exclude first and last, which are artificial markers)
    journey_middle = events[1:-1]
    if not journey_middle:
        continue
    # Build a feature dictionary: count the event names
    feat = Counter(e['event_name'] for e in journey_middle)
    # The target is the dominant channel among these events.
    channels = [e['original_channel'] for e in journey_middle]
    # Compute the mode (most frequent channel) for the journey.
    dominant_channel = Counter(channels).most_common(1)[0][0]
    X_train_dicts.append(feat)
    y_train.append(dominant_channel)

print("Number of complete journeys for training:", len(X_train_dicts))

# Use DictVectorizer to convert feature dicts to a feature matrix.
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train_dicts)

# Train a Multinomial Naive Bayes classifier.
clf = MultinomialNB()
clf.fit(X_train, y_train)
print("Classifier trained.")
print("=" * 60, "\n")

# ------------------------------------------------------------------------------
# STEP 5: Predict and impute missing channels for incomplete journeys.
# ------------------------------------------------------------------------------
print("=" * 60)
print("STEP 5: Predicting dominant channel for journeys with missing values and imputing them")
# For each journey with missing channels, predict the dominant channel using the classifier.
for user, events in user_journeys.items():
    # Check if the journey has any missing channel.
    if not any(is_missing(e['original_channel']) for e in events):
        continue
    journey_middle = events[1:-1]
    if not journey_middle:
        continue
    feat = Counter(e['event_name'] for e in journey_middle)
    X_test = vec.transform([feat])
    predicted_channel = clf.predict(X_test)[0]
    # Impute every event in the journey with missing channel.
    for e in events:
        if is_missing(e['original_channel']):
            e['final_channel'] = predicted_channel

print("Imputation complete. Example imputed events:")
imputed_sample = []
for user, events in user_journeys.items():
    for e in sorted(events, key=lambda x: x['event_timestamp']):
        if is_missing(e['original_channel']):
            imputed_sample.append(e)
    if len(imputed_sample) >= 10:
        break
for e in imputed_sample[:10]:
    print(f"User: {e['user_pseudo_id']}, Timestamp: {e['event_timestamp']}, Original: {e['original_channel']}, Final: {e['final_channel']}")
print("=" * 60, "\n")

# ------------------------------------------------------------------------------
# STEP 6: Prepare event-level data for export.
# ------------------------------------------------------------------------------
print("=" * 60)
print("STEP 6: Saving imputed event-level data to CSV for joining back to BigQuery")
# Create a DataFrame with key fields.
imputed_events = []
for user, events in user_journeys.items():
    for e in events:
        imputed_events.append(e)
imputed_df = pd.DataFrame(imputed_events)
# Retain key fields: user_pseudo_id, event_timestamp, event_name, original_channel, final_channel.
imputed_df = imputed_df[['user_pseudo_id', 'event_timestamp', 'event_name', 'original_channel', 'final_channel']]
# Generate a unique filename with an 8-character hex identifier.
unique_id = uuid.uuid4().hex[:8]
filename = f"imputed_events_{unique_id}.csv"
imputed_df.to_csv(filename, index=False)
print(f"Imputed event-level data saved to '{filename}'")
print("=" * 60, "\n")
