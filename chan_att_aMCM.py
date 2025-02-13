import os
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import numpy as np

# ---------------------------
# Set up BigQuery client using the service account key file.
# ---------------------------
credentials = service_account.Credentials.from_service_account_file("service-account-key.json")
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# ---------------------------
# Step 1. Query event-level data from BigQuery.
# Adjust the table reference below to your actual table.
# ---------------------------
query = """
SELECT
  user_pseudo_id,
  event_timestamp,
  IFNULL(traffic_source.medium, 'direct') AS medium,
  event_name,
  IFNULL(ecommerce.purchase_revenue_in_usd, 0) AS revenue
FROM `your_project.your_dataset.your_table`
WHERE event_timestamp IS NOT NULL
ORDER BY user_pseudo_id, event_timestamp
"""
# Replace `your_project.your_dataset.your_table` with your actual table reference.

print("Querying data from BigQuery...")
df = client.query(query).to_dataframe()
print("Data retrieved: {} rows.".format(len(df)))

# ---------------------------
# Step 2. Mark conversion events.
# Here we define a conversion as a purchase event with revenue > 0.
# ---------------------------
df['is_conversion'] = np.where((df['event_name'] == 'purchase') & (df['revenue'] > 0), 1, 0)

# ---------------------------
# Step 3. Build user journeys.
# For each user, we create a journey list that begins with 'Start' and ends with 'Conversion' if any conversion occurred,
# otherwise with 'Null'. In between we list the channels (from the "medium" field).
# ---------------------------
journeys = []
for user, group in df.groupby('user_pseudo_id'):
    group = group.sort_values('event_timestamp')
    channels = group['medium'].tolist()
    conversion = 1 if group['is_conversion'].max() == 1 else 0
    if conversion == 1:
        journey = ['Start'] + channels + ['Conversion']
    else:
        journey = ['Start'] + channels + ['Null']
    journeys.append(journey)

print("Constructed journeys for {} users.".format(len(journeys)))

# ---------------------------
# Step 4. Compute transition counts from journeys.
# For each adjacent pair of states in a journey, count the transitions.
# ---------------------------
transition_counts = {}
for journey in journeys:
    for i in range(len(journey) - 1):
        transition = (journey[i], journey[i + 1])
        transition_counts[transition] = transition_counts.get(transition, 0) + 1

# Build a sorted list of all states.
states = sorted({s for transition in transition_counts.keys() for s in transition})
print("States identified:", states)

# Create a mapping from state to index.
state_to_idx = {state: i for i, state in enumerate(states)}
n_states = len(states)

# ---------------------------
# Step 5. Build the transition count matrix and then convert it into probabilities.
# ---------------------------
counts_matrix = np.zeros((n_states, n_states))
for (from_state, to_state), count in transition_counts.items():
    i = state_to_idx[from_state]
    j = state_to_idx[to_state]
    counts_matrix[i, j] = count

# Create the transition probability matrix P.
P = np.zeros_like(counts_matrix)
for i in range(n_states):
    row_sum = counts_matrix[i].sum()
    if row_sum > 0:
        P[i, :] = counts_matrix[i] / row_sum

print("Transition probability matrix (P):")
print(P)

# ---------------------------
# Step 6. Identify absorbing and transient states.
# We assume that only 'Conversion' and 'Null' are absorbing.
# ---------------------------
absorbing_order = ['Conversion', 'Null']
transient_states = [s for s in states if s not in absorbing_order]

print("Transient states:", transient_states)
print("Absorbing states:", absorbing_order)

# Get indices for transient and absorbing states.
transient_indices = [state_to_idx[s] for s in transient_states]
absorbing_indices = [state_to_idx[s] for s in absorbing_order]

# ---------------------------
# Step 7. Extract Q and R submatrices.
# Q: transitions among transient states.
# R: transitions from transient states to absorbing states.
# ---------------------------
Q = P[np.ix_(transient_indices, transient_indices)]
R = P[np.ix_(transient_indices, absorbing_indices)]

print("Matrix Q (transient to transient):")
print(Q)
print("Matrix R (transient to absorbing):")
print(R)

# ---------------------------
# Step 8. Compute the fundamental matrix N = (I - Q)^(-1).
# ---------------------------
I = np.eye(Q.shape[0])
try:
    N = np.linalg.inv(I - Q)
except np.linalg.LinAlgError:
    raise Exception("Matrix (I-Q) is singular and cannot be inverted.")
print("Fundamental matrix N:")
print(N)

# ---------------------------
# Step 9. Compute the absorption probabilities: B = N * R.
# ---------------------------
B = N.dot(R)
print("Absorption probability matrix B:")
print(B)

# ---------------------------
# Step 10. Interpret the results.
# For each transient channel (state), we extract the probability of eventually being absorbed in 'Conversion'.
# ---------------------------
# Find the index of 'Conversion' in the absorbing_order list.
conv_idx = absorbing_order.index('Conversion')

results = []
for i, state in enumerate(transient_states):
    conversion_prob = B[i, conv_idx]
    results.append({
        'Channel': state,
        'Conversion_Probability': conversion_prob
    })

results_df = pd.DataFrame(results)
print("\nAttribution Results (Markov Chain Absorption Probabilities):")
print(results_df)

# Optionally, you can write the results back to BigQuery or to a CSV file.
# For example, to save to CSV:
results_df.to_csv("attribution_results.csv", index=False)
print("\nAttribution results saved to attribution_results.csv")
