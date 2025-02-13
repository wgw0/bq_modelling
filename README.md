# BigQuery SQL

This repository contains a set of complex SQL queries for interacting with large analytics data sets stored in BigQuery.

## Advanced Channel Imputation Script Logic

advanced_impute_mediums.py

### 1. Mark Which Events Are Conversions
- **What:**  
  The script checks the event name.
- **How:**  
  If the event is something like `"purchase"` or `"conversion"`, it flags that event as a conversion.
- **Why:**  
  This conversion flag tells the script whether a user's journey eventually led to a sale or conversion.

### 2. Create a User’s Journey
- **What:**  
  For each user, the script sorts their events by time.
- **How:**  
  It creates a list (or "journey") that:
  - Starts with the special word `"Start"`.
  - Follows with all the channels the user encountered.
  - Ends with `"Conversion"` if the user converted, or `"Null"` if they did not.
- **Example:**  
  `['Start', 'organic', 'organic', 'Null']` means:
  - The user journey started,
  - Saw the channel `"organic"` twice,
  - And ended without a conversion.

### 3. Train a Model Using Complete Journeys
- **What:**  
  The script identifies "complete" journeys—those that have no missing channel values.
- **How:**  
  - It looks at all the event names (e.g., `"page_view"`, `"click"`, etc.) that occur in the journey.
  - It counts how often each event appears.
  - Then it determines the most common channel in these journeys (for example, `"organic"` might be the most frequent).
- **Why:**  
  This process learns from clean, complete examples to understand which channel usually drives a certain type of journey.

### 4. Fill in the Missing Channels
- **What:**  
  For journeys that have missing channels (labeled as `"(none)"`).
- **How:**  
  - The script examines the events in the incomplete journey.
  - It uses the pattern learned from complete journeys to predict the most likely channel.
  - It then replaces the missing channel (`"(none)"`) with the predicted channel.
- **Why:**  
  This retroactive imputation enriches the dataset, allowing you to update your records with a more complete and accurate view of the user's journey.
