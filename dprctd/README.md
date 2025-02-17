## Journey-Level Channel Imputation Script

`advanced_impute_mediums.py`

### Overview

This script enhances your digital analytics data by filling in missing channel information in user journeys. It connects to BigQuery to retrieve event-level data from your Google Analytics events table, including: campaign parameters, device details, geographic information, and more, and reconstructs each user’s journey. Using these rich feature profiles, it then trains a predictive model on complete journeys to identify the dominant channel. Once trained, the model imputes the missing channel values for incomplete journeys, outputting a CSV file that can be joined back to your BigQuery tables for further analysis.

### Steps

After the data is retrieved, the script flags conversion events (e.g., form submits) and groups the events by user to form individual journeys. Each journey is then enriched with a wide variety of features. For example, numeric fields like event value and revenue are bucketed into ranges so that the model can work with categorical data rather than raw numbers.

Next, the script trains a **Multinomial Naive Bayes classifier** using only the journeys that have complete channel information. The model learns to associate aggregated journey features (e.g., types of events, device information, geo details, and a lot more) with the most common channel observed in the journey.

Finally, for journeys where the channel data is missing, the same feature extraction process is applied. The trained model predicts the most likely channel, and the script updates the journey with the imputed channel value. The resulting data is then saved as a CSV file for further use.

### Model Details

The core of the prediction process is a [**Multinomial Naive Bayes classifier**](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html). This model is suited to tasks where features represent counts or frequencies, which is why it works well with our aggregated, bucketed event data.

Naive Bayes classifiers work on the principle of calculating the probability of each class (or channel, in this case) based on the frequency of features.

The classifier tests itself on data where it already knows the correct answers. It temporarily 'forgets' these answers, makes its predictions, and then checks how often it got them right. This success rate is then used to estimate how accurate its predictions are for the cases where we don't know the actual answers.