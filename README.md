# Combatting the impact of unconsented data using machine learning

## Introduction

Digital campaigns are expensive, and accurate measurement of user interactions is key to understanding which channels drive performance. However, with growing privacy settings and users opting out of cookie tracking, much of your data lacks channel information. This gap can distort your attribution models, potentially leading you to underestimate some channels and over-invest in others.

While solutions like Google’s Consent Mode provide a high-level statistical approach to filling these gaps, they often apply a one-size-fits-all strategy that may not capture the nuances of your specific data. There is a need for a more tailored solution, one that constructs a detailed picture of each user journey by linking all available event data (including campaign parameters, device information, geotargeting, and other user signals). 

Our proprietary model is designed to address this challenge. It learns from complete user journeys (where the channel is known) and uses that knowledge to predict the most likely channel for journeys with missing data. Through cross-validation, the model ensures that its predictions are both explainable and reliable, providing you with transparent insights that empower smarter, data-driven decisions and optimise your marketing investments.


## Proprietary Model for Channel Imputation

Provides an end-to-end pipeline that uses event-level data from Google BigQuery to train a dual-input deep learning model for channel imputation. The model learns from both aggregated journey features and the sequence of events in each user journey to predict the dominant channel (e.g., Direct, Email, Organic Search, etc.).

### Overview

The pipeline consists of the following steps:

**Data Ingestion:**  
The process begins by authenticating with Google Cloud using a service account key and querying event-level data from BigQuery.

**Data Preprocessing:**  
First, conversion events (such as form submissions) are identified and marked as conversions. Next, user journeys are built by grouping events by user and sorting them by timestamp. The training data is then prepared by extracting aggregated features (by summing feature counts from the “middle” events of each journey) and sequential features (by preserving the order of events, which are padded or truncated to a fixed length). The dominant channel from each journey is used as the training label.

**Model Building:**  
A dual-input deep learning model is constructed with two branches. One branch processes a vector of aggregated features through dense layers (the Aggregated Features Branch), and the other processes the sequence of event features using an LSTM layer (the Sequence Branch). The outputs of both branches are concatenated and fed through additional dense layers to produce a softmax output over the channel classes.

**Training and Evaluation:**  
The model is trained on complete journeys and validated on a separate validation set.

**Imputation:**  
For journeys with missing channel values, the model predicts the dominant channel and the script updates these journeys with the imputed channel values.



## Journey-Level Channel Imputation Script

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


## Local Setup and Running

Follow these simple steps to run the scripts on your local machine:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/wgw0/bq_modelling.git
   cd bq_modelling
   ```

2. **Create a Virtual Environment:**
   Create a virtual environment using Python (ensure you have Python 3.10.x installed):
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment:**
   - **Windows:**
     ```bash
     .\.venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies:**
   Install all required packages with:
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Your Google Cloud Credentials:**
   - Place your `service-account-key.json` file in the repository’s root directory.
   - Ensure the scripts (e.g., in `prop_model_app.py`) reference the correct path to this key.

6. **Run the Scripts:**
   - To run the advanced proprietary model for channel imputation:
     ```bash
     python prop_model_app.py
     ```
   - To run the baseline imputation script:
     ```bash
     python advanced_impute_mediums.py
     ```

