# Combatting the impact of unconsented data using machine learning

## Introduction

Digital campaigns are expensive, and accurate measurement of user interactions is key to understanding which channels drive performance. However, with growing privacy settings and users opting out of cookie tracking, much of your data can lack channel information. This gap can distort your attribution models, potentially leading you to underestimate some channels and over-invest in others.

While solutions like Google’s [Consent Mode](https://support.google.com/google-ads/answer/10000067?hl=en-GB) provide a high-level statistical approach to filling these gaps, they often apply a one-size-fits-all strategy that may not capture the nuances of your specific data. There is a need for a more tailored solution, one that constructs a detailed picture of each user journey by linking all available event data (including campaign parameters, device information, geotargeting, and other user signals). 

Our proprietary model is designed to address this challenge. It learns from complete user journeys (where the channel is known) and uses that knowledge to predict the most likely channel for journeys with missing data. Through cross-validation, the model ensures that its predictions are both explainable and reliable, providing you with transparent insights that empower smarter, data-driven decisions and optimise your marketing investments.


## Proprietary Model for Channel Imputation

`prop_model_app.py`

Provides an end-to-end pipeline that uses event-level data from Google BigQuery to train a dual-input deep learning model for channel imputation. The model learns from both aggregated journey features and the sequence of events in each user journey to predict the dominant channel (e.g., Direct, Email, Organic Search, etc.).

### Overview

**Data Ingestion:**  
The process begins by authenticating with Google Cloud using a service account key and querying event-level data from BigQuery.

**Data Preprocessing:**  
First, conversion events (such as form submissions) are identified and marked as conversions. Next, user journeys are built by grouping events by user and sorting them by timestamp. The training data is then prepared by extracting aggregated features (by summing feature counts from the “middle” events of each journey) and sequential features (by preserving the order of events, which are padded or truncated to a fixed length). The dominant channel from each journey is used as the training label.

**Model Building:**  
The model uses a dual-input design to process two types of information from user journeys. One branch takes a summary vector of aggregated features, such as counts of events, and processes it with [dense layers](https://datascientest.com/en/dense-neural-networks-understanding-their-structure-and-function) to capture the overall characteristics of the journey. The other branch uses an [LSTM layer](https://en.wikipedia.org/wiki/Long_short-term_memory) to analyse the sequence of individual events, learning the order and timing of user interactions. These two streams are then merged and passed through additional dense layers, with a softmax output producing probabilities for each channel class. This combined approach leverages both the big picture and the fine details, resulting in more accurate channel predictions.

This proprietary model offers a lot of deep 'understanding' based accuracy. Instead of solely relying on aggregated features, which condense a user's journey into a single summary vector, the model uses a dual-input architecture that processes two different types of data simultaneously. One branch of the model processes the aggregated journey features, capturing broad trends and overall patterns (such as the total counts of events and bucketed numerical values). The other branch uses an LSTM layer to analyse the sequence of individual events, capturing the order and timing of interactions, which provides detailed temporal context. By combining these two streams of information, the model is able to learn complex, non-linear relationships within the data, leading to higher predictive accuracy when imputing missing channel values.

**Training and Evaluation:**  
The model is trained on complete journeys, those with all the necessary channel information. This is so that it learns from reliable, known data. During training, the dataset is split into training and validation sets, ensuring that the model’s performance is continuously evaluated on unseen data. The training process involves optimising the model's weights using the [Adam optimiser](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) and minimising the [categorical cross-entropy loss](https://www.v7labs.com/blog/cross-entropy-loss-guide) over 40 [epochs with a batch](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/) size of 32. Throughout this process, key metrics such as accuracy are monitored, and the high validation accuracy indicates that the model is effectively learning the underlying patterns and relationships within the data. This rigorous evaluation provides confidence that the model will generalise well when applied to new, incomplete journeys.

**Imputation:**  
Once the model is trained, it is used to predict missing channel values in journeys where this information is absent. For each incomplete journey, the same feature extraction and vectorisation process used during training is applied. The model then processes these inputs to predict the dominant channel, effectively filling in the gaps in the data. The script automatically updates the journey records with these imputed channel values, and the final, enriched dataset is exported as a CSV file. This imputation process helps to reconstruct a more complete picture of user journeys, which in turn enables more accurate attribution analysis and better-informed decision-making in your digital campaigns.

### Hyperparameters

- MAX_SEQ_LENGTH (20):
This defines the maximum number of events (the sequence length) considered in each user journey. Sequences shorter than this are padded, and longer sequences are truncated. Adjusting this value changes how much sequential information is fed into the model. This is done so that all sequences have a consistent shape before being fed into the model.

- EPOCHS (40):
The number of complete passes through the training data during model training. Increasing this number can allow the model more opportunities to learn, but may risk overfitting, whereas reducing it may lead to underfitting. *(Testing resulted in 40 as best match)*

- BATCH_SIZE (32):
The number of samples processed before the model's weights are updated. A larger batch size might lead to more stable gradient updates but requires more memory, while a smaller batch size can result in noisier updates and potentially slower convergence. *(32 is a default value but works well)*

- LEARNING_RATE (0.001):
This is the step size used by the optimiser (Adam in this case) when updating the model’s weights. A higher learning rate can accelerate training but may overshoot optimal values, while a lower rate might yield a more precise convergence at the cost of slower training. *(0.001 is a common default for Adam optimiser)*

- RANDOM_STATE (42):
A seed value used for random number generation to ensure reproducibility of the model training and data splitting. Changing this value will result in different random splits and model initialisations, which could affect performance outcomes.


## Local Setup and Running

Follow these simple steps to run the scripts on your local machine:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/wgw0/bq_modelling.git
   cd bq_modelling
   ```

2. **Create a Virtual Environment:**
   Create a virtual environment using Python (ensure you have Python **3.10.x** installed):
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
