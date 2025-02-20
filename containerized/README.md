# ML Channel Attribution on Google Cloud Run

This repository contains two Python scripts:

- **train_model.py**: Trains a dual-input deep learning model using historical event data from BigQuery. It saves the trained model along with a vectorizer and label encoder.
- **impute_channels.py**: Uses a pre-trained model to impute missing channel values on new event data from BigQuery.

Both scripts are packaged into a Docker container for deployment on Cloud Run. You can run the training job (e.g., every 6 months or on demand) and the inference job (e.g., daily or weekly) by overriding the default command when deploying.

## Files

- `train_model.py` – Training pipeline script.
- `impute_channels.py` – Inference (channel imputation) pipeline script.
- `requirements.txt` – Python dependencies.
- `Dockerfile` – Container build instructions.
- `README.md` – This file.

## Setup Instructions

### 1. Prepare Your Service Account Key

- Create a Google Cloud service account with access to BigQuery.
- Download the JSON key file.
- **(Optional but recommended)**: Store this key in Google Secret Manager and configure your Cloud Run services to mount the secret as `/app/service-account-key.json`.
- Alternatively, include the key in your repository (ensure it is secured and not publicly exposed).

### 2. Build and Push the Container Image

Replace `<PROJECT_ID>` and `<IMAGE_NAME>` with your project and image names.

```bash
# Build the container image.
docker build -t gcr.io/<PROJECT_ID>/<IMAGE_NAME>:latest .

# Push the image to Google Container Registry.
docker push gcr.io/<PROJECT_ID>/<IMAGE_NAME>:latest

```

### 3. Deploy to Cloud Run

```bash
gcloud run deploy model-training \
  --image gcr.io/<PROJECT_ID>/<IMAGE_NAME>:latest \
  --platform managed \
  --region <YOUR_REGION> \
  --allow-unauthenticated \
  --set-env-vars SERVICE_ACCOUNT_FILE=/app/service-account-key.json,BQ_DATE_RANGE_START=20241001,BQ_DATE_RANGE_END=20250228,MODEL_CHECKPOINT_PATH=models/model_<ID>.h5,VECTORIZER_PATH=models/vectorizer_<ID>.pkl,LABEL_ENCODER_PATH=models/label_encoder_<ID>.pkl


gcloud run deploy channel-imputation \
  --image gcr.io/<PROJECT_ID>/<IMAGE_NAME>:latest \
  --platform managed \
  --region <YOUR_REGION> \
  --allow-unauthenticated \
  --set-env-vars SERVICE_ACCOUNT_FILE=/app/service-account-key.json,BQ_DATE_RANGE_START=20250301,BQ_DATE_RANGE_END=20250310,MODEL_CHECKPOINT_PATH=models/model_<ID>.h5,VECTORIZER_PATH=models/vectorizer_<ID>.pkl,LABEL_ENCODER_PATH=models/label_encoder_<ID>.pkl \
  --command "python" --args "impute_channels.py"
```

