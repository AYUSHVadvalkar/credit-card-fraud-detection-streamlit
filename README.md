# Credit Card Fraud Detection

A Streamlit app that trains a compact multilayer perceptron on a labeled transaction CSV, reports held-out evaluation metrics, and scores additional transactions.

## Run locally

```bash
python -m venv .venv
```

Activate the environment, then install dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Use the app

1. Upload a labeled CSV containing an `is_fraud` column (`0` for legitimate transactions and `1` for fraud).
2. Choose the training row limit and click **Train fraud detection model**.
3. Review held-out metrics and choose a fraud probability threshold.
4. Upload an unlabeled transactions CSV and download the scored results.

The app follows the notebook by excluding direct identifiers and one-hot encoding categorical fields. It extracts calendar features from `trans_date_trans_time`. The provided `fraudTest.csv` is about 150 MB and is intentionally excluded from Git; upload it in the app, or use a smaller sample for faster training.

## Deploy to Streamlit Community Cloud

Push this repository to GitHub, then create an app in [Streamlit Community Cloud](https://share.streamlit.io/) using `app.py` as the main file. Community Cloud installs the packages listed in `requirements.txt`.
