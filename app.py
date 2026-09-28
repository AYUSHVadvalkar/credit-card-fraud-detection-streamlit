"""Streamlit interface for training and using the fraud detection ANN."""

from __future__ import annotations

import io
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Card Fraud Detection", page_icon="💳", layout="wide")

TARGET = "is_fraud"
IDENTIFIER_COLUMNS = {
    "unnamed: 0", "cc_num", "merchant", "first", "last", "street", "city",
    "zip", "lat", "long", "dob", "trans_num", "unix_time", "merch_lat",
    "merch_long",
}
DATE_COLUMNS = ("trans_date_trans_time", "transaction_time", "trans_time")


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(name).strip() for name in result.columns]
    return result


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's exclusions and add useful transaction-time features."""
    result = clean_column_names(frame)
    for name in DATE_COLUMNS:
        if name in result.columns:
            timestamp = pd.to_datetime(result.pop(name), errors="coerce")
            result[f"{name}_hour"] = timestamp.dt.hour
            result[f"{name}_day"] = timestamp.dt.day
            result[f"{name}_month"] = timestamp.dt.month
            result[f"{name}_weekday"] = timestamp.dt.dayofweek

    remove = [name for name in result.columns if name.lower() in IDENTIFIER_COLUMNS]
    result = result.drop(columns=remove, errors="ignore")
    if "gender" in result.columns:
        result["gender"] = result["gender"].map({"F": "F", "M": "M"}).fillna(result["gender"])
    return result


def make_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    numeric_steps = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=100)),
        ]
    )
    transform = ColumnTransformer(
        [("numeric", numeric_steps, numeric), ("categorical", categorical_steps, categorical)],
        remainder="drop",
    )
    # A compact multilayer perceptron keeps the notebook's artificial neural network approach.
    return Pipeline(
        [
            ("features", transform),
            (
                "ann",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    batch_size=256,
                    learning_rate_init=0.001,
                    max_iter=30,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=4,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]
    )


def read_csv(upload) -> pd.DataFrame:
    return clean_column_names(pd.read_csv(upload, low_memory=False))


def stratified_sample(features: pd.DataFrame, target: pd.Series, maximum: int) -> tuple[pd.DataFrame, pd.Series]:
    if len(features) <= maximum:
        return features, target
    # Retain every fraud row where practical, and sample the much larger legitimate class.
    fraud_idx = target[target == 1].index
    legitimate_idx = target[target == 0].index
    if len(fraud_idx) >= maximum:
        fraud_idx = target.loc[fraud_idx].sample(n=maximum // 2, random_state=42).index
        legitimate_idx = target.loc[legitimate_idx].sample(n=maximum - len(fraud_idx), random_state=42).index
    else:
        legitimate_idx = target.loc[legitimate_idx].sample(
            n=min(maximum - len(fraud_idx), len(legitimate_idx)), random_state=42
        ).index
    keep = fraud_idx.union(legitimate_idx)
    return features.loc[keep], target.loc[keep]


def balance_training_rows(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Undersample only the training split; keep the test split at its natural prevalence."""
    fraud_idx = target[target == 1].index
    legitimate_idx = target[target == 0].sample(n=len(fraud_idx), random_state=42).index
    keep = fraud_idx.union(legitimate_idx)
    return features.loc[keep], target.loc[keep]


st.title("💳 Credit Card Fraud Detection")
st.write(
    "Train a neural network on a labeled transaction CSV, review its evaluation metrics, "
    "then score another CSV of transactions."
)
st.info(
    "Upload a CSV with an `is_fraud` column (0 = legitimate, 1 = fraud) to train. "
    "The notebook's identifier fields are removed and transaction timestamps are converted "
    "to calendar features. The original 150 MB dataset is excluded from Git; upload it here "
    "or use a smaller sample."
)

with st.sidebar:
    st.header("Training settings")
    max_rows = st.slider("Maximum rows used for training", 5_000, 100_000, 30_000, step=5_000)
    threshold = st.slider("Fraud probability threshold", 0.05, 0.95, 0.50, step=0.05)

train_file = st.file_uploader("Labeled training data (CSV)", type="csv", key="train_csv")
if train_file is not None:
    try:
        raw = read_csv(train_file)
        if TARGET not in raw.columns:
            st.error("The training CSV must contain an `is_fraud` target column.")
            st.stop()
        labels = pd.to_numeric(raw[TARGET], errors="coerce")
        valid = labels.isin([0, 1])
        raw = raw.loc[valid].reset_index(drop=True)
        labels = labels.loc[valid].astype("int8").reset_index(drop=True)
        if labels.nunique() != 2:
            st.error("Training data must include examples of both classes (0 and 1).")
            st.stop()

        features = prepare_features(raw.drop(columns=[TARGET]))
        features, labels = stratified_sample(features, labels, max_rows)
        if st.button("Train fraud detection model", type="primary"):
            x_train, x_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            x_train, y_train = balance_training_rows(x_train, y_train)
            model = make_pipeline(x_train)
            with st.spinner("Training the neural network…"):
                model.fit(x_train, y_train)
                probabilities = model.predict_proba(x_test)[:, 1]
            st.session_state["fraud_model"] = model
            st.session_state["feature_columns"] = features.columns.tolist()
            st.session_state["probabilities"] = probabilities
            st.session_state["actual"] = y_test.to_numpy()
            st.success(f"Model trained using {len(features):,} rows and {len(features.columns)} input features.")
    except Exception as exc:
        st.error(f"Could not read or train from this CSV: {exc}")

if "fraud_model" in st.session_state:
    probabilities = st.session_state["probabilities"]
    actual = st.session_state["actual"]
    predicted = (probabilities >= threshold).astype("int8")
    st.subheader("Held-out evaluation")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{accuracy_score(actual, predicted):.3f}")
    metric_cols[1].metric("ROC AUC", f"{roc_auc_score(actual, probabilities):.3f}")
    metric_cols[2].metric("Average precision", f"{average_precision_score(actual, probabilities):.3f}")
    metric_cols[3].metric("Fraud threshold", f"{threshold:.2f}")
    st.write("Confusion matrix (rows = actual, columns = predicted)")
    st.dataframe(pd.DataFrame(confusion_matrix(actual, predicted), index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
    with st.expander("Classification report"):
        st.code(classification_report(actual, predicted, zero_division=0))

    st.subheader("Score transactions")
    prediction_file = st.file_uploader("Unlabeled transaction CSV", type="csv", key="prediction_csv")
    if prediction_file is not None:
        try:
            prediction_raw = read_csv(prediction_file)
            prediction_features = prepare_features(prediction_raw.drop(columns=[TARGET], errors="ignore"))
            expected = st.session_state["feature_columns"]
            prediction_features = prediction_features.reindex(columns=expected)
            prediction_probabilities = st.session_state["fraud_model"].predict_proba(prediction_features)[:, 1]
            results = prediction_raw.copy()
            results["fraud_probability"] = prediction_probabilities
            results["fraud_prediction"] = (prediction_probabilities >= threshold).astype("int8")
            st.dataframe(results.head(100), use_container_width=True)
            output = io.StringIO()
            results.to_csv(output, index=False)
            st.download_button(
                "Download scored transactions",
                data=output.getvalue().encode("utf-8"),
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not score this CSV: {exc}")

st.caption("This model is for educational analysis. Review false positives and false negatives before acting on predictions.")
