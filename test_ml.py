import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def _load_data():
    """
    Load census.csv from the starter repo data folder.
    """
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_path, "data", "census.csv")
    return pd.read_csv(data_path)


def test_process_data_outputs_shapes_and_types():
    """
    process_data should return numpy arrays for X/y and preserve row counts.
    """
    data = _load_data().sample(n=200, random_state=42)

    X, y, encoder, lb = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == data.shape[0]
    assert y.shape[0] == data.shape[0]
    assert encoder is not None
    assert lb is not None


def test_train_model_returns_random_forest():
    """
    train_model should return a fitted RandomForestClassifier.
    """
    data = _load_data().sample(n=500, random_state=42)
    train_df, _ = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data[LABEL]
    )

    X_train, y_train, _, _ = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_inference_and_metrics_are_valid():
    """
    inference should return predictions of correct length and metrics should be in [0, 1].
    """
    data = _load_data().sample(n=400, random_state=42)
    train_df, test_df = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data[LABEL]
    )

    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]

    p, r, f1 = compute_model_metrics(y_test, preds)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f1 <= 1.0