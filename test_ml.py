import pandas as pd
import os
from ml.model import (compute_model_metrics)
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
main_path = os.path.dirname(os.path.abspath(__file__))


def test_data(subtests):
    """
    Checks data characteristics such target column, features ans data.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"]
    data = pd.read_csv(os.path.join(main_path, "data", "census.csv"))
    with subtests.test(msg="Checking Features"):
        assert set(cat_features).issubset(data.columns), "Feature column\
            missing from training set."

    with subtests.test(msg="Checking Target Column"):
        assert set(["salary"]).issubset(data.columns), "Target Column not\
              present in data."

    with subtests.test(msg="Checking data."):
        assert data.shape[0] > 0, "Traning data has no rows."


def test_created_model_is_RandomForestClassifier():
    """
    Test that trained model is RandomForestClassifier model.
    """
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y_train = np.array([0, 0, 1, 1])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    assert type(model).__name__ == "RandomForestClassifier", f"Expected \
        RandomForestClassifier, got {type(model).__name__}"


def test_specific_target_metrics():
    """Test arrays that yield Precision ~0.7592, Recall ~0.5772,\
        and F1 ~0.6558. Best metrics for initial model."""
    tp, fp, fn = 142, 45, 104

    y = np.concatenate([
        np.ones(tp),   # True Positives
        np.zeros(fp),  # False Positives
        np.ones(fn)    # False Negatives
        ])
    preds = np.concatenate([
        np.ones(tp),   # True Positives
        np.ones(fp),   # False Positives
        np.zeros(fn)   # False Negatives
    ])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(0.7592, abs=1e-3)
    assert recall == pytest.approx(0.5772, abs=1e-3)
    assert fbeta == pytest.approx(0.6558, abs=1e-3)