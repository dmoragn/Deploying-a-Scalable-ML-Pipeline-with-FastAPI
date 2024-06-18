import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, load_model

# Sample data for testing
data = {
    "age": [37, 50],
    "workclass": ["Private", "Self-emp-not-inc"],
    "fnlgt": [178356, 83311],
    "education": ["HS-grad", "Bachelors"],
    "education-num": [10, 13],
    "marital-status": ["Married-civ-spouse", "Married-civ-spouse"],
    "occupation": ["Prof-specialty", "Exec-managerial"],
    "relationship": ["Husband", "Husband"],
    "race": ["White", "White"],
    "sex": ["Male", "Male"],
    "capital-gain": [0, 0],
    "capital-loss": [0, 0],
    "hours-per-week": [40, 13],
    "native-country": ["United-States", "United-States"],
    "salary": [">50K", "<=50K"]
}

@pytest.fixture
def processed_data():
    df = pd.DataFrame(data)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    return X, y, encoder, lb

def test_train_model(processed_data):
    """
    Test if the train_model function trains a RandomForestClassifier and returns the expected type of result.
    """
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"
    assert hasattr(model, 'predict'), "Model does not have a predict method"

def test_inference(processed_data):
    """
    Test if the inference function returns a numpy array of predictions with the expected length.
    """
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert isinstance(preds, np.ndarray), "Inference does not return a numpy array"
    assert len(preds) == len(X_train), "The length of predictions does not match the length of input data"

def test_compute_model_metrics():
    """
    Test if the compute_model_metrics function returns expected precision, recall, and fbeta values.
    """
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0, "Precision is not as expected"
    assert recall == 0.6666666666666666, "Recall is not as expected"
    assert fbeta == 0.8, "F-beta score is not as expected"

def test_data_processing_size(processed_data):
    """
    Test if the processed data has the expected size and type.
    """
    X, y, _, _ = processed_data
    assert X.shape[0] == 2, "The number of samples in X is not as expected"
    assert X.shape[1] > 0, "The number of features in X is not as expected"
    assert len(y) == 2, "The number of samples in y is not as expected"
    assert isinstance(X, np.ndarray), "Processed data X is not a numpy array"
    assert isinstance(y, np.ndarray), "Processed labels y are not a numpy array"
