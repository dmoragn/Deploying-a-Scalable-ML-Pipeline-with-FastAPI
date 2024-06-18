import pytest
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, save_model, load_model

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
def sample_data():
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def processed_data(sample_data):
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
    X, y, encoder, lb = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)
    return X, y, encoder, lb

def test_process_data(sample_data):
    """
    Test if the process_data function returns the expected output types.
    """
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
    X, y, encoder, lb = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)
    assert isinstance(X, np.ndarray), "Processed data X is not a numpy array"
    assert isinstance(y, np.ndarray), "Processed labels y are not a numpy array"
    assert X.shape[0] == sample_data.shape[0], "Number of samples in X does not match input data"
    assert y.shape[0] == sample_data.shape[0], "Number of samples in y does not match input data"

def test_train_model_no_exception(processed_data):
    """
    Test if the model training raises no exceptions and returns a model.
    """
    X_train, y_train, _, _ = processed_data
    try:
        model = train_model(X_train, y_train)
    except Exception as e:
        pytest.fail(f"Model training raised an exception: {e}")
    assert isinstance(model, RandomForestClassifier), "Trained model is not a RandomForestClassifier"

def test_inference_no_exception(processed_data):
    """
    Test if the inference function raises no exceptions and returns predictions.
    """
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    try:
        preds = inference(model, X_train)
    except Exception as e:
        pytest.fail(f"Inference raised an exception: {e}")
    assert isinstance(preds, np.ndarray), "Inference does not return a numpy array"
    assert len(preds) == len(X_train), "Number of predictions does not match number of samples in X_train"

def test_model_serialization(processed_data):
    """
    Test if the model can be saved and loaded correctly.
    """
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    model_path = "test_model.pkl"
    save_model(model, model_path)
    loaded_model = load_model(model_path)
    assert isinstance(loaded_model, RandomForestClassifier), "Loaded model is not a RandomForestClassifier"
    assert hasattr(loaded_model, 'predict'), "Loaded model does not have a predict method"
    os.remove(model_path)  # Clean up the saved model file

