import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Ensure the project path is correctly set
project_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Data path: {data_path}")

# Load the census.csv data
data = pd.read_csv(data_path)

# Split the provided data to have a train dataset and a test dataset
train, test = train_test_split(data, test_size=0.2, random_state=42)

# DO NOT MODIFY
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

# Use the process_data function provided to process the training data
X_train, y_train, encoder, lb = process_data(
    train, 
    categorical_features=cat_features, 
    label="salary", 
    training=True
)

# Process the test data using the same encoder and label binarizer
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# Save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
print(f"Model saved to {model_path}")

encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)
print(f"Encoder saved to {encoder_path}")

# Load the model
model = load_model(model_path)
print(f"Loading model from {model_path}")

# Use the inference function to run the model inferences on the test dataset
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute the performance on model slices using the performance_on_categorical_slice function
slice_output_path = os.path.join(project_path, "slice_output.txt")
with open(slice_output_path, "w") as f:
    for col in cat_features:
        # Iterate through the unique values in one categorical feature
        for slice_value in sorted(test[col].unique()):
            count = test[test[col] == slice_value].shape[0]
            p, r, fb = performance_on_categorical_slice(
                test, column_name=col, slice_value=slice_value,
                categorical_features=cat_features, label="salary", 
                encoder=encoder, lb=lb, model=model
            )
            f.write(f"{col}: {slice_value}, Count: {count:,}\n")
            f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n")
            f.write("\n")
print(f"Performance on model slices saved to {slice_output_path}")
