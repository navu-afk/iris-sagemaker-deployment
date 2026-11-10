
import os
import json
import joblib
import numpy as np


def model_fn(model_dir):
    """Load the trained model from SageMaker model directory."""
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)


def input_fn(request_body, content_type):
    """Convert incoming JSON request into model input."""
    if content_type != "application/json":
        raise ValueError("Only application/json is supported")

    data = json.loads(request_body)

    if isinstance(data, dict):
        data = data["instances"]

    return np.array(data)


def predict_fn(input_data, model_data):
    """Generate predictions."""
    model = model_data["model"]
    target_names = model_data["target_names"]

    predictions = model.predict(input_data)

    return {
        "predictions": [
            target_names[int(prediction)]
            for prediction in predictions
        ]
    }


def output_fn(prediction, accept):
    """Convert predictions into JSON response."""
    return json.dumps(prediction), "application/json"
