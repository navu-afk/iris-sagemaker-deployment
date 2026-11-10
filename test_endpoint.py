
import joblib

# Load the saved model
model_data = joblib.load("model.joblib")

model = model_data["model"]
target_names = model_data["target_names"]

# Test Iris sample
test_sample = [[5.1, 3.5, 1.4, 0.2]]

# Generate prediction
prediction = model.predict(test_sample)

# Convert predicted number to flower name
predicted_class = target_names[prediction[0]]

print("Test Sample:", test_sample)
print("Predicted Class:", predicted_class)
print("Prediction Successful!")
