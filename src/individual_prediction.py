import joblib
import shap
import pandas as pd

# Load the optimized pipeline
pipeline = joblib.load("models/best_pipeline.pkl")

# Load the explainer object
explainer = joblib.load("models/explainer.pkl")

# Load your data (or use a custom input)
data = pd.read_csv("data.csv")
X = data.drop("target", axis=1)

# Select a single instance for prediction
instance = X.iloc[[0]]  # Replace 0 with the index of the instance you want to explain

# Make a prediction for the instance
prediction = pipeline.predict(instance)

# Compute SHAP values for the instance
shap_values = explainer(instance)

# Print the prediction and SHAP values
print(f"Prediction: {prediction}")
print("SHAP values:")
print(shap_values)

# Generate a SHAP force plot for the instance
shap.force_plot(explainer.expected_value, shap_values.values, instance, feature_names=instance.columns, matplotlib=True)
