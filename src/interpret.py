import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split

# Load the optimized pipeline
pipeline = joblib.load("models/best_pipeline.pkl")

# Load your data
data = pd.read_csv("data/clean_data.csv")
X = data.drop("psm", axis=1)
y = data["psm"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get the XGBoost model from the pipeline
model = pipeline.named_steps["model"]

# Create a SHAP explainer
explainer = shap.Explainer(model)

# Compute SHAP values for the test set
shap_values = explainer(X_test)

# Generate a summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
shap.summary_plot(shap_values, X_test, show=False)

# Save the summary plot
shap.plots.savefig("models/shap_summary_plot.png")

# Save the explainer object
joblib.dump(explainer, "models/explainer.pkl")