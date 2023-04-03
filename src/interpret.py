import os
import joblib
import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the data
data_path = os.path.join(os.path.abspath("data"), "clean_data.csv")
data = pd.read_csv(data_path)
X = data.drop("psm", axis=1)
y = data["psm"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the best pipeline
best_pipeline_path = os.path.join(os.path.abspath("models"), "best_pipeline.pkl")
best_pipeline = joblib.load(best_pipeline_path)

# Fit and transform the data using the pipeline's pre-processing steps
X_train_transformed = best_pipeline.named_steps["mean_encoder"].fit_transform(X_train, y_train)
X_train_transformed = best_pipeline.named_steps["scaler"].transform(X_train_transformed)

# Fit the XGBRegressor model with the transformed data
model = best_pipeline.named_steps["model"]
model.fit(X_train_transformed, y_train)

# Get the transformed column names
transformed_columns = X_train.columns.tolist()

# Use the TreeExplainer with the XGBRegressor model and feature names
explainer = shap.Explainer(model, feature_names=transformed_columns)
shap_values = explainer(X_train_transformed)

# Plot the SHAP summary plot with actual feature names
shap.summary_plot(shap_values, X_train_transformed, feature_names=transformed_columns, show=False)
shap_summary_plot_path = os.path.join(os.path.abspath("models"), "shap_summary_plot.png")

# Save the plot using matplotlib
plt.savefig(shap_summary_plot_path)
plt.close()

# Save the explainer for front-end use
explainer_path = os.path.join(os.path.abspath("models"), "explainer.pkl")
joblib.dump(explainer, explainer_path)
