import subprocess

import joblib
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from feature_engine.encoding import MeanEncoder
from sklearn.metrics import mean_absolute_error, make_scorer
import os


# Custom scoring function
def mean_absolute_percentage_error(y_true, y_pred):
    return (100 * mean_absolute_error(y_true, y_pred)) / y_true.mean()


# Create a custom scorer for scikit-learn
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

# Load your data
data_path = os.path.join(os.path.abspath("data"), "clean_data.csv")
data = pd.read_csv(data_path)
X = data.drop("psm", axis=1)
y = data["psm"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the categorical columns
categorical_columns = ["floor_range", "type_of_sale", "district_name"]

# Define the pipeline
pipeline = Pipeline([
    ("mean_encoder", MeanEncoder(variables=categorical_columns)),
    ("scaler", StandardScaler()),
    ("model", XGBRegressor())
])

# Define the hyperparameter search space
param_space = {
    "model__learning_rate": Real(0.01, 0.2),
    "model__n_estimators": Integer(50, 300),
    "model__max_depth": Integer(3, 10),
    "model__min_child_weight": Integer(1, 10),
    "model__gamma": Real(0, 0.5),
    "model__subsample": Real(0.5, 1),
    "model__colsample_bytree": Real(0.5, 1),
    "model__reg_alpha": Real(0, 1),
    "model__reg_lambda": Real(0, 1),
    "model__booster": Categorical(["gbtree"]) # Remove "gblinear" and "dart"
}

# Initialize the Bayesian optimization object
bayes_opt = BayesSearchCV(
    pipeline,
    param_space,
    n_iter=50,
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    scoring=mape_scorer
)

# Custom callback to print the evaluation metric during the optimization process
def on_step(optim_result):
    score = -optim_result.fun
    print(f"Best MAPE: {score:.2f}%")
    return False

# Run the Bayesian optimization with the custom callback
bayes_opt.fit(X_train, y_train, callback=on_step)

# Get the best hyperparameters
best_params = bayes_opt.best_params_

# Save the best model
best_pipeline = bayes_opt.best_estimator_
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, "models/best_pipeline.pkl")

# Call the DVC experiment with the best hyperparameters
result = subprocess.run(
    [
        "dvc", "exp", "run", "--set-param", f"learning_rate={best_params['model__learning_rate']}",
        "--set-param", f"n_estimators={best_params['model__n_estimators']}",
        "--set-param", f"max_depth={best_params['model__max_depth']}",
        "--set-param", f"min_child_weight={best_params['model__min_child_weight']}",
        "--set-param", f"gamma={best_params['model__gamma']}",
        "--set-param", f"subsample={best_params['model__subsample']}",
        "--set-param", f"colsample_bytree={best_params['model__colsample_bytree']}",
        "--set-param", f"reg_alpha={best_params['model__reg_alpha']}",
        "--set-param", f"reg_lambda={best_params['model__reg_lambda']}",
        "--set-param", f"booster={best_params['model__booster']}",
    ],
    capture_output=True,
    text=True,
    cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Check if the experiment completed successfully
if result.returncode == 0:
    print("Experiment completed successfully.")
else:
    print("Experiment failed.")
    print(result.stderr)
