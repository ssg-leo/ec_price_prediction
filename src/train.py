import argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from feature_engine.encoding import MeanEncoder
import joblib
import os


# Custom scoring function
def mean_absolute_percentage_error(y_true, y_pred):
    return (100 * mean_absolute_error(y_true, y_pred)) / y_true.mean()

# Load your data
data_path = os.path.join(os.path.abspath("data"), "clean_data.csv")
data = pd.read_csv(data_path)
X = data.drop("psm", axis=1)
y = data["psm"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the categorical columns
categorical_columns = ["floor_range", "type_of_sale", "district_name"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_child_weight", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--subsample", type=float, default=1)
    parser.add_argument("--colsample_bytree", type=float, default=1)
    parser.add_argument("--reg_alpha", type=float, default=0)
    parser.add_argument("--reg_lambda", type=float, default=1)
    parser.add_argument("--booster", type=str, default="gbtree")
    args = parser.parse_args()

    # Create the pipeline
    pipeline = Pipeline([
        ("mean_encoder", MeanEncoder(variables=categorical_columns)),
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_child_weight=args.min_child_weight,
            gamma=args.gamma,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            booster=args.booster
        ))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Save the model
    joblib.dump(pipeline, "models/best_pipeline.pkl")

    # Calculate the MAPE on the test set
    y_pred = pipeline.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
