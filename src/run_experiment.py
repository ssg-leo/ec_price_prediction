import subprocess
import sys
import os

output_dir = "experiment_models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Get the best hyperparameters from the command-line arguments
best_params = {
    "learning_rate": float(sys.argv[1]),
    "n_estimators": int(sys.argv[2]),
    "max_depth": int(sys.argv[3]),
    "min_child_weight": int(sys.argv[4]),
    "gamma": float(sys.argv[5]),
    "subsample": float(sys.argv[6]),
    "colsample_bytree": float(sys.argv[7]),
    "reg_alpha": float(sys.argv[8]),
    "reg_lambda": float(sys.argv[9]),
    "booster": sys.argv[10],
}

# Get the root directory of your project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Call the DVC experiment with the best hyperparameters
result = subprocess.run(
    [
        "dvc", "exp", "run",
        "--set-param", f"train.learning_rate={best_params['learning_rate']}",
        "--set-param", f"train.n_estimators={best_params['n_estimators']}",
        "--set-param", f"train.max_depth={best_params['max_depth']}",
        "--set-param", f"train.min_child_weight={best_params['min_child_weight']}",
        "--set-param", f"train.gamma={best_params['gamma']}",
        "--set-param", f"train.subsample={best_params['subsample']}",
        "--set-param", f"train.colsample_bytree={best_params['colsample_bytree']}",
        "--set-param", f"train.reg_alpha={best_params['reg_alpha']}",
        "--set-param", f"train.reg_lambda={best_params['reg_lambda']}",
        "--set-param", f"train.booster={best_params['booster']}",
    ],
    capture_output=True,
    text=True,
    cwd=root_dir
)

# Check if the experiment completed successfully
if result.returncode == 0:
    print("Experiment completed successfully.")
else:
    print("Experiment failed.")
    print(result.stderr)
