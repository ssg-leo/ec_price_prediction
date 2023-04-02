# EC Price Prediction Project

This repository contains the code and resources for training and packaging an EC price prediction model based on XGBoost. The model predicts the price per square meter (psm) of Executive Condominiums (EC) in Singapore based on various input features.

## Getting Started

### Prerequisites

- Python 3.7+
- Git
- DVC
- AWS account (for data storage)

### Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/ec_price_prediction.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Configure DVC to work with your AWS account:
```bash
dvc remote modify myremote access_key_id <your_aws_access_key_id>
dvc remote modify myremote secret_access_key <your_aws_secret_access_key>
```
## Usage
The dvc.yaml file contains the following stages:

```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/transactions.csv
      - data/hdb-resale-price-index.csv
    outs:
      - data/clean_data.csv

  eda:
    cmd: python src/eda.py
    deps:
      - src/eda.py
      - data/clean_data.csv
    outs:
      - data/pandas_profiling_report.html

  train:
    cmd: python src/train.py
    deps:
      - data/clean_data.csv
      - src/train.py
    outs:
      - models/pipeline.pkl

  optimize:
    cmd: python src/optimize.py
    deps:
      - data/clean_data.csv
      - src/optimize.py
      - models/pipeline.pkl
    outs:
      - models/best_pipeline.pkl

  interpret:
    cmd: python src/interpret.py
    deps:
      - data/clean_data.csv
      - src/interpret.py
      - models/best_pipeline.pkl
    outs:
      - models/shap_summary_plot.png
      - models/explainer.pkl
```

### Running the Entire Pipeline
To execute the entire pipeline, including data preprocessing, EDA, model training, optimization, and interpretation, run:

```bash
dvc repro
```

### Running Individual Stages
To run a specific stage, use the dvc repro command followed by the stage name, for example:

```bash
dvc repro preprocess
```

## Experiment Tracking
To track and compare experiments, use DVC:

1. Run a new experiment with different hyperparameters:
```bash
dvc exp run --set-param train.learning_rate=0.05
```
2. Compare experiments:
```bash
dvc exp show --include-params=train.learning_rate
```
3. Persist the best experiment:
```bash
dvc exp apply <experiment_id>
git add . && git commit -m "Persist best experiment"
```

## Testing
To run tests for the EC price prediction project, execute the following command:

```bash
Copy code
pytest test_*.py
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- XGBoost team for the XGBoost library
- SHAP team for the SHAP library