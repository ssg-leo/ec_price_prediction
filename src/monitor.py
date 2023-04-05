import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
import warnings
warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

def monitor_data_drift(reference_df, current_df):
    drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)

    drift_report.save_html(os.path.join(root_dir, 'monitor/drift.html'))

if __name__ == "__main__":
    
    #Load reference df and current df
    reference_df = pd.read_csv(os.path.join(root_dir, 'data/clean_data.csv'))
    change_df = pd.read_csv(os.path.join(root_dir, 'data/clean_data_delta.csv'))

    monitor_data_drift(reference_df, change_df)
