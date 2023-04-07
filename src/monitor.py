import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
import warnings
warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

def monitor_data_drift(reference_df, current_df):
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)

    if not os.path.exists(os.path.join(root_dir, 'monitor')):
        os.makedirs(os.path.join(root_dir, 'monitor'))
    drift_report.save_html(os.path.join(root_dir, 'monitor/data_drift.html'))

def monitor_target_drift(reference_df, current_df):
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(reference_data=reference_df, current_data=current_df)
    target_drift_report.save_html(os.path.join(root_dir, 'monitor/target_drift.html'))

# def monitor_regression_performance(reference_df, current_df):
#     regression_performance_report = Report(metrics=[
#         RegressionPreset(),
#     ])

#     regression_performance_report.run(reference_data=reference_df, current_data=current_df)
#     regression_performance_report.save_html(os.path.join(root_dir, 'monitor/regression_performance.html'))

if __name__ == "__main__":
    
    #Load reference df and current df
    reference_df = pd.read_csv(os.path.join(root_dir, 'data/clean_data.csv'))
    change_df = pd.read_csv(os.path.join(root_dir, 'data/clean_data_delta.csv'))
    target_cols = ['psm']

    #Split df to the inputs and target cols
    keep_columns = [col for col in reference_df.columns if not col in target_cols]
    target_reference_df = reference_df[target_cols]
    reference_df = reference_df[keep_columns]
    target_change_df = change_df[target_cols]
    change_df = change_df[keep_columns]

    if change_df.empty:
        print("No change in data, skipping monitoring...")
    else:
        monitor_data_drift(reference_df, change_df)
        #Target drift and regression report requires 'target' cols
        reference_df['target'] = target_reference_df['psm']
        change_df['target'] = target_change_df['psm']

        monitor_target_drift(reference_df, change_df)
        # monitor_regression_performance(reference_df, change_df)
