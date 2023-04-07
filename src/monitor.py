import pandas as pd
import os
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
import warnings
warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

def monitor_data_drift(reference_df, current_df):
    reference_df = reference_df.drop('psm',axis=1)
    current_df = current_df.drop('psm',axis=1)

    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)

    if not os.path.exists(os.path.join(root_dir, 'monitor')):
        os.makedirs(os.path.join(root_dir, 'monitor'))
    drift_report.save_html(os.path.join(root_dir, 'monitor/data_drift.html'))

def monitor_target_drift(reference_df, current_df):

    reference_df.rename({'psm':'target'}, inplace=True)
    current_df.rename({'psm':'target'}, inplace=True)
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(reference_data=reference_df, current_data=current_df)
    target_drift_report.save_html(os.path.join(root_dir, 'monitor/target_drift.html'))

def monitor_regression_performance(reference_df, current_df):

    ref_psm = reference_df['psm']
    cur_psm = current_df['psm']
    reference_df = reference_df.drop('psm', axis=1)
    current_df = current_df.drop('psm', axis=1)

    pipeline = joblib.load("models/best_pipeline.pkl")
    ref_predict = pipeline.predict(reference_df)
    reference_df['prediction'] = ref_predict
    cur_predict = pipeline.predict(current_df)
    current_df['prediction'] = cur_predict

    reference_df['target'] = ref_psm
    current_df['target'] = cur_psm

    regression_performance_report = Report(metrics=[
        RegressionPreset(),
    ])

    regression_performance_report.run(reference_data=reference_df, current_data=current_df)
    regression_performance_report.save_html(os.path.join(root_dir, 'monitor/regression_performance.html'))

if __name__ == "__main__":
    
    #Load reference df and current df
    reference_df = pd.read_csv(os.path.join(root_dir, 'data/clean_data.csv'))
    change_df = pd.read_csv(os.path.join(root_dir, 'data/clean_data_delta.csv'))

    if change_df.empty:
        print("No change in data, skipping monitoring...")
    else:
        monitor_data_drift(reference_df, change_df)
        #Target drift and regression report requires 'target' cols
        monitor_target_drift(reference_df, change_df)
        monitor_regression_performance(reference_df, change_df)
