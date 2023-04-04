import pandas as pd
from pandas_profiling import ProfileReport


def generate_eda_report(input_file, output_file):
    df = pd.read_csv(input_file)
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    profile.to_file(output_file)


if __name__ == "__main__":
    input_file = "data/clean_data.csv"
    output_file = "data/pandas_profiling_report.html"
    generate_eda_report(input_file, output_file)
