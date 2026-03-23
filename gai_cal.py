import sys
import datetime
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
from pycaret.regression import *

def split_otu_by_health(meta_path, otu_path):
    # Read meta.tsv and otu.tsv
    meta_df = pd.read_csv(meta_path, sep='\t')
    meta_df = meta_df.set_index('id')

    otu_df = pd.read_csv(otu_path, sep='\t')
    otu_df = otu_df.set_index('id')

    # Split otu_df based on the 'health' column in meta_df
    healthy_otu_df = otu_df[meta_df['health'] == 'y']
    nonhealthy_otu_df = otu_df[meta_df['health'] != 'n']
    predicted_age_df = pd.merge(healthy_otu_df, meta_df["age"], left_index=True, right_index=True, how='inner')

    return healthy_otu_df, nonhealthy_otu_df, predicted_age_df, meta_df, otu_df

class CatBoostRegressorClonable(CatBoostRegressor):
    def __sklearn_clone__(self):
        return CatBoostRegressorClonable(**self.get_params())

def model_health_ages(predicted_age_df, otu_df, output_dir):
    # Use pycaret to model healthy otu_df and predict the physiological age of the samples
    reg = setup(data=predicted_age_df, target='age', session_id=123, verbose = True, n_jobs=16)#, silent = True)
    best_model = compare_models()
    compare_models_df = pull()
    compare_models_df.to_csv('compare_models.tsv', sep='\t', index=True)

    if isinstance(best_model, CatBoostRegressor):
        best_model = CatBoostRegressorClonable(**best_model.get_params())

    tuned_best_model = tune_model(best_model)
    tuned_best_model_df = pull()
    tuned_best_model_df.to_csv(output_dir / 'tuned_best_model.tsv', sep='\t', index=True)

    final_best_model = finalize_model(tuned_best_model)
    age_predictions = predict_model(final_best_model, data=otu_df)

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    save_model(final_best_model, output_dir / f'final_best_model_{current_date}')

    return age_predictions


def calculate_raw_gai(meta_df, age_predictions):
    # Calculate raw GAI by subtracting predicted age from true age and add it as a column in meta_df
    meta_df['raw GAI'] = age_predictions['prediction_label'] - meta_df['age']

    return meta_df


def calculate_adjust_value(meta_df, output_dir):
    # Extract raw GAI for healthy individuals
    health_raw_gai = meta_df[meta_df['health'] == 'y']['raw GAI']

    # Calculate average raw GAI for different age ranges
    age_ranges = [(18, 20), (20, 25), (25, 30), (30, 35), (35, 40), (45, 50), (50, 55), (55, 60), (60, 65),
                  (65, 70), (70, 75), (75, 100)]
    adjust_values = []
    for age_range in age_ranges:
        start_age, end_age = age_range
        avg_raw_gai = health_raw_gai[(meta_df['age'] >= start_age) & (meta_df['age'] < end_age)].mean()
        adjust_values.append(avg_raw_gai)

    # Save adjust_values
    adjust_values_df = pd.DataFrame({'age_range': age_ranges, 'adjust_value': adjust_values})
    adjust_values_df.to_csv(output_dir / 'adjust_values.tsv', sep='\t', index=False)

    # Assign adjust values to samples in meta_df based on their age range
    for i, age_range in enumerate(age_ranges):
        start_age, end_age = age_range
        meta_df.loc[(meta_df['age'] >= start_age) & (meta_df['age'] < end_age), 'adjust value'] = adjust_values[i]

    return meta_df


def calculate_corrected_gai(meta_df):
    # Calculate corrected GAI by subtracting adjust value from raw GAI
    meta_df['corrected GAI'] = meta_df['raw GAI'] - meta_df['adjust value']

    return meta_df


def save_result(meta_df, result_path):
    # Save result_df as result.tsv
    # result_df = meta_df[["age", "raw GAI", "adjust value", "corrected GAI"]]
    meta_df.to_csv(result_path, sep='\t', index=True)
    print(f"Saved result as {result_path}")


def main(meta_path, otu_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split otu.tsv into healthy and nonhealthy otu dataframes
    healthy_otu_df, nonhealthy_otu_df, predicted_age_df, meta_df, otu_df = split_otu_by_health(meta_path, otu_path)

    # Model healthy otu dataframe and predict ages
    age_predictions = model_health_ages(predicted_age_df, otu_df, output_dir)

    # Calculate raw GAI for all samples and add it to meta_df
    meta_df = calculate_raw_gai(meta_df, age_predictions)

    # Calculate adjust values based on age ranges and add them to meta_df
    meta_df = calculate_adjust_value(meta_df, output_dir)

    # Calculate corrected GAI and add it to meta_df
    meta_df = calculate_corrected_gai(meta_df)

    # Save final result as result.tsv
    save_result(meta_df, output_dir / 'result.tsv')


if __name__ == "__main__":
    # TODO update paper's code to actually use argparse
    # Check if the correct number of arguments is passed
    if len(sys.argv) != 4:
        print("Invalid arguments! Please provide the paths to meta.tsv and otu.tsv.")
        print("Usage: python gai_cal.py meta.tsv otu.tsv output_dir")
    else:
        # Get the file paths from command line arguments
        meta_path = sys.argv[1]
        otu_path = sys.argv[2]
        output_dir = sys.argv[3]

        # Call the main function with the file paths
        main(meta_path, otu_path, output_dir)