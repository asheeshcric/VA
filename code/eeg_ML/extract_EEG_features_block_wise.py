"""
This script extracts temporal features from the EEG data for data from multiple windows (with fixed size)
Stores all data from one subject in one single csv file
"""
from operator import index
import os
from argparse import ArgumentParser
from click import Argument

import numpy as np
import pandas as pd
from parso import parse
from scipy.stats import stats


class TemporalFeatures:
    def __init__(self):
        pass

    def get_slope(self, window):
        times = np.array(range(0, len(window.index)))
        data = window.astype(np.float32)

        # Check for NaNs
        mask = ~np.isnan(data)

        # If all points inside the window are NaN, then we return NaN
        if len(data[mask]) == 0:
            return np.nan
        else:
            slope, _, _, _, _ = stats.linregress(times[mask].data[mask])
            return slope

    def aggregate_values(self, df, window_size, aggregation_metric):
        try:
            if aggregation_metric == "mean":
                # return df.rolling(window_size, min_periods=0).mean()
                return df.groupby(df.index // window_size).mean()
            elif aggregation_metric == "max":
                # return df.rolling(window_size, min_periods=0).max()
                return df.groupby(df.index // window_size).max()
            elif aggregation_metric == "min":
                # return df.rolling(window_size, min_periods=0).min()
                return df.groupby(df.index // window_size).min()
            elif aggregation_metric == "median":
                # return df.rolling(window_size, min_periods=0).median()
                return df.groupby(df.index // window_size).median()
            elif aggregation_metric == "std":
                # return df.rolling(window_size, min_periods=0).std()
                return df.groupby(df.index // window_size).std()
            elif aggregation_metric == "slope":
                # return df.rolling(window_size, min_periods=0).apply(self.get_slope)
                return df.groupby(df.index // window_size).apply(self.get_slope)
            else:
                return np.nan
        except Exception as error:
            # For any kind of file/data error
            return np.nan

    def add_temporal_features(self, df, cols, window_size, aggregation_metrics):
        features_df = pd.DataFrame([])
        for aggregation_metric in aggregation_metrics:
            for col in cols:
                col_name = col + f"_{aggregation_metric}_ws_" + str(window_size)
                if col_name not in features_df:
                    features_df[col_name] = self.aggregate_values(
                        df[col], window_size, aggregation_metric
                    )
                else:
                    features_df[col_name].append(
                        self.aggregate_values(df[col], window_size, aggregation_metric)
                    )
        return features_df


def get_eeg_csv_filename(eeg_dir):
    for filename in os.listdir(eeg_dir):
        if ".csv" in filename:
            return filename


def extract_features(eeg_data_file, window_size):
    X = pd.read_csv(eeg_data_file)
    # Select columns with (alpha, beta, delta, gamma, and theta)
    selected_cols = list(X.columns)[1:-5]
    temp_features = TemporalFeatures()
    X_features = temp_features.add_temporal_features(
        X.copy(), selected_cols, window_size, ["mean", "std", "max", "min", "median"]
    )
    return X_features


def main(window_size, fatigue_block):
    cog_data_dir = "/home/ashish/Documents/github/VA/data/cognitive_data"
    phy_data_dir = "/home/ashish/Documents/github/VA/data/physical_data"

    session_counter = 0
    for user_id in range(1, 10):
        user_dir = os.path.join(cog_data_dir, f"user_{user_id}")
        for session in os.listdir(user_dir):
            # Sanity check for session directory name
            if "session" not in session:
                continue
            session_dir = os.path.join(user_dir, session)
            for block in os.listdir(session_dir):
                # Sanity check if the directory has the name "block" or not
                if "block" not in block or "practice" in block.lower():
                    # Ignore directories other than block
                    continue
                block_dir = os.path.join(session_dir, block)
                eeg_dir = os.path.join(block_dir, "eeg")
                eeg_filename = get_eeg_csv_filename(eeg_dir)
                eeg_path = os.path.join(block_dir, "eeg", eeg_filename)
                user_features = extract_features(eeg_path, window_size)
                fatigue = 0 if int(block[5]) < fatigue_block else 1
                user_features["fatigue_label"] = fatigue
                features_path = os.path.join(
                    block_dir,
                    "eeg",
                    f"features_ws_{window_size}_fb_{fatigue_block}.csv",
                )
                user_features.to_csv(features_path, index=False)

                print(
                    f"{session_counter+1}. Session: {session[-1]} | User_ID: {user_id} | Session: {session} | Block_dir: {block}"
                )
                session_counter += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-w",
        "--window_size",
        type=int,
        default=50,
        help="Window size to aggregate temporal features",
    )
    parser.add_argument(
        "-fb",
        "--fatigue_block",
        type=int,
        default=4,
        help="Block number from which we consider subject to be fatigued",
    )
    args = parser.parse_args()
    main(args.window_size, args.fatigue_block)
