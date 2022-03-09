from multiprocessing.sharedctypes import Value
import os
from csv import reader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import neurokit2 as nk
import hrvanalysis as hrvana


def is_cognitively_fatigued(block):
    # If the subject is in the first 3 blocks, we assume they are not fatigued
    block_num = int(block.split("_")[0][-1])
    return block_num, 0 if block_num <= 3 else 1


def get_n_back_score(block_dir):
    # return the final score (float from 0.0-1.0) of the N-back task that the user played (0-100)
    n_back_file = None
    for content in os.listdir(block_dir):
        if ".csv" in content:
            n_back_file = content
            break
    if not n_back_file:
        return 0

    with open(os.path.join(block_dir, n_back_file), "r") as file:
        csv_reader = reader(file)
        last_row = list(csv_reader)[-1]

    # Example row: ['3_Letter_C.png', '', '11', '0', '67', '1', '97.46835443037975', '0.0']
    # Second last column of the last row represents the final score in the game
    return round(float(last_row[-2]) / 100, 2)


def extract_EMG_features():
    pass


def extract_EDA_features():
    pass


def extract_ECG_features(ecg_df):
    peaks, info = nk.ecg_peaks(ecg_df.values, sampling_rate=1000)
    hrv_features = nk.hrv(peaks, sampling_rate=1000, show=False)
    return hrv_features


def extract_features(block_dir):
    bsp_dir = os.path.join(block_dir, "bsp")
    bsp_file_name = os.listdir(bsp_dir)[0]
    bsp_file_path = os.path.join(bsp_dir, bsp_file_name)
    # bsp_data contains: [ECG, GSR, Breathing, EMG, PulOxR, PulOxIR] columns
    bsp_data = pd.read_csv(bsp_file_path)
    ecg_features = extract_ECG_features(bsp_data["ECG"])
    return ecg_features


if __name__ == "__main__":
    cog_data_dir = "/home/ashish/Documents/github/VA/data/cognitive_data"
    phy_data_dir = "/home/ashish/Documents/github/VA/data/physical_data"
    session_counter = 0
    data_features = pd.DataFrame([])
    for user_id in range(1, 10):
        user_dir = os.path.join(cog_data_dir, f"user_{user_id}")
        for session in os.listdir(user_dir):
            session_dir = os.path.join(user_dir, session)
            for block in os.listdir(session_dir):
                # Sanity check if the directory has the name "block" or not
                if "block" not in block or "practice" in block.lower():
                    # Ignore directories other than block
                    continue

                try:
                    block_dir = os.path.join(session_dir, block)
                    # For each block, we want to extract three different sets of data
                    score = get_n_back_score(block_dir)
                    print(
                        f"{session_counter+1}. Score: {score} | Session: {session[-1]} | User_ID: {user_id}"
                    )
                    session_counter += 1
                    bsp_features = extract_features(block_dir)
                    block_num, cog_fatigue = is_cognitively_fatigued(block)
                    bsp_features.insert(loc=0, column="user_id", value=user_id)
                    bsp_features.insert(loc=1, column="block_num", value=block_num)
                    bsp_features["n_back_score"] = score
                    bsp_features["cog_fatigue"] = cog_fatigue
                    data_features = data_features.append(
                        bsp_features, ignore_index=True
                    )

                except Exception as error:
                    print("--------------------")
                    print(
                        f"ERROR: {session_counter+1}. Score: {score} | Session: {session[-1]} | User_ID: {user_id}"
                    )
                    print(error)
                    print("--------------------")

    data_features.to_csv("bsp_features.csv")
