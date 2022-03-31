from distutils.log import error
import os

import numpy as np
import pandas as pd


def read_eeg_bands(eeg_path):
    bands = {
        "a": [],
        "b": [],
        "d": [],
        "g": [],
        "t": [],
        # 'Aa': [], 'Ab': [], 'Ad': [], 'Ag': [], 'At': [],
        # 'as': [], 'bs': [], 'ds': [], 'gs': [], 'ts': [],
        # For now, we are only interested in the processed a,b,d,g,t bands for our datasets
    }
    with open(eeg_path, "r") as file:
        for line in file.readlines():
            values = line.split()
            if not values:
                continue

            if values[0] in bands.keys():
                bands[values[0]].append(list(map(lambda x: float(x), values[1:])))

    return bands


def get_eeg_filename(eeg_dir):
    for filename in os.listdir(eeg_dir):
        if len(filename.split(".")) == 1:
            # if the file doesn't have any extension
            return filename
        
    return os.listdir(eeg_dir)[0]


def store_to_csv(eeg_path, dest_dir, eeg_filename):
    try:
        col_names = ["timestep"]
        for wave in ["alpha", "beta", "delta", "gamma", "theta", "h"]:
            for i in [1, 2, 3, 4]:
                col_names.append(f"{wave}_{i}")

        col_names.append("c")

        bands = [
            "a",
            "b",
            "d",
            "g",
            "t",
            "h",
            "c",
        ]  # h and c are quality and concentration mesaurements
        timestep = 1
        file_data, row_data = [], [timestep]
        with open(eeg_path, "r") as file:
            for line in file.readlines():
                values = line.split()
                if not values:
                    continue

                band = values[0]
                if band in bands:
                    for value in values[1:]:
                        row_data.append(float(value))

                    if band == "c":  # the last one required for this row
                        file_data.append(row_data)
                        timestep += 1
                        row_data = [timestep]

        print(len(file_data))

        df = pd.DataFrame(file_data, columns=col_names)
        df.to_csv(os.path.join(dest_dir, f"{eeg_filename}.csv"), index=False)
        return True, "Success"
    except Exception as error:
        return False, error


def main():
    cog_data_dir = "/home/ashish/Documents/github/VA/data/cognitive_data"
    phy_data_dir = "/home/ashish/Documents/github/VA/data/physical_data"

    session_counter = 0
    for user_id in range(1, 10):
        user_dir = os.path.join(cog_data_dir, f"user_{user_id}")
        for session in os.listdir(user_dir):
            session_dir = os.path.join(user_dir, session)
            for block in os.listdir(session_dir):
                # Sanity check if the directory has the name "block" or not
                if "block" not in block or "practice" in block.lower():
                    # Ignore directories other than block
                    continue
                block_dir = os.path.join(session_dir, block)
                eeg_dir = os.path.join(block_dir, "eeg")
                eeg_filename = get_eeg_filename(eeg_dir)
                eeg_path = os.path.join(block_dir, "eeg", eeg_filename)
                print(
                    f"{session_counter+1}. Session: {session[-1]} | User_ID: {user_id} | Session: {session} | Block_dir: {block}"
                )
                session_counter += 1
                # bands = read_eeg_bands(eeg_path) # Use for plotting purposes only
                success, message = store_to_csv(eeg_path, eeg_dir, eeg_filename)
                if not success:
                    print(f"{eeg_path} | {message}")


if __name__ == "__main__":
    main()
