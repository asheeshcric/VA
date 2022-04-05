import os

cog_data_dir = "/home/ashish/Documents/github/VA/data/cognitive_data"

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
            for file in os.listdir(eeg_dir):
                if "features" in file:
                    file_path = os.path.join(eeg_dir, file)
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
