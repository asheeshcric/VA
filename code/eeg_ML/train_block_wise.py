import random
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def gen_dataset(data_dir, user_ids, window_size, fatigue_block):
    X, y = pd.DataFrame(), pd.DataFrame()
    for user_id in user_ids:
        user_dir = os.path.join(data_dir, f"user_{user_id}")
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
                eeg_path = os.path.join(
                    block_dir,
                    "eeg",
                    f"features_ws_{window_size}_fb_{fatigue_block}.csv",
                )
                if not os.path.exists(eeg_path):
                    continue

                block_features = pd.read_csv(eeg_path)
                block_features = block_features.fillna(0)
                y = y.append(pd.DataFrame(block_features["fatigue_label"]))
                X = X.append(block_features.drop(columns="fatigue_label"))

    return X, y


def train_test_split(data_dir, window_size, fatigue_block, test_pct=0.3):
    users = list(range(1, 10))
    test_ids = random.sample(users, int(len(users) * test_pct))
    train_ids = list(set(users) - set(test_ids))

    X_train, y_train = gen_dataset(data_dir, train_ids, window_size, fatigue_block)
    X_test, y_test = gen_dataset(data_dir, test_ids, window_size, fatigue_block)

    return X_train, y_train, X_test, y_test, test_ids


def train(X_train, y_train):
    print("Training started...")
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    print("Training ended...")
    return clf


def train_RF(X_train, y_train):
    print("Training started...")
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Training ended...")
    return clf


def test(clf, X_test, y_test):
    print("Testing started...")
    y_pred = clf.predict(X_test)
    y_true, y_predicted = y_test.tolist(), y_pred.tolist()
    cf = confusion_matrix(y_true, y_predicted)
    acc = accuracy_score(y_true, y_predicted)
    print(cf)
    print(f"Accuracy Score: {acc}")


def test_block_wise(clf, test_user_ids, data_dir, window_size, fatigue_block):
    true_classes = []
    pred_classes = []
    for user_id in test_user_ids:
        user_dir = os.path.join(data_dir, f"user_{user_id}")
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
                eeg_path = os.path.join(
                    block_dir,
                    "eeg",
                    f"features_ws_{window_size}_fb_{fatigue_block}.csv",
                )
                if not os.path.exists(eeg_path):
                    continue

                block_features = pd.read_csv(eeg_path)
                block_features = block_features.fillna(0)
                y_test = np.ravel(
                    pd.DataFrame(block_features["fatigue_label"]).to_numpy().astype(int)
                )
                X_test = (
                    block_features.drop(columns="fatigue_label")
                    .to_numpy()
                    .astype(float)
                )
                y_pred = clf.predict(X_test)
                pred_class = np.bincount(y_pred).argmax()
                true_class = np.bincount(y_test).argmax()
                true_classes.append(true_class)
                pred_classes.append(pred_class)

    # print(pred_classes, true_classes)
    print(confusion_matrix(true_classes, pred_classes))
    print(accuracy_score(true_classes, pred_classes))


def main(window_size, fatigue_block):
    print("Running script...")
    data_dir = "/home/ashish/Documents/github/VA/data/cognitive_data/"
    X_train, y_train, X_test, y_test, test_user_ids = train_test_split(
        data_dir, window_size, fatigue_block, test_pct=0.3
    )
    X_train_np, X_test_np = X_train.to_numpy(), X_test.to_numpy()
    X_train_np, X_test_np = X_train_np.astype(float), X_test_np.astype(float)
    y_train_np, y_test_np = y_train.to_numpy(), y_test.to_numpy()
    y_train_np, y_test_np = y_train_np.astype(int), y_test_np.astype(int)
    clf = train_RF(X_train_np, np.ravel(y_train_np))
    test(clf, X_test_np, np.ravel(y_test_np))
    test_block_wise(clf, test_user_ids, data_dir, window_size, fatigue_block)


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
