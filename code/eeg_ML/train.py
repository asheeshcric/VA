import random
import os

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def train_test_split(data_dir, test_pct=0.3):
    users = list(range(1, 10))
    test = random.sample(users, int(len(users) * test_pct))
    train = list(set(users) - set(test))
    X_train, y_train, X_test, y_test = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    for user_id in train:
        X = pd.read_csv(os.path.join(data_dir, f"user_{user_id}.csv"))
        y_train = y_train.append(pd.DataFrame(X["fatigue_label"]))
        X_train = X_train.append(X.drop(columns="fatigue_label"))
        X_train = X_train.fillna(0)

    for user_id in test:
        X = pd.read_csv(os.path.join(data_dir, f"user_{user_id}.csv"))
        y_test = y_test.append(pd.DataFrame(X["fatigue_label"]))
        X_test = X_test.append(X.drop(columns="fatigue_label"))
        X_test = X_test.fillna(0)

    return X_train, y_train, X_test, y_test


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


def main():
    print("Running script...")
    data_dir = (
        "/home/ashish/Documents/github/VA/data/cognitive_data/eeg_features_ws_50"
    )
    X_train, y_train, X_test, y_test = train_test_split(data_dir, test_pct=0.3)
    X_train_np, X_test_np = X_train.to_numpy(), X_test.to_numpy()
    X_train_np, X_test_np = X_train_np.astype(float), X_test_np.astype(float)
    y_train_np, y_test_np = y_train.to_numpy(), y_test.to_numpy()
    y_train_np, y_test_np = y_train_np.astype(int), y_test_np.astype(int)
    clf = train_RF(X_train_np, np.ravel(y_train_np))
    test(clf, X_test_np, np.ravel(y_test_np))


if __name__ == "__main__":
    main()
