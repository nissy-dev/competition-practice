import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier


def parse_arguments():
    # Set up the argument parser.
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--data-path', '-p', type=str, default='../dataset/raw')
    parser.add_argument('--fold', '-f', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_arguments()
    # params
    DATA_DIR = args.data_path
    num_folds = args.fold
    seed = 1234

    # setup data
    with open(DATA_DIR + '/features.txt') as f:
        features_txt = f.readlines()
    features_name = [x.strip() for x in features_txt]
    features_name = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in features_name]
    X_train = pd.read_csv(DATA_DIR + '/X_train.csv', names=features_name)
    X_test = pd.read_csv(DATA_DIR + '/X_test.csv', names=features_name)
    y_train = pd.read_csv(DATA_DIR + '/y_train.csv', names=['activity_label'])
    subject_train = pd.read_csv(DATA_DIR + '/subject_train.csv', names=['subject_id'])

    # 0始まりにする
    y_train['activity_label'] = y_train['activity_label'] - 1

    # set up models
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, random_state=seed)),
        ('svr', SVC(probability=True, random_state=seed)),
        ('knn', KNeighborsClassifier())
    ]
    final_estimator = LogisticRegression(random_state=seed)
    kf = GroupKFold(n_splits=num_folds)
    cv_idx = kf.split(X=subject_train, groups=subject_train)
    clf = StackingClassifier(
        estimators=estimators, final_estimator=final_estimator, cv=cv_idx
    )

    # train
    clf.fit(X_train, y_train)

    # make submission
    test_preds = clf.predict(X_test)
    submit = test_preds + 1
    np.savetxt('baseline.txt', submit)


main()
