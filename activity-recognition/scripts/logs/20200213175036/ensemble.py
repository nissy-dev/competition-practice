import sys
import shutil
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from os import path, getcwd, makedirs
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from model_zoo import LGBClassifier, LRClassifier, KNNClassifier, SVCClassifier


def parse_arguments():
    # Set up the argument parser.
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--data-path', '-p', type=str, default='../dataset/raw')
    parser.add_argument('--fold', '-f', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_arguments()
    # create save path
    DATA_DIR = args.data_path
    num_folds = args.fold

    # log directory
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir_path = path.normpath(path.join(getcwd(), 'logs/{}'.format(time)))
    makedirs(out_dir_path, exist_ok=True)
    # copy this file to log dir
    shutil.copy(path.abspath(sys.argv[0]), out_dir_path)

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

    # CV
    valid_preds = np.zeros((len(X_train), 6))
    test_preds = np.zeros((num_folds, len(X_test), 6))
    kf = GroupKFold(n_splits=num_folds)
    score_df = pd.DataFrame()
    all_score = []
    for fold, (train_index, valid_index) in enumerate(
            kf.split(X=subject_train, groups=subject_train)):
        str_fold = 'fold_{}'.format(fold + 1)
        print(str_fold)

        # set data
        x_trn, x_val = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_trn, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]

        lgb_params = {
            'learning_rate': 0.1,
            'objective': 'multiclass',
            'num_class': 6,
            'n_jobs': -1,
            'seed': 1,
        }

        classifiers = [
            LGBClassifier(lgb_params),
            SVCClassifier(), LRClassifier()
        ]

        # train adn predict
        cv_val_preds = []
        cv_test_preds = []
        for classifier in classifiers:
            classifier.train(x_trn, y_trn, x_val, y_val)
            cv_val_preds.append(classifier.predict(x_val))
            cv_test_preds.append(classifier.predict(X_test))

        # ensemble
        valid_preds[valid_index] = np.mean(cv_val_preds, axis=0)
        test_preds[fold] = np.mean(cv_test_preds, axis=0)

        # scoring
        score = accuracy_score(y_val, np.argmax(valid_preds[valid_index], axis=1))
        print('Fold {} Score : {}'.format(fold+1, score))
        score_df[str_fold] = [score]
        all_score.append(score)

    # final score
    print('CV (mean) : {}'.format(np.mean(all_score)))
    score_df['mean'] = [np.mean(all_score)]

    # make submission
    score_df.to_csv(out_dir_path + '/score.csv')
    submit = np.argmax(np.mean(test_preds, axis=0), axis=1) + 1
    np.savetxt(out_dir_path + '/baseline.txt', submit)


main()
