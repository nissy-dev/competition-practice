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
    num_class = 6

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

    # set up models
    lgb_params = {
        'learning_rate': 0.1,
        'objective': 'multiclass',
        'num_class': num_class,
        'n_jobs': -1,
        'seed': 1,
    }
    first_layer_classifiers = [KNNClassifier(), SVCClassifier(), LRClassifier()]
    second_layer_classifiers = LGBClassifier(lgb_params)

    # 1st layer
    print('Creating second_layer_inputs ....')
    second_layer_inputs = np.zeros((len(X_train), num_class * len(first_layer_classifiers)))
    kf = GroupKFold(n_splits=num_folds)
    for fold, (train_index, valid_index) in enumerate(
            kf.split(X=subject_train, groups=subject_train)):
        str_fold = 'fold_{}'.format(fold + 1)
        print(str_fold)

        # set data
        x_trn, x_val = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_trn, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]

        # train adn predict
        for i, classifier in enumerate(first_layer_classifiers):
            start_idx, end_idx = i * num_class, (i+1) * num_class
            classifier.train(x_trn, y_trn)
            second_layer_inputs[valid_index, start_idx:end_idx] = classifier.predict(x_val)

    print('finish')
    print(second_layer_inputs[0:3])

    print('Creating test_inputs ....')
    test_inputs = np.zeros((len(X_test), num_class * len(first_layer_classifiers)))
    for i, classifier in enumerate(first_layer_classifiers):
        start_idx, end_idx = i * num_class, (i+1) * num_class
        # 全データで学習
        classifier.train(X_train, y_train)
        test_inputs[:, start_idx:end_idx] = classifier.predict(X_test)

    print('finish')
    print(test_inputs[0:3])

    # 2nd layer
    valid_preds = np.zeros((len(X_train), 6))
    test_preds = np.zeros((num_folds, len(X_test), 6))
    all_score = []
    score_df = pd.DataFrame()
    for fold, (train_index, valid_index) in enumerate(
            kf.split(X=subject_train, groups=subject_train)):
        str_fold = 'fold_{}'.format(fold + 1)
        print(str_fold)

        # set data
        x_trn, x_val = second_layer_inputs[train_index], second_layer_inputs[valid_index]
        y_trn, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]
        second_layer_classifiers.train(x_trn, y_trn, x_val, y_val)
        second_layer_classifiers.predict(x_val)
        valid_preds[valid_index] = second_layer_classifiers.predict(x_val)
        test_preds[fold] = second_layer_classifiers.predict(test_inputs)

        # scoring
        score = accuracy_score(y_val, np.argmax(valid_preds[valid_index], axis=1))
        print('Fold {} Score : {}'.format(fold+1, score))
        all_score.append(score)

    # final score
    print('CV (mean) : {}'.format(np.mean(all_score)))
    score_df['mean'] = [np.mean(all_score)]

    # make submission
    score_df.to_csv(out_dir_path + '/score.csv')
    submit = np.argmax(np.mean(test_preds, axis=0), axis=1) + 1
    np.savetxt(out_dir_path + '/baseline.txt', submit)


main()
