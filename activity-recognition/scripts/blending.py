import sys
import shutil
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from os import path, getcwd, makedirs
from sklearn.model_selection import KFold, train_test_split
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
    # params
    DATA_DIR = args.data_path
    num_folds = args.fold
    num_class = 6
    seed = 1234

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
    # subject_train = pd.read_csv(DATA_DIR + '/subject_train.csv', names=['subject_id'])

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
    print('Creating second_layer_inputs and test_inputs ....')
    # hold out
    x_trn, x_val, y_trn, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    second_layer_inputs = np.zeros((len(x_val), num_class * len(first_layer_classifiers)))
    test_inputs = np.zeros((len(X_test), num_class * len(first_layer_classifiers)))
    # train adn predict
    for i, classifier in enumerate(first_layer_classifiers):
        start_idx, end_idx = i * num_class, (i+1) * num_class
        classifier.train(x_trn, y_trn)
        second_layer_inputs[:, start_idx:end_idx] = classifier.predict(x_val)
        test_inputs[:, start_idx:end_idx] = classifier.predict(X_test)

    print('finish')
    print(second_layer_inputs[0:3])
    print(test_inputs[0:3])

    # 2nd layer
    valid_preds = np.zeros((len(second_layer_inputs), 6))
    test_preds = np.zeros((num_folds, len(X_test), 6))
    all_score = []
    score_df = pd.DataFrame()
    kf = KFold(shuffle=True, random_state=seed)
    for fold, (train_index, valid_index) in enumerate(kf.split(second_layer_inputs)):
        str_fold = 'fold_{}'.format(fold + 1)
        print(str_fold)

        # set data
        x_trn, x_val = second_layer_inputs[train_index], second_layer_inputs[valid_index]
        y_2nd_trn, y_2nd_val = y_val.iloc[train_index], y_val.iloc[valid_index]
        second_layer_classifiers.train(x_trn, y_2nd_trn, x_val, y_2nd_val)
        valid_preds[valid_index] = second_layer_classifiers.predict(x_val)
        test_preds[fold] = second_layer_classifiers.predict(test_inputs)

        # scoring
        score = accuracy_score(y_2nd_val, np.argmax(valid_preds[valid_index], axis=1))
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
