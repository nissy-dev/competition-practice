import os
import yaml
import json
import torch
import argparse
import shutil
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime
from catalyst.utils import get_device, set_global_seed
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback
from sklearn.model_selection import KFold
from src.models.nn import SampleNN, create_data_loader
from src.utils.metrics import mean_spearmanr_correlation_score


def yaml_to_json(path_to_yaml):
    with open(path_to_yaml) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params


def parse_arguments():
    parser = argparse.ArgumentParser(description='Catalyst Sample Trainer')
    parser.add_argument('--yaml-path', default='config.yaml',
                        type=str, help='path to config (relative path)')
    return parser.parse_args()


def main(train, test, features, target):
    # get args
    args = parse_arguments()
    params = yaml_to_json(args.yaml_path)

    # hyper param
    num_folds = params.fold
    seed = params.seed
    base_path = params.base_path
    target_cols = params.target
    features_cols = params.features
    preprocessed_data_path = params.preprocessed_data
    batch_size = params.batch_size
    num_epochs = params.epochs
    # ex) '/hoge/logs'
    base_logdir = params.base_logdir

    # fix seed
    set_global_seed(seed)
    device = get_device()

    # set up logdir
    now = datetime.now()
    base_logdir = os.path.join(base_logdir + now.strftime("%Y%m%d%H%M%S"))
    os.makedirs(base_logdir, exist_ok=True)
    # dump yaml contents
    with open(os.path.join(base_logdir, 'params.json'), mode="w") as f:
        json.dump(params, f, indent=4)
    # dump this scripts
    my_file_path = os.path.abspath(__file__)
    shutil.copyfile(my_file_path, base_logdir)

    # load dataset
    if preprocessed_data_path == '':
        train, test, sample_submission = read_data(base_path)  # noqa
        # TODO: You should implement these function!!
        train, test = preprocess(train, test)  # noqa
        train, test = build_feature(train, test)  # noqa
    else:
        train = pd.read_csv(preprocessed_data_path + 'train.csv')
        test = pd.read_csv(preprocessed_data_path + 'test.csv')
        sample_submission = pd.read_csv(preprocessed_data_path + 'sample_submission.csv')

    # execute CV
    # TODO: set your CV method
    kf = KFold(n_splits=num_folds, random_state=seed)
    ids = kf.split(train)
    fold_scores = []
    test_preds = []
    for fold, (train_idx, valid_idx) in enumerate(ids):
        print('Fold {}'.format(fold + 1))

        logdir = os.path.join(base_logdir + 'fold_{}'.format(fold + 1))
        os.makedirs(logdir, exist_ok=True)

        # data
        X_train = train[features_cols]
        # 目的変数の正規化は...?
        Y_train = train[target_cols]
        X_test = train[features_cols]

        # create dataloaders
        train_dls, test_dl = create_data_loader(
            X_train.iloc[train_idx].to_numpy(), Y_train.iloc[train_idx].to_numpy(),
            X_train.iloc[valid_idx].to_numpy(), Y_train.iloc[valid_idx].to_numpy(),
            X_test.to_numpy(), batch_size=batch_size
        )

        # init models
        # TODO: set your model and learning condition
        # ここは関数を用意して、キーワードで取り出すようにできると汎用性は上がる
        model = SampleNN(input_dim=1000, out_dim=1)
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # init catalyst runner
        runner = SupervisedRunner(device=device)
        # model training
        runner.train(model=model, criterion=criterion, optimizer=optimizer,
                     scheduler=scheduler, loaders=train_dls, logdir=logdir, num_epochs=num_epochs,
                     callbacks=[EarlyStoppingCallback(patience=15, min_delta=0)], verbose=False)

        # calculate valid score
        best_model_path = logdir + '/checkpoints/best.pth'
        val_preds = runner.predict_loader(model, train_dls['valid'], resume=best_model_path, verbose=False)
        val_truth = Y_train.iloc[valid_idx].values
        # TODO: set your score function
        cv_score = mean_spearmanr_correlation_score(val_truth, val_preds)
        print('Fold {} CV score : {}'.format(fold + 1, cv_score))
        fold_scores.append(cv_score)

        # test prediction
        test_pred = runner.predict_loader(
            model, test_dl, resume=best_model_path, verbose=False) / num_folds
        test_preds.append(test_pred)

    # submit
    # TODO: set your submit process
    sample_submission[target_cols] = np.mean(test_preds, axis=0)
    sample_submission.to_csv('submission.csv')
    return True
