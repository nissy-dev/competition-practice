import os
import torch
import numpy as np
import torch.nn as nn
from catalyst.utils import get_device
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback
from sklearn.model_selection import GroupKFold
from src.models.nn import SimpleNN, create_data_loader
from src.utils.metrics import mean_spearmanr_correlation_score


def nn_group_kfold_trainer(train, test, features, target):
    # data setup
    X_train = train[features]
    Y_train = train[target]
    X_test = test[features]

    # initialize
    num_folds = 5
    input_dim = X_train.shape[1]
    out_dim = Y_train.shape[1]
    test_preds = np.zeros(len(X_test))
    kf = GroupKFold(n_splits=num_folds)
    fold_scores = []
    test_preds = np.zeros((len(X_test), len(target)))

    # group k-folds
    for fold, (train_index, valid_index) in enumerate(
            kf.split(X=train.question_body, groups=train.question_body)):
        print('Fold {}'.format(fold + 1))

        # params
        batch_size = 128
        num_epochs = 50
        logdir = '/kaggle/working/logs'
        os.makedirs(logdir, exist_ok=True)
        device = get_device()

        # create dataloaders
        train_dls, test_dl = create_data_loader(
            X_train.iloc[train_index].to_numpy(), Y_train.iloc[train_index].to_numpy(),
            X_train.iloc[valid_index].to_numpy(), Y_train.iloc[valid_index].to_numpy(),
            X_test.to_numpy(), batch_size=batch_size
        )

        # init models
        model = SimpleNN(input_dim=input_dim, out_dim=out_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # init catalyst runner
        runner = SupervisedRunner()
        # model training
        runner.train(model=model, criterion=criterion, optimizer=optimizer,
                     scheduler=scheduler, loaders=train_dls, logdir=logdir, num_epochs=num_epochs,
                     callbacks=[EarlyStoppingCallback(patience=15, min_delta=0)], verbose=False)

        # calculate valid score
        best_model_path = logdir + '/checkpoints/best.pth'
        val_preds = runner.predict_loader(model, train_dls['valid'], resume=best_model_path, verbose=False)
        val_truth = Y_train.iloc[valid_index].values
        cv_score = mean_spearmanr_correlation_score(val_truth, val_preds)
        print('Fold {} CV score : {}'.format(fold + 1, cv_score))

        # test prediction
        test_preds += runner.predict_loader(
            model, test_dl, resume=best_model_path, verbose=False) / num_folds
        fold_scores.append(cv_score)

    return fold_scores, test_preds
