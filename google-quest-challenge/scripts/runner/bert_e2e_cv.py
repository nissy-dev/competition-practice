import os
import shutil
import numpy as np
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from catalyst.utils import get_device, set_global_seed
from sklearn.model_selection import GroupKFold
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

# my scripts
from scripts.data.read_data import read_data
from scripts.utils.metrics import mean_spearmanr_correlation_score
from scripts.pytorch.bert import compute_input_arrays, compute_output_arrays, \
    QuestDataset, CustomBertForSequenceClassification, BertRunner


def main():
    # hyper param
    # TODO: set your params
    num_folds = 5
    seed = 1234
    base_dataset_path = '/content/drive/My Drive/kaggle/google-quest-challenge/dataset/'
    batch_size = 4
    num_epochs = 4
    bert_model = 'bert-base-uncased'
    base_logdir = '/kaggle/google_quest/bert'

    # fix seed
    set_global_seed(seed)
    device = get_device()

    # set up logdir
    now = datetime.now()
    base_logdir = os.path.join(base_logdir, now.strftime("%Y%m%d%H%M%S"))
    os.makedirs(base_logdir, exist_ok=True)
    # dump this scripts
    my_file_path = os.path.abspath(__file__)
    shutil.copyfile(my_file_path, base_logdir)

    # load dataset
    # TODO: set your dataset
    train, test, sample_submission = read_data(base_dataset_path)
    input_cols = list(train.columns[[1, 2, 5]])
    target_cols = list(train.columns[11:])
    num_labels = len(target_cols)

    # init Bert
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    # execute CV
    # TODO: set your CV method
    kf = GroupKFold(n_splits=num_folds)
    ids = kf.split(train['question_body'], groups=train['question_body'])
    fold_scores = []
    for fold, (train_idx, valid_idx) in enumerate(ids):
        print("Current Fold: ", fold + 1)
        logdir = os.path.join(base_logdir, 'fold_{}'.format(fold + 1))
        os.makedirs(logdir, exist_ok=True)

        # create dataloader
        train_df, val_df = train.iloc[train_idx], train.iloc[valid_idx]
        print("Train and Valid Shapes are", train_df.shape, val_df.shape)

        print("Preparing train datasets....")
        inputs_train = compute_input_arrays(train_df, input_cols, tokenizer, max_sequence_length=512)
        outputs_train = compute_output_arrays(train_df, columns=target_cols)
        lengths_train = np.argmax(inputs_train[0] == 0, axis=1)
        lengths_train[lengths_train == 0] = inputs_train[0].shape[1]

        print("Preparing valid datasets....")
        inputs_valid = compute_input_arrays(val_df, input_cols, tokenizer, max_sequence_length=512)
        outputs_valid = compute_output_arrays(val_df, columns=target_cols)
        lengths_valid = np.argmax(inputs_valid[0] == 0, axis=1)
        lengths_valid[lengths_valid == 0] = inputs_valid[0].shape[1]

        print("Preparing dataloaders datasets....")
        train_set = QuestDataset(inputs=inputs_train, lengths=lengths_train, labels=outputs_train)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_set = QuestDataset(inputs=inputs_valid, lengths=lengths_valid, labels=outputs_valid)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

        # init models
        model = CustomBertForSequenceClassification.from_pretrained(
            bert_model, num_labels=num_labels, output_hidden_states=True)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0.05, num_training_steps=num_epochs * len(train_loader)
        )

        # model training
        runner = BertRunner(device=device)
        loaders = {'train': train_loader, 'valid': valid_loader}
        print("Model Training....")
        runner.train(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                     loaders=loaders, logdir=logdir, num_epochs=num_epochs,
                     score_func=mean_spearmanr_correlation_score)

        # calc valid score
        best_model_path = os.path.join(logdir, 'best_model.pth')
        val_preds = runner.predict_loader(model, loaders['valid'], resume=best_model_path)
        val_truth = train[target_cols].iloc[valid_idx].values
        # TODO: set your score function
        cv_score = mean_spearmanr_correlation_score(val_truth, val_preds)
        print('Fold {} CV score : {}'.format(fold + 1, cv_score))
        fold_scores.append(cv_score)

    return True
