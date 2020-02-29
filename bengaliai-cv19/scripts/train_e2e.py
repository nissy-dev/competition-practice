import os
import gc
import torch
import albumentations as albu
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from catalyst.utils import get_device, set_global_seed
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from dataset import BengaliAIDataset
from read_data import read_data, prepare_image
from custom_loss import BaselineLoss
from model import BengaliBaselineClassifier
from metrics import MacroRecallCallback
from offline_models.efficientnet import CustomEfficientNet  # noqa
from offline_models.se_resnext50_32x4d import se_resnext50_32x4d


def main():
    # set your params
    DATA_PATH = '/content/drive/My Drive/kaggle/bengaliai-cv19/dataset'
    MODEL_PATH = '/content/drive/My Drive/kaggle/bengaliai-cv19/model/se_resnext50_32x4d-a260b3a4.pth'
    # MODEL_PATH='/content/drive/My Drive/kaggle/bengaliai-cv19/model/efficientnet-b3-5fb5a3c3.pth'
    BASE_LOGDIR = '/content/drive/My Drive/kaggle/bengaliai-cv19/logs'
    NUM_FOLDS = 5
    BATCH_SIZE = 64
    EPOCHS = 10
    SEED = 1234
    SIZE = 224
    LR = 0.003
    HOLD_OUT = False

    # fix seed
    set_global_seed(SEED)

    # read dataset
    train, _, _ = read_data(DATA_PATH)
    train_all_images = prepare_image(DATA_PATH, data_type='train', submission=False)

    # init
    target_col = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']
    device = get_device()
    train_data_transforms = albu.Compose([
        albu.ShiftScaleRotate(rotate_limit=10, scale_limit=.1),
        albu.Cutout(p=0.5),
    ])
    test_data_transforms = None

    # cross validation
    kf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, random_state=SEED)
    ids = kf.split(X=train_all_images, y=train[target_col].values)
    # fold_scores = []
    for fold, (train_idx, valid_idx) in enumerate(ids):
        print("Current Fold: ", fold + 1)
        logdir = os.path.join(BASE_LOGDIR, 'fold_{}'.format(fold + 1))
        os.makedirs(logdir, exist_ok=True)

        train_df, valid_df = train.iloc[train_idx], train.iloc[valid_idx]
        print("Train and Valid Shapes are", train_df.shape, valid_df.shape)

        print("Preparing train datasets....")
        train_dataset = BengaliAIDataset(
            images=train_all_images[train_idx], labels=train_df[target_col].values,
            size=SIZE, transforms=train_data_transforms
        )

        print("Preparing valid datasets....")
        valid_dataset = BengaliAIDataset(
            images=train_all_images[valid_idx], labels=valid_df[target_col].values,
            size=SIZE, transforms=test_data_transforms
        )

        print("Preparing dataloaders datasets....")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        loaders = {'train': train_loader, 'valid': valid_loader}

        # release memory
        del train_df, valid_df, train_dataset, valid_dataset
        gc.collect()
        torch.cuda.empty_cache()

        # init models
        model = BengaliBaselineClassifier(pretrainedmodels=se_resnext50_32x4d(model_path=MODEL_PATH))
        # model = CustomEfficientNet.from_pretrained('efficientnet-b3', MODEL_PATH)
        model = model.to(device)
        criterion = BaselineLoss()
        optimizer = Adam(model.parameters(), lr=LR)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        callbacks = [
            MacroRecallCallback(),
            EarlyStoppingCallback(patience=5)
        ]

        # catalyst trainer
        runner = SupervisedRunner(device=device)
        # model training
        runner.train(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                     loaders=loaders, callbacks=callbacks, logdir=logdir, num_epochs=EPOCHS,
                     verbose=True)

        # release memory
        del model, runner, train_loader, valid_loader, loaders
        gc.collect()
        torch.cuda.empty_cache()

        if HOLD_OUT is True:
            break

    return True


main()
