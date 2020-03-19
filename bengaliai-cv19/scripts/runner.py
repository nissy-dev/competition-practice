import os
import gc
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


class BengaliRunner:
    def __init__(self, device='cpu'):
        self.device = device

    def train(self, model, criterions, optimizer, loaders, scheduler=None, logdir=None,
              num_epochs=20, score_func=None, is_greater_better=True):
        # validation
        for dict_val in [criterions, loaders]:
            if 'train' in dict_val and 'valid' in dict_val:
                pass
            else:
                raise ValueError('You should set train and valid key.')

        # setup training
        model = model.to(self.device)
        train_loader = loaders['train']
        valid_loader = loaders['valid']
        train_criterion = criterions['valid']
        valid_criterion = criterions['valid']
        best_score = -1.0 if is_greater_better else 10000000
        best_avg_val_loss = 100
        log_df = pd.DataFrame(
            [], columns=['epoch', 'loss', 'valid_loss', 'score', 'recall_grapheme',
                         'recall_consonant', 'recall_vowel', 'time'],
            index=range(num_epochs)
        )
        for epoch in range(num_epochs):
            start_time = time.time()
            # release memory
            torch.cuda.empty_cache()
            gc.collect()
            # train for one epoch
            avg_loss = self._train_model(model, train_criterion, optimizer, train_loader, scheduler)
            # evaluate on validation set
            avg_val_loss, score, scores = self._validate_model(model, valid_criterion, valid_loader, score_func)

            # log
            elapsed_time = time.time() - start_time
            log_df.iloc[epoch] = [epoch + 1, avg_loss, avg_val_loss, score, scores[0], scores[1], scores[2], elapsed_time]

            # the position of this depends on the scheduler you use
            if scheduler is not None:
                scheduler.step()

            # save best params
            save_path = 'best_model.pth'
            if logdir is not None:
                save_path = os.path.join(logdir, save_path)

            if score is None:
                if best_avg_val_loss > avg_val_loss:
                    best_avg_val_loss = avg_val_loss
                    best_param_loss = model.state_dict()
                    torch.save(best_param_loss, save_path)
                    print('Save the best model on Epoch {}'.format(epoch + 1))
            else:
                if is_greater_better and best_score < score:
                    best_score = score
                    best_param_score = model.state_dict()
                    torch.save(best_param_score, save_path)
                    print('Save the best model on Epoch {}'.format(epoch + 1))
                elif is_greater_better is False and best_score > score:
                    best_score = score
                    best_param_score = model.state_dict()
                    torch.save(best_param_score, save_path)
                    print('Save the best model on Epoch {}'.format(epoch + 1))

            # save log
            log_df.to_csv(os.path.join(logdir, 'log.csv'))

        return True

    def predict_loader(self, model, loader, resume='best_model.pth'):
        # set up models
        model = model.to(self.device)
        model.load_state_dict(torch.load(resume))
        model.eval()

        # prediction
        grapheme_preds = []
        consonant_preds = []
        vowel_preds = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(loader), total=len(loader)):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # output
                output_valid = model(images)
                logits = [out.detach().cpu().numpy() for out in output_valid]
                # target_col = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']
                grapheme_preds.extend(logits[0])
                consonant_preds.extend(logits[1])
                vowel_preds.extend(logits[2])

        return grapheme_preds, consonant_preds, vowel_preds

    def _train_model(self, model, criterion, optimizer, train_loader, scheduler=None):
        # switch to train mode
        model.train()
        avg_loss = 0.0
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            # training
            output_train = model(images)
            loss = criterion(output_train, labels)

            # update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calc loss
            avg_loss += loss.item() / len(train_loader)

        return avg_loss

    def _validate_model(self, model, criterion, valid_loader, score_func=None):
        # switch to eval mode
        model.eval()
        avg_val_loss = 0.
        valid_grapheme = []
        valid_consonant = []
        valid_vowel = []
        targets = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # output
                output_valid = model(images)
                avg_val_loss += criterion(output_valid, labels).item() / len(valid_loader)

                # calc score
                logits = [out.detach().cpu().numpy() for out in output_valid]
                labels = labels.detach().cpu().numpy()
                valid_grapheme.extend(logits[0])
                valid_consonant.extend(logits[1])
                valid_vowel.extend(logits[2])
                targets.extend(labels)

            score = None
            if score_func is not None:
                # TODO : you should write valid score calculation
                # In this case, we pass sigmoid function
                targets = np.array(targets)
                valid_preds = [valid_grapheme, valid_consonant, valid_vowel]
                score, scores = score_func(valid_preds, targets)

        return avg_val_loss, score, scores
