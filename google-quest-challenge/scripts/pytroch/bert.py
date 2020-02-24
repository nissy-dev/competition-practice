import os
import gc
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from math import floor, ceil
from transformers import BertModel, BertPreTrainedModel


def compute_input_arrays(df, columns, tokenizer, max_sequence_length,
                         t_max_len=30, q_max_len=239, a_max_len=239, head_tail=True):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in df[columns].iterrows():
        # TODO: you will customize this stoken on each competition
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(t, q, a, tokenizer, max_sequence_length, t_max_len, q_max_len, a_max_len, head_tail)
        stoken = ["[CLS]"] + t + ["[SEP]"] + q + ["[SEP]"] + a + ["[SEP]"]
        ids, masks, segments = _convert_to_bert_inputs(stoken, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]


def compute_output_arrays(df, columns):
    # TODO: if label is int, dtype is torch.long
    return torch.tensor(np.asarray(df[columns]), dtype=torch.float32)


class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels=None):
        self.inputs = inputs
        self.lengths = lengths
        self.labels = labels

    def __getitem__(self, idx):
        input_ids = self.inputs[0][idx]
        input_masks = self.inputs[1][idx]
        input_segments = self.inputs[2][idx]
        lengths = self.lengths[idx]
        # for no target
        if self.labels is None:
            return input_ids, input_masks, input_segments, lengths

        labels = self.labels[idx]
        return input_ids, input_masks, input_segments, labels, lengths

    def __len__(self):
        return len(self.inputs[0])


class CustomBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(CustomBertForSequenceClassification, self).__init__(config)
        self.n_use_layer = 4
        self.bert = BertModel(config)
        self.dense = nn.Linear(768*self.n_use_layer, 768*self.n_use_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768*self.n_use_layer, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel.forward
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # TODO: 4層分のCLSの出力を使う方法で実装する
        # outputs[0]を平均とかでPoolするの一般的 (GlobalAveragePoolingを使っていた)
        # outputs[0] : all output of last layer
        # outputs[1] : [CLS] output of last layer
        # outputs[2] : input embedding + 12 hidden layer output
        pooled_output = torch.cat([
            outputs[2][-1*i][:, 0] for i in range(1, self.n_use_layer+1)
        ], dim=1)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertRunner:
    def __init__(self, device='cpu'):
        self.device = device

    def train(self, model, criterion, optimizer, loaders, scheduler=None, logdir=None,
              num_epochs=5, score_func=None):
        model = model.to(self.device)
        train_loader = loaders['train']
        valid_loader = loaders['valid']
        best_score = -1.0
        best_avg_val_loss = 100
        for epoch in range(num_epochs):
            start_time = time.time()
            # release memory
            torch.cuda.empty_cache()
            gc.collect()
            # train for one epoch
            avg_loss = self._train_model(model, criterion, optimizer, train_loader, scheduler)
            # evaluate on validation set
            avg_val_loss, score = self._validate_model(model, criterion, valid_loader, score_func)

            # log
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t score={:.6f} \t time={:.2f}s'.format(
                epoch + 1, num_epochs, avg_loss, avg_val_loss, score, elapsed_time))

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
                if best_score < score:
                    best_score = score
                    best_param_score = model.state_dict()
                    torch.save(best_param_score, save_path)
                    print('Save the best model on Epoch {}'.format(epoch + 1))

        return True

    def predict_loader(self, model, loader, resume='best_model.pth'):
        model = model.to(self.device)
        # load best model
        model.load_state_dict(torch.load(resume))
        model.eval()
        preds = []
        # prediction
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(loader), total=len(loader)):
                input_ids, input_masks, input_segments, labels, _ = batch
                input_ids = input_ids.to(self.device)
                input_masks = input_masks.to(self.device)
                input_segments = input_segments.to(self.device)
                labels = labels.to(self.device)

                # output
                output_valid = model(
                    input_ids=input_ids,
                    labels=None,
                    attention_mask=input_masks,
                    token_type_ids=input_segments,
                )
                logits = output_valid[0]  # output preds
                preds.extend(logits.detach().cpu().squeeze().numpy())

            # TODO : you should write your process
            preds = np.array(preds)
            preds = torch.sigmoid(torch.tensor(preds)).numpy()

        return preds

    def _train_model(self, model, criterion, optimizer, train_loader, scheduler=None):
        # switch to train mode
        model.train()
        avg_loss = 0.0
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            input_ids, input_masks, input_segments, labels, _ = batch
            input_ids = input_ids.to(self.device)
            input_masks = input_masks.to(self.device)
            input_segments = input_segments.to(self.device)
            labels = labels.to(self.device)

            # bert training
            output_train = model(
                input_ids=input_ids,
                labels=None,
                attention_mask=input_masks,
                token_type_ids=input_segments,
            )
            logits = output_train[0]  # output preds
            loss = criterion(logits, labels)

            # update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # the position of this depends on the scheduler you use
            if scheduler is not None:
                scheduler.step()

            # calc loss
            avg_loss += loss.item() / len(train_loader)

        return avg_loss

    def _validate_model(self, model, criterion, valid_loader, score_func=None):
        # switch to eval mode
        model.eval()
        avg_val_loss = 0.
        valid_preds = []
        original = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                input_ids, input_masks, input_segments, labels, _ = batch
                input_ids = input_ids.to(self.device)
                input_masks = input_masks.to(self.device)
                input_segments = input_segments.to(self.device)
                labels = labels.to(self.device)

                # output
                output_valid = model(
                    input_ids=input_ids,
                    labels=None,
                    attention_mask=input_masks,
                    token_type_ids=input_segments,
                )
                logits = output_valid[0]  # output preds

                # calc score
                avg_val_loss += criterion(logits, labels).item() / len(valid_loader)
                valid_preds.extend(logits.detach().cpu().squeeze().numpy())
                original.extend(labels.detach().cpu().squeeze().numpy())

            score = None
            if score_func is not None:
                # TODO : you should write valid score calculation
                # In this case, we pass sigmoid function
                valid_preds = np.array(valid_preds)
                original = np.array(original)
                preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()
                score = score_func(original, preds)

        return avg_val_loss, score


def _trim_input(title, question, answer, tokenizer, max_sequence_length,
                t_max_len=30, q_max_len=239, a_max_len=239, head_tail=False):
    # 239 + 239 + 30 = 508 + 4 = 512
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len + q_len + a_len + 4) > max_sequence_length:

        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        if t_new_len + a_new_len + q_new_len + 4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d"
                             % (max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))

        # Head only
        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]

        # Head + Tail
        # https://arxiv.org/pdf/1905.05583.pdf
        if head_tail:
            q_len_head = q_new_len // 2
            q_len_tail = - (q_new_len - q_len_head)
            a_len_head = a_new_len // 2
            a_len_tail = - (a_new_len - a_len_head)
            q = q[:q_len_head] + q[q_len_tail:]
            a = a[:a_len_head] + a[a_len_tail:]

    return t, q, a


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""

    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")

    segments = []
    first_sep = True
    current_segment_id = 0

    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def _convert_to_bert_inputs(stoken, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)
    return [input_ids, input_masks, input_segments]
