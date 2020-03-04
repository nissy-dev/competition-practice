import torch
import torch.nn as nn
import torch.nn.functional as F


# https://www.kaggle.com/c/bengaliai-cv19/discussion/128637
def ohem_loss(cls_pred, cls_target, rate=0.75):
    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


class BaselineLoss(nn.Module):
    def __init__(self):
        super(BaselineLoss, self).__init__()

    def forward(self, pred, target):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred[0], target[:, 0]) + \
            criterion(pred[1], target[:, 1]) + criterion(pred[2], target[:, 2])
        return loss


class MixupLoss(nn.Module):
    def __init__(self):
        super(MixupLoss, self).__init__()

    def forward(self, preds, targets):
        preds1, preds2, preds3 = preds
        targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets2, lam = targets
        criterion = nn.CrossEntropyLoss(reduction='mean')
        return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, shuffled_targets1) + \
            lam * criterion(preds2, targets2) + (1 - lam) * criterion(preds2, shuffled_targets2) + \
            lam * criterion(preds3, targets3) + (1 - lam) * criterion(preds3, shuffled_targets2)
