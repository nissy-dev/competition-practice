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
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(pred[0], target[:, 0]) + \
            loss_func(pred[1], target[:, 1]) + loss_func(pred[2], target[:, 2])
        return loss
