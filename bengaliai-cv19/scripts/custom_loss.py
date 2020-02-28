import torch.nn as nn


class BaselineLoss(nn.Module):
    def __init__(self):
        super(BaselineLoss, self).__init__()

    def forward(self, pred, target):
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(pred[0], target[:, 0]) + \
            loss_func(pred[1], target[:, 1]) + loss_func(pred[2], target[:, 2])
        return loss
