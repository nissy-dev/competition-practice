import torch.nn as nn


def baseline_loss(pred, target):
    loss_grapheme = nn.CrossEntropyLoss()
    loss_vowel = nn.CrossEntropyLoss()
    loss_consonant = nn.CrossEntropyLoss()
    loss = loss_grapheme(pred[0], target[:, 0]) + \
        loss_vowel(pred[1], target[:, 1]) + loss_consonant(pred[2], target[:, 2])
    return loss
