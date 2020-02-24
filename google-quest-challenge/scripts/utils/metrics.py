import numpy as np
from scipy.stats import spearmanr


def mean_spearmanr_correlation_score(y_true, y_pred):
    num_labels = y_pred.shape[1]
    score = np.nanmean([spearmanr(y_pred[:, col], y_true[:, col]).correlation
                        for col in range(num_labels)])
    return score


def qwk(y_true, y_pred, max_rat=3):
    # max_rat = class_num - 1
    # same as the folowing
    # from sklearn.metrics import cohen_kappa_score
    # qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    y_true_ = np.asarray(y_true)
    y_pred_ = np.asarray(y_pred)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    uniq_class = np.unique(y_true_)
    for i in uniq_class:
        hist1[int(i)] = len(np.argwhere(y_true_ == i))
        hist2[int(i)] = len(np.argwhere(y_pred_ == i))

    numerator = np.square(y_true_ - y_pred_).sum()

    denominator = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            denominator += hist1[i] * hist2[j] * (i - j) * (i - j)

    denominator /= y_true_.shape[0]
    return 1 - numerator / denominator
