import numpy as np
from scipy.stats import spearmanr


def mean_spearmanr_correlation_score(y_true, y_pred):
    num_labels = y_pred.shape[1]
    score = np.nanmean([spearmanr(y_pred[:, col], y_true[:, col]).correlation
                        for col in range(num_labels)])
    return score
