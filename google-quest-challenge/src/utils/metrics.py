import numpy as np
from scipy.stats import spearmanr


def mean_spearmanr_correlation_score(y, y_pred):
    spearsum = 0
    cnt = 0
    for col in range(y_pred.shape[1]):
        v = spearmanr(y_pred[:, col], y[:, col]).correlation
        # 値が全部同じだとNanになるから除く
        if np.isnan(v):
            continue
        spearsum += v
        cnt += 1
    res = spearsum / cnt
    return res
