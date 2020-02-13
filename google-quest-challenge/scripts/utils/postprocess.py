import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler


def rank_average(preds):
    rank_preds = np.zeros(preds.shape)
    num_col = preds.shape[1]
    for i in range(num_col):
        # ランクに変換
        rank_preds[:, i] = rankdata(preds[:, i])
    # 正規化
    rank_preds = MinMaxScaler().fit_transform(rank_preds)
    return rank_preds
