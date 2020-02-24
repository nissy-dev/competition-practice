import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score
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


def threshold_search(y_true, y_proba, func, is_higher_better=True):
    """2値分類の閾値探索
    F1 score, Matthews Correlation Coefficient(MCC), Fβ score で使える
    """
    best_threshold = 0.0
    best_score = -np.inf if is_higher_better else np.inf
    for threshold in [i * 0.01 for i in range(100)]:
        score = func(y_true=y_true, y_pred=y_proba > threshold)
        if is_higher_better:
            if score > best_score:
                best_threshold = threshold
                best_score = score
        else:
            if score < best_score:
                best_threshold = threshold
                best_score = score

    search_result = {
        "threshold": best_threshold,
        "score": best_score
    }
    return search_result


# 1つのラベルしか使えないことに注意
class OptimizedRounder(object):
    def __init__(self, n_overall=5, n_classwise=5, n_classes=7, metric=accuracy_score):
        self.n_overall = n_overall  # 探索回数...?
        self.n_classwise = n_classwise  # 探索回数...?
        self.n_classes = n_classes  # クラス数
        self.coef = [1.0 / n_classes * i for i in range(1, n_classes)]
        self.metric = metric  # set your metric function

    def _loss(self, X, y):
        X_p = np.digitize(X, self.coef)
        ll = -self.metric(y, X_p)
        # for qwk, ll = -self.metric(y, X_p, self.n_classes - 1)
        return ll

    def fit(self, X, y):
        golden1 = 0.618
        golden2 = 1 - golden1
        ab_start = [
            (0.01, 1.0 / self.n_classes + 0.05),
        ]
        for i in range(1, self.n_classes):
            ab_start.append((i * 1.0 / self.n_classes + 0.05,
                             (i + 1) * 1.0 / self.n_classes + 0.05))
        for _ in range(self.n_overall):
            for idx in range(self.n_classes - 1):
                # golden section search
                a, b = ab_start[idx]
                # calc losses
                self.coef[idx] = a
                la = self._loss(X, y)
                self.coef[idx] = b
                lb = self._loss(X, y)
                for it in range(self.n_classwise):
                    # choose value
                    if la > lb:
                        a = b - (b - a) * golden1
                        self.coef[idx] = a
                        la = self._loss(X, y)
                    else:
                        b = b - (b - a) * golden2
                        self.coef[idx] = b
                        lb = self._loss(X, y)

    def predict(self, X):
        X_p = np.digitize(X, self.coef)
        return X_p
