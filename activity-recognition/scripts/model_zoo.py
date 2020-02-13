import pickle
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

DEFAULT_LGB_PARAMS = {
    'learning_rate': 0.1,
    'objective': 'regression',
    'seed': 1234,
}

DEFAULT_LR_PARAMS = {
    'random_state': 1234,
}

DEFAULT_KNN_PARAMS = {
    'n_neighbors': 5,  # default
}

DEFAULT_SVC_PARAMS = {
    'probability': True,
    'random_state': 1234,
}


class LGBClassifier(object):
    def __init__(self, params=DEFAULT_LGB_PARAMS):
        self.model = None
        self.params = params
        # self.cat_cols = None

    def train(self, X_train, y_train, X_valid, y_valid, save_path=None):
        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_valid, y_valid)
        self.model = lgb.train(self.params, train_set, num_boost_round=1000, early_stopping_rounds=10,
                               valid_sets=[train_set, val_set], verbose_eval=False)

        # save trained model
        if save_path is not None:
            self.save(save_path)

        return self.model.predict(X_valid)

    def save(self, file_path):
        self.model.save(file_path)
        return True

    def predict(self, X_test):
        test_preds = self.model.predict(X_test)
        return test_preds


class LRClassifier(object):
    def __init__(self, params=DEFAULT_LR_PARAMS):
        self.model = None
        self.params = params

    def train(self, X_train, y_train, X_valid=None, y_valid=None, save_path=None):
        self.model = LogisticRegression(**self.params)
        self.model.fit(X_train, y_train)

        # save trained model
        if save_path is not None:
            self.save(save_path)

        return True

    def save(self, file_path):
        pickle.dump(self.model, open(file_path, 'wb'))
        return True

    def predict(self, X_test):
        test_preds = self.model.predict_proba(X_test)
        return test_preds


class KNNClassifier(object):
    def __init__(self, params=DEFAULT_KNN_PARAMS):
        self.model = None
        self.params = params

    def train(self, X_train, y_train, X_valid=None, y_valid=None, save_path=None):
        self.model = KNeighborsClassifier(**self.params)
        self.model.fit(X_train, y_train)

        # save trained model
        if save_path is not None:
            self.save(save_path)

        return True

    def save(self, file_path):
        pickle.dump(self.model, open(file_path, 'wb'))
        return True

    def predict(self, X_test):
        test_preds = self.model.predict_proba(X_test)
        return test_preds


class SVCClassifier(object):
    def __init__(self, params=DEFAULT_SVC_PARAMS):
        self.model = None
        self.params = params

    def train(self, X_train, y_train, X_valid=None, y_valid=None, save_path=None):
        self.model = SVC(**self.params)
        self.model.fit(X_train, y_train)

        # save trained model
        if save_path is not None:
            self.save(save_path)

        return True

    def save(self, file_path):
        pickle.dump(self.model, open(file_path, 'wb'))
        return True

    def predict(self, X_test):
        test_preds = self.model.predict_proba(X_test)
        return test_preds
