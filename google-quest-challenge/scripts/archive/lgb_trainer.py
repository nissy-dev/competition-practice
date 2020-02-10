import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold


# lgb can't predict multi label
def lgb_group_kfold_trainer(train, test, features, target):
    # data setup
    X_train = train[features]
    Y_train = train[target]
    X_test = test[features]

    # initialize
    num_folds = 5
    valid_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    kf = GroupKFold(n_splits=num_folds)
    for fold, (train_index, valid_index) in enumerate(
            kf.split(X=train.question_body, groups=train.question_body)):
        print('Fold {}'.format(fold + 1))
        x_trn, x_val = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_trn, y_val = Y_train.iloc[train_index], Y_train.iloc[valid_index]
        train_set = lgb.Dataset(x_trn, y_trn)
        val_set = lgb.Dataset(x_val, y_val)

        params = {
            'learning_rate': 0.1,
            'metric': 'rmse',
            'objective': 'regression',
            'feature_fraction': 0.85,
            'subsample': 0.85,
            'n_jobs': -1,
            'seed': 1234,
            'max_depth': -1
        }

        # train the model with early stoping
        model = lgb.train(params, train_set, num_boost_round=10000, early_stopping_rounds=10,
                          valid_sets=[train_set, val_set], verbose_eval=10)
        valid_preds[valid_index] = model.predict(x_val)
        test_preds += model.predict(X_test) / num_folds

    # calc score
    score = spearmanr(valid_preds, Y_train.values).correlation
    print(target, score)
    return test_preds, score
