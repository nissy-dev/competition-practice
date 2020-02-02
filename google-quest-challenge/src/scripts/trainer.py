import numpy as np
from src.data.read_data import read_data
from src.features.preprocess import preprocess
from src.features.build_feature import build_feature
from src.scripts.lgb_trainer import lgb_model_with_target
from src.scripts.nn_trainer import nn_group_kfold_trainer


def main():
    BASE_PATH = '../input/google-quest-challenge/'
    train, test, sample_submission = read_data(BASE_PATH)
    train, test, target_columns = preprocess(train, test)
    train, test, unused_columns = build_feature(train, test)

    # data setup
    case = 'test'
    if case == 'lgb':
        features = [col for col in test.columns if col not in (unused_columns + ['qa_id'])]
        test_preds = []
        full_score = []
        for num, target in enumerate(target_columns):
            print('Train model {}'.format(num + 1))
            print('Predicting target {}'.format(target))
            # train
            preds, score = lgb_model_with_target(train, test, features, target)
            test_preds.append(np.clip(preds, a_min=0, a_max=1))
            full_score.append(score)
        print('CV score : ', np.mean(full_score))

        # submit
        sample_submission[target_columns] = np.array(test_preds).T
        sample_submission.to_csv('submission.csv', index=False)
    elif case == 'nn':
        # not good...
        train = train.fillna(0)
        test = test.fillna(0)
        features = [col for col in test.columns if col not in (unused_columns + ['qa_id'])]
        # train
        fold_scores, test_preds = nn_group_kfold_trainer(train, test, features, target_columns)
        # submit
        print('CV score : ', np.mean(fold_scores))
        sample_submission[target_columns] = np.array(test_preds)
        sample_submission.to_csv('submission.csv', index=False)
