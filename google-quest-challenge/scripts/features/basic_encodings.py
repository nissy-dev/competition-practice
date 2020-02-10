import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


def one_hot_encoding(train, test, col):
    tmp_train = train.copy()
    tmp_test = test.copy()
    lb = LabelBinarizer()
    train_onehot = lb.fit_transform(tmp_train[col])
    test_onehot = lb.transform(tmp_test[col])
    columns = [col + '_' + val for val in lb.classes_]
    train_converted_df = pd.DataFrame(data=train_onehot, columns=columns)
    test_converted_df = pd.DataFrame(data=test_onehot, columns=columns)
    tmp_train = pd.concat([tmp_train, train_converted_df], axis=1)
    tmp_test = pd.concat([tmp_test, test_converted_df], axis=1)
    return tmp_train, tmp_test


def count_encoding(train, test, col):
    tmp_train = train.copy()
    tmp_test = test.copy()
    counter = Counter(train[col].values)
    count_dict = dict(counter.most_common())
    keys = count_dict.keys()
    label_count_dict = {key: i for i, key in enumerate(keys, start=1)}
    tmp_train['cnt_enc_' + col] = tmp_train[col].map(lambda x: label_count_dict[x]).values
    tmp_test['cnt_enc_' + col] = tmp_test[col].map(
        lambda x: label_count_dict[x] if x in keys else 0).values
    return tmp_train, tmp_test


def label_encoding(train, test, col):
    tmp_train = train.copy()
    tmp_test = test.copy()
    le = LabelEncoder()
    tmp_train['label_enc_' + col] = le.fit_transform(tmp_train[col])
    tmp_test['label_enc_' + col] = le.transform(tmp_test[col])
    return tmp_train, tmp_test
