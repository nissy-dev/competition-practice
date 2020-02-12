import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


def one_hot_encoding(train, test, col):
    lb = LabelBinarizer()
    train_onehot = lb.fit_transform(train[col])
    test_onehot = lb.transform(test[col])
    columns = [col + '_' + val for val in lb.classes_]
    train_converted_df = pd.DataFrame(data=train_onehot, columns=columns)
    test_converted_df = pd.DataFrame(data=test_onehot, columns=columns)
    train = pd.concat([train, train_converted_df], axis=1)
    test = pd.concat([test, test_converted_df], axis=1)
    return train, test


def count_encoding(train, test, col):
    counter = Counter(train[col].values)
    count_dict = dict(counter.most_common())
    keys = count_dict.keys()
    label_count_dict = {key: i for i, key in enumerate(keys, start=1)}
    train['cnt_enc_' + col] = train[col].map(lambda x: label_count_dict[x]).values
    test['cnt_enc_' + col] = test[col].map(
        lambda x: label_count_dict[x] if x in keys else 0).values
    return train, test


def label_encoding(train, test, col):
    # データが小さい時はコピーした方が安全
    # train = train.copy()
    # test = test.copy()
    le = LabelEncoder()
    train['label_enc_' + col] = le.fit_transform(train[col])
    test['label_enc_' + col] = le.transform(test[col])
    return train, test
