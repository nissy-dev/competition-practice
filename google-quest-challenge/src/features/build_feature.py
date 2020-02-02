import string
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelBinarizer
from src.features.paragraph_vectorizer import BoWVectorizerWithSVD, TfIdfVectorizerWithSVD, \
    Doc2VecVectorizer, TfidfEmbeddingVectorizer, SWEMEmbeddingVectorizer


ENG_STOPWORDS = set(stopwords.words("english"))


def char_count(s):
    return len(s)


def word_count(s):
    return s.count(' ')


def stopwords_count(s):
    return len([w for w in str(s).lower().split() if w in ENG_STOPWORDS])


def punctuations_count(s):
    return len([c for c in str(s) if c in string.punctuation])


def words_upper_count(s):
    return len([w for w in str(s).split() if w.isupper()])


def calc_lexical_diversity(s):
    return len(set(s.split())) / len(s.split())


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


def build_feature(train, test):
    # feature engineering
    # common processing
    print('Common processing....')
    for tmp_df in [train, test]:
        text_columns = ['question_title', 'question_body', 'answer']
        # from question_title, question_body, answer
        for col in text_columns:
            tmp_df[col + '_n_chars'] = tmp_df[col].apply(char_count)
            tmp_df[col + '_n_words'] = tmp_df[col].apply(word_count)
            tmp_df[col + '_num_stopwords'] = tmp_df[col].apply(stopwords_count)
            tmp_df[col + '_num_punctuations'] = tmp_df[col].apply(punctuations_count)
            tmp_df[col + '_num_upperwords'] = tmp_df[col].apply(words_upper_count)

        # check lexical diversity (unique words count vs total)
        tmp_df['answer_word_diversity'] = tmp_df['answer'].apply(calc_lexical_diversity)
        # check for nonames, i.e. users with logins like user12389
        tmp_df['is_question_no_name_user'] = tmp_df['question_user_name'].str.contains('^user\d+$') + 0
        tmp_df['is_answer_no_name_user'] = tmp_df['answer_user_name'].str.contains('^user\d+$') + 0

        # count word_overlap between question_body and answer
        tmp_df['q_words'] = tmp_df['question_body'].apply(
            lambda s: [f for f in s.split() if f not in ENG_STOPWORDS])
        tmp_df['a_words'] = tmp_df['answer'].apply(
            lambda s: [f for f in s.split() if f not in ENG_STOPWORDS])
        tmp_df['qa_word_overlap'] = tmp_df.apply(
            lambda s: len(np.intersect1d(s['q_words'], s['a_words'])), axis=1)
        tmp_df['qa_word_overlap_norm1'] = tmp_df.apply(
            lambda s: s['qa_word_overlap']/(1 + len(s['a_words'])), axis=1)
        tmp_df['qa_word_overlap_norm2'] = tmp_df.apply(
            lambda s: s['qa_word_overlap']/(1 + len(s['q_words'])), axis=1)
        tmp_df.drop(['q_words', 'a_words'], axis=1, inplace=True)

        # from url (question_user_page, answer_user_page, host)
        tmp_df['domcom'] = tmp_df['host'].apply(lambda s: s.split('.'))
        tmp_df['dom_cnt'] = tmp_df['domcom'].apply(lambda s: len(s))
        tmp_df['domcom'] = tmp_df['domcom'].apply(lambda s: s + ['none'])
        for ii in range(0, 3):
            tmp_df['dom_' + str(ii)] = tmp_df['domcom'].apply(lambda s: s[ii])
        tmp_df.drop('domcom', axis=1, inplace=True)

    print('Paragraph converting....')
    # not common processing
    # paragraph convert
    converters = [
        BoWVectorizerWithSVD(dims=50),
        TfIdfVectorizerWithSVD(dims=50),
        Doc2VecVectorizer(dims=100),
        # TfidfEmbeddingVectorizer(dims=100),
        # SWEMEmbeddingVectorizer(dims=100, pooling='mean')
    ]
    for converter in converters:
        for col in text_columns:
            train_converted_df = converter.fit_transform(train[col].values)
            test_converted_df = converter.transform(test[col].values)
            train_converted_df.add_prefix(col + '_')
            test_converted_df.add_prefix(col + '_')
            train = pd.concat([train, train_converted_df], axis=1)
            test = pd.concat([test, test_converted_df], axis=1)

    print('Categorical columns converting....')
    # categorical_columns
    # one-hot encoding
    one_hot_columns = ['dom_0', 'dom_1', 'dom_2', 'category']
    for col in one_hot_columns:
        train, test = one_hot_encoding(train, test, col)

    # count encoding
    count_columns = ['question_user_name', 'answer_user_name']
    for col in count_columns:
        train, test = count_encoding(train, test, col)

    # unused columns
    unused_columns = text_columns + one_hot_columns + count_columns + \
        ['host', 'answer_user_page', 'question_user_page', 'url']

    # check total features
    print('train shape : ', train.shape)
    print('test shape : ', test.shape)
    assert train.shape[1] - 30 == test.shape[1]

    return train, test, unused_columns
