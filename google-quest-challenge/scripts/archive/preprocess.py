import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

ENG_STOPWORDS = set(stopwords.words("english"))


def decontract(text):
    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)
    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)
    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)
    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)
    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)
    text = re.sub(r"(A|a)isn(\'|\’)t ", "is not ", text)
    text = re.sub(r"n(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)re ", " are ", text)
    text = re.sub(r"(\'|\’)d ", " would ", text)
    text = re.sub(r"(\'|\’)ll ", " will ", text)
    text = re.sub(r"(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)ve ", " have ", text)
    text = re.sub(r"(L|l)et(\'|\’)s", "let us", text)
    text = re.sub(r"theres", "there is", text)
    return text


def clean_text(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x


def clean_number(x):
    x = re.sub('[0-9]{5,}', '12345', x)
    x = re.sub('[0-9]{4}', '1234', x)
    x = re.sub('[0-9]{3}', '123', x)
    x = re.sub('[0-9]{2}', '12', x)
    return x


def clean_html_tag(x):
    bs = BeautifulSoup(x)
    x = bs.get_text()
    return x


# def remove_stop_words(x, stop_words=ENG_STOPWORDS):
#     x = [word for word in x if word not in stop_words]
#     return x


def text_preprocess(x):
    x = decontract(x)
    x = clean_html_tag(x)
    x = clean_text(x)
    x = clean_number(x)
    return x


def preprocess(train, test):
    # text preprocessing
    text_columns = ['question_title', 'question_body', 'answer']
    for text_col in text_columns:
        train[text_col] = train[text_col].apply(lambda x: text_preprocess(x))
        test[text_col] = test[text_col].apply(lambda x: text_preprocess(x))

    # get target columns
    target_columns = list(set(train.columns) - set(test.columns))
    return train, test, target_columns
