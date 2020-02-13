import re
import nltk
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from constant import CONTRACTION_MAPPING, PUNCT_MAPPING


# Constant
ENG_STOPWORDS = set(stopwords.words("english"))
WORDNET_MAP = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

# Instance
SpellCheckerInstance = SpellChecker()
LemmatizerInstance = WordNetLemmatizer()


def clean_html_tag(text):
    """HTMLの削除"""
    return BeautifulSoup(text, "lxml").text


def clean_url(text):
    """URLの削除"""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def clean_number(text):
    """数字の削除"""
    return re.sub(r'\d +', '', text)


def upper_to_lower(text):
    """文字の小文字化"""
    return text.lower()


def clean_contractions(text, mapping=CONTRACTION_MAPPING):
    """短縮系の修正"""
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


def clean_punctuation(text, punct=string.punctuation):
    """特殊文字の削除"""
    return text.translate(str.maketrans('', '', punct))


def clean_special_chars(text, punct=string.punctuation, mapping=PUNCT_MAPPING):
    """特殊文字の置換"""
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    # 例外の置換処理
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])

    return text


def fix_miss_spellings(text):
    """typoの修正"""
    corrected_text = []
    misspelled_words = SpellCheckerInstance.unknown(text.split())

    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(SpellCheckerInstance.correction(word))
        else:
            corrected_text.append(word)

    return " ".join(corrected_text)


def clean_stopwords(text, stop_words=ENG_STOPWORDS):
    """stopwordの削除"""
    return " ".join([word for word in str(text).split() if word not in stop_words])


def lemmatize(text):
    """動詞や名詞の活用形の統一"""
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([LemmatizerInstance.lemmatize(word, WORDNET_MAP.get(pos[0], wordnet.NOUN))
                     for word, pos in pos_tagged_text])


def text_preprocess(x):
    x = clean_html_tag(x)
    x = clean_url(x)
    x = clean_number(x)
    x = upper_to_lower(x)
    x = clean_contractions(x)
    x = clean_punctuation(x)
    # x = clean_special_chars(x)
    x = fix_miss_spellings(x)
    # x = clean_stopwords(x)
    # x = lemmatize(x)
    return x

# def preprocess(df, text_columns):
#     # text preprocessing
#     # text_columns = ['question_title', 'question_body', 'answer']
#     for text_col in text_columns:
#         df[text_col] = df[text_col].apply(lambda x: text_preprocess(x))
#     return df
