import nltk
import warnings
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('punkt')
warnings.filterwarnings("ignore")


class BoWVectorizerWithSVD(object):
    def __init__(self, dims=50, random_state=1234):
        self.count_vectorizer = None
        self.tsvd = None
        self.dims = dims
        self.random_state = random_state

    def fit_transform(self, X):
        # create BoW vector
        self.count_vectorizer = CountVectorizer()
        bow_X = self.count_vectorizer.fit_transform(X)
        # reduce vector dimension
        self.tsvd = TruncatedSVD(n_components=self.dims, random_state=self.random_state)
        truncated_bow_X = self.tsvd.fit_transform(bow_X)

        df = pd.DataFrame(data=truncated_bow_X,
                          columns=['BoW-WithSVD-' + str(i) for i in range(self.dims)])
        return df

    def transform(self, X):
        bow_X = self.count_vectorizer.transform(X)
        truncated_bow_X = self.tsvd.transform(bow_X)
        df = pd.DataFrame(data=truncated_bow_X,
                          columns=['BoW-WithSVD-' + str(i) for i in range(self.dims)])
        return df


class TfIdfVectorizerWithSVD(object):
    def __init__(self, dims=50, random_state=1234):
        self.tfidf_vectorizer = None
        self.tsvd = None
        self.dims = dims
        self.random_state = random_state

    def fit_transform(self, X):
        # create tf-idf vector
        self.tfidf_vectorizer = TfidfVectorizer()
        tfidf_X = self.tfidf_vectorizer.fit_transform(X)
        # reduce vector dimension
        self.tsvd = TruncatedSVD(n_components=self.dims, random_state=self.random_state)
        truncated_tfidf_X = self.tsvd.fit_transform(tfidf_X)

        df = pd.DataFrame(data=truncated_tfidf_X,
                          columns=['tf-idf-WithSVD-' + str(i) for i in range(self.dims)])
        return df

    def transform(self, X):
        tfidf_X = self.tfidf_vectorizer.transform(X)
        truncated_tfidf_X = self.tsvd.transform(tfidf_X)
        df = pd.DataFrame(data=truncated_tfidf_X,
                          columns=['tf-idf-WithSVD-' + str(i) for i in range(self.dims)])
        return df


class Doc2VecVectorizer(object):
    def __init__(self, dims=100):
        self.model = None
        self.dims = dims

    def fit_transform(self, X):
        tagged_data = [TaggedDocument(words=nltk.word_tokenize(_d.lower()), tags=[str(i)])
                       for i, _d in enumerate(X)]
        self.model = Doc2Vec(tagged_data, vector_size=self.dims)
        feature = np.array([self.model.docvecs[i] for i in range(len(X))])
        df = pd.DataFrame(data=feature,
                          columns=['doc2vec-' + str(i) for i in range(self.dims)])

        return df

    def transform(self, X):
        words_list = [nltk.word_tokenize(sentence.lower()) for sentence in X]
        feature = np.array([self.model.infer_vector(words) for words in words_list])
        df = pd.DataFrame(data=feature,
                          columns=['doc2vec-' + str(i) for i in range(self.dims)])
        return df


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vector=None, dims=100, use_word2vec=False):
        # receive pre-trained model
        self.word2vector = word2vector
        self.word2weight = None
        self.dims = dims
        self.use_word2vec = use_word2vec

    def fit_transform(self, X):
        # calculate tf-idf
        tfidf = TfidfVectorizer(tokenizer=lambda x: nltk.word_tokenize(x))
        tfidf.fit(X)
        # create word2weight dict
        self.word2weight = {}
        for w, i in tfidf.vocabulary_.items():
            self.word2weight[w] = tfidf.idf_[i]

        # create word2vec dict
        words_list = [nltk.word_tokenize(sentence) for sentence in X]
        if self.word2vector is None:
            model = Word2Vec(words_list, size=self.dims, seed=1234) \
                if self.use_word2vec else FastText(words_list, size=self.dims, seed=1234)
            self.word2vector = dict(zip(model.wv.index2word, model.wv.syn0))

        feature = np.array([self.get_sentence_vector(words) for words in words_list])
        df = pd.DataFrame(data=feature,
                          columns=['Tfidf-with-wordvec' + str(i) for i in range(feature.shape[1])])

        return df

    def transform(self, X):
        words_list = [nltk.word_tokenize(sentence) for sentence in X]
        feature = np.array([self.get_sentence_vector(words) for words in words_list])
        df = pd.DataFrame(data=feature,
                          columns=['Tfidf-with-wordvec' + str(i) for i in range(feature.shape[1])])
        return df

    def get_sentence_vector(self, words):
        dims = len(list(self.word2vector.values())[0])
        vectors = np.zeros((len(words), dims))
        total_weight = 0
        for i, word in enumerate(words):
            try:
                vectors[i] = self.word2vector[word] * self.word2weight[word]
                total_weight += self.word2weight[word]
            except:  # noqa
                # doesn't find the vector
                vectors[i] = np.random.normal(loc=0, scale=1, size=dims)

        return np.sum(vectors, axis=0) / total_weight


class SWEMEmbeddingVectorizer(object):
    def __init__(self, word2vector=None, dims=100, use_word2vec=False, pooling='mean'):
        # receive pre-trained model
        self.word2vector = word2vector
        self.word2weight = None
        self.dims = dims
        self.use_word2vec = use_word2vec
        self.pooling = pooling

    def fit_transform(self, X):
        # create word2vec dict
        words_list = [nltk.word_tokenize(sentence) for sentence in X]
        if self.word2vector is None:
            model = Word2Vec(words_list, size=self.dims, seed=1234) \
                if self.use_word2vec else FastText(words_list, size=self.dims, seed=1234)
            self.word2vector = dict(zip(model.wv.index2word, model.wv.syn0))

        feature = np.array([self.get_sentence_vector(words) for words in words_list])
        df = pd.DataFrame(data=feature,
                          columns=['SWEM-' + str(i) for i in range(feature.shape[1])])

        return df

    def transform(self, X):
        words_list = [nltk.word_tokenize(sentence) for sentence in X]
        feature = np.array([self.get_sentence_vector(words) for words in words_list])
        df = pd.DataFrame(data=feature,
                          columns=['SWEM-' + str(i) for i in range(feature.shape[1])])
        return df

    def get_sentence_vector(self, words):
        dims = len(list(self.word2vector.values())[0])
        vectors = np.zeros((len(words), dims))
        for i, word in enumerate(words):
            try:
                vectors[i] = self.word2vector[word]
            except:  # noqa
                # doesn't find the vector
                vectors[i] = np.random.normal(loc=0, scale=1, size=dims)

        # only max, min, mean pooling
        if self.pooling == 'max':
            return np.max(vectors, axis=0)
        elif self.pooling == 'min':
            return np.min(vectors, axis=0)
        elif self.pooling == 'mean':
            return np.mean(vectors, axis=0)
        else:
            return np.mean(vectors, axis=0)
