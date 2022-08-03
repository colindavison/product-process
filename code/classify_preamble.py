import functools, operator, json, re, string, os, csv, urllib.parse, nltk, math, multiprocessing, concurrent.futures
from statistics import mean, pstdev, stdev
from collections import Counter
import pandas as pd
import numpy as np
from numpy import array
from random import sample
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2, SelectKBest, SelectFromModel
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedKFold
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef, \
balanced_accuracy_score
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.base import BaseEstimator, TransformerMixin
pd.set_option('mode.chained_assignment', None)

root = '//afs//crc.nd.edu//user//c//cdaviso1//pvp_git'
directory_initial = root + '//data//'
classifier_list = [ComplementNB(), MultinomialNB(), PassiveAggressiveClassifier(random_state=5)]
feature_select_list = ['no', 'yes']
classify = 'process'
predict_on = 'yes'
print_on = 'no'
final_prediction_on = 'yes'

df_sic_list = pd.read_csv(root + '//data//sic_list.csv').dropna(subset = ['request_extra_sic'])
df_sic_list['extra_sic'] = df_sic_list['extra_sic'].apply(lambda x: str(x).split(', '))
df_sic_list_full = pd.read_csv(root + '//data//sic_list.csv')

#create sic code lists
no_extra_sic = list(df_sic_list['sic'][df_sic_list['request_extra_sic'] == 0])
extra_sic = list(df_sic_list['sic'][df_sic_list['request_extra_sic'] == 1])
sic_code_list = [no_extra_sic, extra_sic]

extra_sic_codes = list(df_sic_list['extra_sic'][df_sic_list['request_extra_sic'] == 1])
extra_sic_codes = [[int(float(x)) for x in i] for i in extra_sic_codes]

class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        print(X.shape)
        return X

    def fit(self, X, y=None, **fit_params):
        return self


class ColumnExtractor(BaseEstimator, TransformerMixin):
  def __init__(self, feature_name):
    self.feature_name = feature_name

  def fit(self, X, y = None):
    return self

  def transform(self, X, y = None):
    X_ = X.copy() # creating a copy to avoid changes to original dataset
    X_ = X_[self.feature_name]
    return X_

class ColumnExtractorToDict(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, X, y = None):
    return self

  def transform(self, X, y = None):
    X_ = X.copy() # creating a copy to avoid changes to original dataset
    df = pd.DataFrame(list(X_))
    return df.to_dict(orient='records')
