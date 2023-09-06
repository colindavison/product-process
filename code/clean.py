#!/usr/bin/env python
# coding: utf-8

# In[1]:


import functools, operator, json, re, string, os, csv, urllib.parse, nltk, math, concurrent.futures, zipfile
from os import path
import pandas as pd
import numpy as np
import spacy
import en_core_web_sm
from random import sample
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier
from textblob import TextBlob

nlp = en_core_web_sm.load(disable=['parser', 'ner'])
table = str.maketrans('', '', string.punctuation)
suffix = ''

df_sic_list = pd.read_csv(root + '/data/sic_list.csv').dropna(subset = ['request_extra_sic'])
sic_list = list(df_sic_list['sic'])
sic_list = [int(float(x)) for x in sic_list]

root = '/Users/cdavison/Library/CloudStorage/OneDrive-TheCollegeofWooster/research/product-process/'
directory = root + 'data/'
os.chdir(directory)

def clean(sic):

    import functools, operator, json, re, string, os, csv, urllib.parse, nltk, math, concurrent.futures, spacy, en_core_web_sm, zipfile
    from os import path
    import pandas as pd
    import numpy as np
    from random import sample
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from bs4 import BeautifulSoup
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.feature_selection import chi2, SelectKBest
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier
    from textblob import TextBlob

    root = '//afs//crc.nd.edu//user//c//cdaviso1//pvp_git//'
    directory = root + 'data//' + str(sic) + '//'
    os.chdir(directory)
    data = pd.read_csv(str(sic) + '_discern.csv', index_col = ['publn_claim_id'], low_memory = False)

    #clean title
    data['title_clean'] = data['title'].apply(lambda x: word_tokenize(x.lower().strip()))
    data['title_clean'] = data['title_clean'].apply(lambda x: [word.translate(table) for word in x])
    data['title_clean'] = data['title_clean'].apply(lambda x: [re.sub(r'.*\d.*', '', word) for word in x])
    data['title_clean'] = data['title_clean'].apply(lambda x: [word for word in x if not word == ''])

    data['title_lem'] = data['title_clean'].apply(lambda x: [word.lemma_ for word in nlp(' '.join(x))])
    data['stopword_indices_title'] = data['title_clean'].apply(lambda x: [i for i, word in enumerate(x) if word in stopwords.words('english')])
    title_lem = list(data['title_lem'])
    stopword_indices_title = list(data['stopword_indices_title'])
    title_lem = [[word for i, word in enumerate(title) if not word == ' ' and not i in stopword_indices_title[ip]] \
                 for ip, title in enumerate(title_lem)]
    data['title_lem'] = title_lem
    data['title_clean'] = data['title_clean'].apply(lambda x: [word for word in x if not word in stopwords.words('english')])
    data['title_stem'] = data['title_clean'].apply(lambda x: [PorterStemmer().stem(word) for word in x])

    data['title_stem'] = data['title_stem'].apply(lambda x: ' '.join(x))
    data['title_lem'] = data['title_lem'].apply(lambda x: ' '.join(x))
    data['title_clean'] = data['title_clean'].apply(lambda x: ' '.join(x))

    columns_to_clean = ['publn_claims', 'publn_claims_abbr']

    for icol, col in enumerate(columns_to_clean):
        #lowercase, remove whitespace, punctuation, and digits
        data[col + '_clean'] = data[col].apply(lambda x: word_tokenize(x.lower().strip()))
        data[col + '_clean'] = data[col + '_clean'].apply(lambda x: [word.translate(table) for word in x])
        data[col + '_clean'] = data[col + '_clean'].apply(lambda x: [re.sub(r'.*\d.*', '', word) for word in x])
        data[col + '_clean'] = data[col + '_clean'].apply(lambda x: [word for word in x if not word == ''])

        #count stopwords and length of this partially cleaned text
        data[col + '_stopword_count'] = data[col + '_clean'].apply(lambda x: len([word for word in x if word                                                                                      in stopwords.words('english')]))
        data[col + '_count'] = data[col + '_clean'].apply(lambda x: len([word for word in x]))
        data[col + '_stopword_frac'] = .05*round(data[col + '_stopword_count'].divide(data[col + '_count'])/.05)
        data[col + '_count'] = data[col + '_count'].apply(lambda x: 15*round(int(x)/15))
        data[col + '_stopword_frac'] = data[col + '_stopword_frac'].apply(lambda x: x if not math.isinf(x) else np.nan)

        #lemmatize
        data[col + '_lem'] = data[col + '_clean'].apply(lambda x: [word.lemma_ for word in nlp(' '.join(x))])
        data['stopword_indices'] = data[col + '_clean'].apply(lambda x: [i for i, word in enumerate(x) if word in stopwords.words('english')])

        publn_claims_lem = list(data[col + '_lem'])
        stopword_indices = list(data['stopword_indices'])
        publn_claims_lem = [[word for i, word in enumerate(publn_claim) if not word == ' ' and not i in stopword_indices[ip]]                                 for ip, publn_claim in enumerate(publn_claims_lem)]

        data[col + '_lem'] = publn_claims_lem

        # remove stopwords, and stem
        data[col + '_clean'] = data[col + '_clean'].apply(lambda x: [word for word in x if not word in stopwords.words('english')])

        data[col + '_stem'] = data[col + '_clean'].apply(lambda x: [PorterStemmer().stem(word) for word in x])

        #create numerical features
        data[col + '_clean_avg_word_length'] = data[col + '_clean'].apply(lambda x:                                                                             .25*round(np.mean([len(word) for word in x])/.25)                                                                               if len([word for word in x]) else np.nan)

        pos_dic = {
            'noun' : ['NN','NNS','NNP','NNPS'],
            'pron' : ['PRP','PRP$','WP','WP$'],
            'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
            'adj' :  ['JJ','JJR','JJS'],
            'adv' : ['RB','RBR','RBS','WRB']
        }

        # function to check and get the part of speech tag count of a words in a given sentence
        def pos_check(x, flag):
            cnt = 0
            try:
                wiki = TextBlob(x)
                for tup in wiki.tags:
                    ppo = list(tup)[1]
                    if ppo in pos_dic[flag]:
                        cnt += 1
            except:
                pass
            return cnt

        data[col + '_clean_noun_frac'] = data[col + '_clean'].apply(lambda x: 0.03*round(pos_check(' '.join(x), 'noun')                                                                         /(len([word for word in x]))/0.03)                                                                        if len([word for word in x]) else np.nan)
        data[col + '_clean_verb_frac'] = data[col + '_clean'].apply(lambda x: 0.03*round(pos_check(' '.join(x), 'verb')                                                                         /(len([word for word in x]))/0.03)                                                                        if len([word for word in x]) else np.nan)
        data[col + '_clean_adj_frac'] = data[col + '_clean'].apply(lambda x: 0.03*round(pos_check(' '.join(x), 'adj')                                                                         /(len([word for word in x]))/0.03)                                                                        if len([word for word in x]) else np.nan)
        data[col + '_clean_adv_frac'] = data[col + '_clean'].apply(lambda x: 0.03*round(pos_check(' '.join(x), 'adv')                                                                         /(len([word for word in x]))/0.03)                                                                        if len([word for word in x]) else np.nan)

    #create alternate publication claim
    features = ['_clean', '_count', '_stopword_frac', '_lem', '_stem', '_clean_avg_word_length', '_clean_noun_frac', '_clean_verb_frac', '_clean_adj_frac', '_clean_adv_frac']

    data['alt_condition'] = data['publn_claims_abbr_clean'].apply(lambda x: len([word for word in x]))
    for feature in features:
        data['publn_claims_alt' + feature] = np.where(data['alt_condition'] <= 3,                                                             data['publn_claims' + feature], data['publn_claims_abbr' + feature])

        #create first n words
        for z in ['publn_claims']:
            for y in ['_clean', '_lem', '_stem']:
                for i in range(2, 11):
                    data['first_' + str(i) + '_words' + y] = data[z + y].apply(lambda x: x[0:i])

    #interactions for text features
    text_abbreviations = ['publn_claims', 'publn_claims_abbr', 'publn_claims_alt', 'first_2_words', 'first_3_words', \
                          'first_4_words', 'first_5_words', 'first_6_words', 'first_7_words', 'first_8_words', 'first_9_words',                              'first_10_words']
    text_features = ['_clean', '_lem', '_stem']
    interactions = ['sic', 'gvkey', 'cpc4', 'cpc3']

    for a in text_abbreviations:
        for f in text_features:
            for i in interactions:
                text = list(data[a + f])
                text_interact = list(data[i])
                text = [[word + str(text_interact[ip]) for iw, word in enumerate(publn_claim)] for ip, publn_claim in enumerate(text)]
                data[a + f + '_' + i] = text

    #interactions for numerical features
    text_abbreviations = ['publn_claims']
    numerical_features = ['_count']

    for a in text_abbreviations:
        for f in numerical_features:
            for i in interactions:
                data[a + f + '_' + i] = data[a + f].astype(str) + '_' + data[i].astype(str)

    text_abbreviations = ['publn_claims', 'publn_claims_abbr', 'publn_claims_alt']
    numerical_features = ['_stopword_frac', '_avg_word_length', '_noun_frac', '_verb_frac', '_adj_frac', '_adv_frac']

    for a in text_abbreviations:
        for f in numerical_features:
            for ii, i in enumerate(interactions):

                if f != '_stopword_frac' and ii == 0:
                    f = '_clean' + f

                data[a + f + '_' + i] = data[a + f].round(2).astype(str) + '_' + data[i].astype(str)

    #join together text lists
    text_abbreviations = ['publn_claims', 'publn_claims_abbr', 'publn_claims_alt', 'first_2_words', 'first_3_words', \
                          'first_4_words', 'first_5_words', 'first_6_words', 'first_7_words', 'first_8_words', 'first_9_words',                              'first_10_words']
    text_features = ['_clean', '_lem', '_stem']
    interactions = ['', '_sic', '_gvkey', '_cpc4', '_cpc3']

    for a in text_abbreviations:
        for f in text_features:
            for i in interactions:
                try:
                    data[a + f + i] = data[a + f + i].apply(lambda x: ' '.join(x))
                except:
                    continue

    data = data.drop(['stopword_indices', 'stopword_indices_title'], axis=1)
    data.to_csv(str(sic) + '_discern.csv')

    #zip the file
    with zipfile.ZipFile(str(sic) + '_discern.zip', 'w', compression = zipfile.ZIP_DEFLATED) as my_zip:
        my_zip.write(str(sic) + '_discern.csv')
    os.remove(str(sic) + '_discern.csv')


if __name__ ==  '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(clean, sic) for sic in sic_list]

        for f in concurrent.futures.as_completed(results):
            print(f.result())
