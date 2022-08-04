#!/usr/bin/env python
# coding: utf-8

# In[1]:

import functools, operator, json, re, string, os, csv, urllib.parse, nltk, shutil, concurrent.futures, sys
import pandas as pd
import numpy as np
from random import sample
from os import path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier

sic = 2040
root = '//afs//crc.nd.edu//user//c//cdaviso1//pvp_git//'
directory = root + 'data//'
os.chdir(directory)

table = str.maketrans('', '', string.punctuation)

def ind_claims(tag):
    return tag.has_attr('class') and not tag.has_attr('num') and tag['class'] == ['claim']

def has_num(tag):
    return tag.has_attr('num')

def count_words(text):
    return len(text.split())

def first_words_process(x):
    return ' '.join(x.split()[0:2])

def google_patent_webscraper(sic):
    import functools, operator, json, re, string, os, csv, urllib.parse, nltk, shutil, concurrent.futures
    import pandas as pd
    import numpy as np
    from random import sample
    from os import path
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from bs4 import BeautifulSoup
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.feature_selection import chi2, SelectKBest
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier

    root = '//afs//crc.nd.edu//user//c//cdaviso1//pvp_git//'
    directory = root + 'data//'
    os.chdir(directory)

    df = pd.read_csv('patents.csv', low_memory = False)
    df_sic = df[df['sic'] == sic]
    patent_number_list = list(df_sic['patent'][df_sic['sic'] == sic])

    patent_number_list_reshape = []
    publn_claims_abbr_list = []
    publn_claims_list = []
    cpc_first_list_reshape = []
    cpc4_first_list_reshape = []
    cpc3_first_list_reshape = []
    cpc_first_desc_list_reshape = []
    cpc4_first_desc_list_reshape = []
    cpc3_first_desc_list_reshape = []
    publn_claims_sic_list = []
    publn_claims_sic_abbr_list = []
    publn_claims_list = []
    publn_claims_abbr_list = []
    publn_claims_name_list = []
    publn_claims_name_abbr_list = []
    publn_claims_nostem_list = []
    publn_claims_nostem_abbr_list = []
    adate_reshape = []
    ayear_reshape = []
    pdate_reshape = []
    pyear_reshape = []
    fcites_nonfam_reshape = []
    bcites_nonfam_reshape = []
    fcites_fam_reshape = []
    bcites_fam_reshape = []
    title_reshape = []

    for patent_number in patent_number_list:
        try:
            url = 'https://patents.google.com/patent/' + patent_number
            req = Request(url)
            webpage = urlopen(req).read()
            soup = BeautifulSoup(webpage, features="lxml")
            desc = soup.find('div', class_='description')
            title = soup.find(itemprop = 'title').text
            abstract = soup.find('meta',attrs={'name':'DC.description'})
            publn_claims = soup.find_all(ind_claims)
            cpc_first = soup.find(content = 'true', itemprop='Leaf').find_previous(itemprop='Code').text
            cpc_first_desc = soup.find(content = 'true', itemprop='Leaf').find_previous(itemprop='Description').text
            cpc4_first = cpc_first[0:4]
            cpc4_first_desc = soup.find(itemprop="Code", string=re.compile(cpc4_first)).find_next(itemprop='Description').text
            cpc3_first = cpc_first[0:3]
            cpc3_first_desc = soup.find(itemprop="Code", string=re.compile(cpc3_first)).find_next(itemprop='Description').text
            adate = soup.find('time', itemprop = 'filingDate').text
            ayear = adate[0:4]
            pdate = soup.find('time', itemprop = 'publicationDate').text
            pyear = pdate[0:4]

            fcites_nonfam = 0
            for item in soup.find_all(itemprop = 'forwardReferencesOrig'):
                fcites_nonfam += 1

            fcites_fam = 0
            for item in soup.find_all(itemprop = 'forwardReferencesFamily'):
                fcites_fam += 1

            bcites_nonfam = 0
            for item in soup.find_all(itemprop = 'backwardReferencesOrig'):
                bcites_nonfam += 1

            bcites_fam = 0
            for item in soup.find_all(itemprop = 'backwardReferencesFamily'):
                bcites_fam += 1

            for claim in publn_claims:
                patent_number_list_reshape.append(patent_number)

                adate_reshape.append(adate)
                pdate_reshape.append(pdate)
                ayear_reshape.append(ayear)
                pyear_reshape.append(pyear)

                cpc_first_list_reshape.append(cpc_first)
                cpc4_first_list_reshape.append(cpc4_first)
                cpc3_first_list_reshape.append(cpc3_first)

                cpc_first_desc_list_reshape.append(cpc_first_desc)
                cpc4_first_desc_list_reshape.append(cpc4_first_desc)
                cpc3_first_desc_list_reshape.append(cpc3_first_desc)

                fcites_nonfam_reshape.append(fcites_nonfam)
                bcites_nonfam_reshape.append(bcites_nonfam)
                fcites_fam_reshape.append(fcites_fam)
                bcites_fam_reshape.append(bcites_fam)
                title_reshape.append(title)

                publn_claims_list.append(claim.text.strip())

                try:
                    try:
                        claim_abbr = str(claim.find('div', class_='claim-text').find('div', class_='claim-text').previous_sibling)
                    except:
                        claim_abbr = str(claim.text.strip().partition(':')[0])
                except:
                    claim_abbr = ''

                publn_claims_abbr_list.append(claim_abbr)

        except:
            print(patent_number)
            continue

    data = pd.DataFrame(list(zip(patent_number_list_reshape, title_reshape, publn_claims_list, publn_claims_abbr_list, cpc_first_list_reshape, \
                                 cpc4_first_list_reshape, cpc3_first_list_reshape, cpc_first_desc_list_reshape,  \
                                 cpc4_first_desc_list_reshape, cpc3_first_desc_list_reshape, ayear_reshape, pyear_reshape, \
                                adate_reshape, pdate_reshape, fcites_nonfam_reshape, fcites_fam_reshape, bcites_nonfam_reshape, \
                                 bcites_fam_reshape)),  \
                        columns = ['patent', 'title', 'publn_claims', 'publn_claims_abbr', 'cpc', 'cpc4', \
                                   'cpc3', 'cpc_desc', 'cpc4_desc', 'cpc3_desc', 'ayear', 'pyear', \
                                   'adate', 'pdate', 'fcites_nonfam', 'fcites_fam', 'bcites_nonfam', 'bcites_fam'])

    data = df_sic.merge(data, on=['patent'], how='right')

    publn_claim_n_list = []
    publn_claims_list = []
    publn_claims_abbr_list = []
    patent_group = data.groupby(['patent'], sort=False, as_index=False)

    for x, group in patent_group:
        counter = 0
        for index, row in group.iterrows():
            counter += 1
            row['publn_claim_n'] = counter
            publn_claim_n_list.append(counter)

    data['publn_claim_n'] = publn_claim_n_list
    conm = data['conm']
    data.drop(labels=['conm'], axis=1,inplace = True)
    data.insert(0, 'conm', conm)
    data['publn_claim_id'] = data['patent'] + '_' + data['publn_claim_n'].apply(lambda x: str(x))
    data = data.set_index('publn_claim_id')
    directory_sic = directory + str(sic) + '//'

    if path.exists(directory_sic):
        os.chdir(directory_sic)
        data.to_csv(str(sic) + '_discern.csv')

    if not path.exists(directory_sic):
        os.mkdir(directory_sic)
        os.chdir(directory_sic)
        data.to_csv(str(sic) + '_discern.csv')

    data = pd.DataFrame()
    df1 = pd.DataFrame()
    patent_number_list_reshape = []
    publn_claims_abbr_list = []
    publn_claims_list = []
    cpc_first_list_reshape = []
    cpc4_first_list_reshape = []
    cpc3_first_list_reshape = []
    cpc_first_desc_list_reshape = []
    cpc4_first_desc_list_reshape = []
    cpc3_first_desc_list_reshape = []
    publn_claims_sic_list = []
    publn_claims_sic_abbr_list = []
    publn_claims_list = []
    publn_claims_abbr_list = []
    publn_claims_name_list = []
    publn_claims_name_abbr_list = []
    publn_claims_nostem_list = []
    publn_claims_nostem_abbr_list = []
    adate_reshape = []
    ayear_reshape = []
    pdate_reshape = []
    pyear_reshape = []
    fcites_nonfam_reshape = []
    bcites_nonfam_reshape = []
    fcites_fam_reshape = []
    bcites_fam_reshape = []
    title_reshape = []

google_patent_webscraper(sic)

