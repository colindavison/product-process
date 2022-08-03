import functools, operator, json, re, string, os, csv, urllib.parse, nltk, math, multiprocessing, concurrent.futures, os.path, zipfile
from os import path
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

from classify_preamble import ColumnExtractor, ColumnExtractorToDict, Debug, extra_sic_codes, sic_code_list, \
extra_sic, no_extra_sic, df_sic_list_full, df_sic_list, classifier_list, feature_select_list, classify, \
directory_initial, predict_on, print_on, final_prediction_on, root

os.chdir(root + '//code//')
def get_data(sic_code, directory_initial, classify, counter=0, match=0):
    directory = directory_initial + str(sic_code) + '//'
    
    if path.exists(directory):
        os.chdir(directory)
    else:
        pass

    if path.exists(str(sic_code) + '_discern_hand_classified.csv'):
        df_get_data = pd.read_csv(str(sic_code) + '_discern_hand_classified.csv', index_col='publn_claim_id', low_memory = False)

    else:
        df_get_data = pd.DataFrame()
    return df_get_data

def classify_sic(isic_code, sic_code, isic_list, sic_list, initial_success=0):
    import functools, operator, json, re, string, os, csv, urllib.parse, nltk, math, multiprocessing, concurrent.futures, os.path, zipfile
    from os import path
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

    #DATA COLLECTION
    if isic_list != 0:
        for iextra_sic, extra_sic in enumerate(extra_sic_codes[isic_code]):
            if iextra_sic == 0:
                sic_list = list(df_sic_list_full['sic'][df_sic_list_full['sic'].apply(lambda x: str(x)[:len(str(extra_sic))]) \
                                                        == str(extra_sic)])
                sic_list.append(sic_code)
            else:
                sic_list.extend(list(df_sic_list_full['sic'][df_sic_list_full['sic'].apply(lambda x: str(x)[:len(str(extra_sic))]) \
                                                        == str(extra_sic)]))

            sic_list = list(set(sic_list))
        for isic, sic in enumerate(sic_list):
            if get_data(sic, directory_initial, classify).empty:
                continue
            if initial_success == 0 and get_data(sic, directory_initial, classify).empty is False:
                data = get_data(sic, directory_initial, classify)
                initial_success += 1
            else:
                data = data.append(get_data(sic, directory_initial, classify))
        sic_list = [str(x) for x in sic_list]
    else:
        data = get_data(sic_code, directory_initial, classify)
        sic_list = [str(sic_code)]

    data = data.dropna(subset = ['publn_claims_clean', classify])
    print(data.shape)

    d = {}
    cleaning_variations = ['_clean', '_lem', '_stem']
    length_variations = ['', '_alt']
    interactions = ['', '_sic', '_gvkey', '_cpc4', '_cpc3']
    parts_of_speech = ['noun', 'verb', 'adj', 'adv']

    #COUNT VECTORIZER FEATURES
    for x in cleaning_variations:
        for y in length_variations:
            for z in interactions:
                d['publn_claims{0}'.format(y) + '{0}'.format(x) + '{0}'.format(z)] = \
                            [('publn_claims{0}'.format(y) + '{0}'.format(x) + '{0}'.format(z),
                            Pipeline([('extract', ColumnExtractor('publn_claims' + str(y) + str(x) + str(z))),
                            ('countvec', CountVectorizer(analyzer='word', ngram_range=(1, 3)))]))]

    #first 10 words
    for x in cleaning_variations:
        for i in range(2, 3):
            for z in interactions:
                d['first_{0}'.format(i) + '_words{0}'.format(x) + '{0}'.format(z)] = \
                            [('first_{0}'.format(i) + '_words{0}'.format(x) + '{0}'.format(z),
                            Pipeline([('extract', ColumnExtractor('first_' + str(i) + '_words' + str(x) + str(z))),
                            ('countvec', CountVectorizer(analyzer='word', ngram_range=(1, 2)))]))]

    for x in cleaning_variations:
        for i in range(3, 11):
            for z in interactions:
                d['first_{0}'.format(i) + '_words{0}'.format(x) + '{0}'.format(z)] = \
                            [('first_{0}'.format(i) + '_words{0}'.format(x) + '{0}'.format(z),
                            Pipeline([('extract', ColumnExtractor('first_' + str(i) + '_words' + str(x) + str(z))),
                            ('countvec', CountVectorizer(analyzer='word', ngram_range=(1, 3)))]))]


    for c in cleaning_variations:
        for i in interactions:
            d['f2_f10{0}'.format(c) + '{0}'.format(i)] = d['first_2_words{0}'.format(c) + '{0}'.format(i)] + \
                                                        d['first_3_words{0}'.format(c) + '{0}'.format(i)] + \
                                                        d['first_4_words{0}'.format(c) + '{0}'.format(i)] + \
                                                        d['first_5_words{0}'.format(c) + '{0}'.format(i)] + \
                                                        d['first_6_words{0}'.format(c) + '{0}'.format(i)] + \
                                                        d['first_7_words{0}'.format(c) + '{0}'.format(i)] + \
                                                        d['first_8_words{0}'.format(c) + '{0}'.format(i)] + \
                                                        d['first_9_words{0}'.format(c) + '{0}'.format(i)] + \
                                                        d['first_10_words{0}'.format(c) + '{0}'.format(i)]



    # NUMERIC FEATURES
    #parts of speech
    for y in length_variations:
        for z in interactions:
            for pos in parts_of_speech:
                d['publn_claims{0}'.format(y) + '_clean_{0}_frac'.format(pos) + '{0}'.format(z)] = \
                                [('publn_claims{0}'.format(y) + '_clean_{0}_frac'.format(pos) + '{0}'.format(z),
                                Pipeline([('extract', ColumnExtractor('publn_claims' + str(y) + '_clean_{0}_frac'.format(pos) + str(z))),
                                    ('col_to_dict', ColumnExtractorToDict()),
                                    ('dict', DictVectorizer())]))]

    #number of words
    for x in ['_count', '_stopword_frac', '_clean_avg_word_length']:
        for y in length_variations:
            for z in interactions:
                d['publn_claims{0}'.format(y) + '{0}'.format(x) + '{0}'.format(z)] = \
                [('publn_claims{0}'.format(y) + '{0}'.format(x) + '{0}'.format(z),
                    Pipeline([('extract', ColumnExtractor('publn_claims{0}'.format(y) + '{0}'.format(x) + '{0}'.format(z))),
                        ('col_to_dict', ColumnExtractorToDict()),
                        ('dict', DictVectorizer())]))]


    #CATEGORICAL FEATURES
    d['cpc4'] = [('cpc4',
                  Pipeline([('extract', ColumnExtractor('cpc4')),
                        ('col_to_dict', ColumnExtractorToDict()),
                        ('dict', DictVectorizer())]))]

    pipe_short = [[[] for _ in range(len(feature_select_list))] for _ in range(len(classifier_list))]
    pipe_long = [[[] for _ in range(len(feature_select_list))] for _ in range(len(classifier_list))]
    for iclf, clf in enumerate(classifier_list):
        feat_select = ('feat_select', SelectFromModel(clf))
        for iselect, select in enumerate(feature_select_list):
            if select == 'yes':
                pipe_short[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['first_2_words_lem'] + d['cpc4'])), \
                                                                                      feat_select, ('clf', clf)]))
                pipe_short[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['first_2_words_lem'] + d['cpc4'] + \
                                                                                     d['first_2_words_lem_gvkey'])), \
                                                                                      feat_select, ('clf', clf)]))
                pipe_short[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['first_2_words_lem'] + d['cpc4'] + \
                                                                                     d['first_2_words_lem_cpc4'])), \
                                                                                      feat_select, ('clf', clf)]))
                pipe_short[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['f2_f10_lem'] + d['cpc4'])), \
                                              feat_select, ('clf', clf)]))
                pipe_short[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['f2_f10_lem'] + d['cpc4'] + \
                                                                                     d['f2_f10_lem_gvkey'])), \
                                                                                       feat_select, ('clf', clf)]))
                pipe_short[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['f2_f10_lem'] + d['cpc4'] + \
                                                                                     d['f2_f10_lem_cpc4'])), \
                                                                                     feat_select, ('clf', clf)]))
                pipe_long[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['publn_claims_lem'] + d['cpc4'])), \
                                                          feat_select, ('clf', clf)]))
                pipe_long[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['publn_claims_alt_lem'] + d['cpc4'])), \
                                                          feat_select, ('clf', clf)]))
                pipe_long[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['publn_claims_alt_lem'] + d['cpc4'] + \
                                                                               d['publn_claims_alt_lem_gvkey'])), \
                                                          feat_select, ('clf', clf)]))
                pipe_long[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['publn_claims_alt_lem'] + d['cpc4'] + \
                                                                               d['publn_claims_alt_lem_cpc4'])), \
                                                          feat_select, ('clf', clf)]))
            else:
                pipe_short[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['first_2_words_lem'] + d['cpc4'])), ('clf', clf)]))
                pipe_short[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['first_2_words_lem'] + d['cpc4'] + \
                                                                                     d['first_2_words_lem_gvkey'])), \
                                                                                      ('clf', clf)]))
                pipe_short[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['first_2_words_lem'] + d['cpc4'] + \
                                                                                     d['first_2_words_lem_cpc4'])), \
                                                                                      ('clf', clf)]))
                pipe_short[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['f2_f10_lem'] + d['cpc4'])), \
                                              ('clf', clf)]))
                pipe_short[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['f2_f10_lem'] + d['cpc4'] + \
                                                                                     d['f2_f10_lem_gvkey'])), \
                                                                                     ('clf', clf)]))
                pipe_short[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['f2_f10_lem'] + d['cpc4'] + \
                                                                                     d['f2_f10_lem_cpc4'])), \
                                                                                     ('clf', clf)]))
                pipe_long[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['publn_claims_lem'] + d['cpc4'])), ('clf', clf)]))
                pipe_long[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['publn_claims_alt_lem'] + d['cpc4'])), ('clf', clf)]))
                pipe_long[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['publn_claims_alt_lem'] + d['cpc4'] + \
                                                                               d['publn_claims_alt_lem_gvkey'])), ('clf', clf)]))
                pipe_long[iclf][iselect].append(Pipeline([('features', FeatureUnion(d['publn_claims_alt_lem'] + d['cpc4'] + \
                                                                               d['publn_claims_alt_lem_cpc4'])), ('clf', clf)]))

    pipe_type_list = [pipe_short, pipe_long]
    pipe_list = []
    for pipe_type in pipe_type_list:
        pipe_list.append(pipe_type)

    counter = - 1
    for ipipe_type, pipe_type in enumerate(pipe_list):
        for iclf, clf in enumerate(pipe_type):
            for iselect, select in enumerate(clf):
                for imodel, model in enumerate(select):
                    counter += 1
                    feature_list = []
                    patents = pd.DataFrame(list(data['patent'].unique()), columns=['patent'])
                    kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=5)
                    counter_cv = -1

                    for train_index, test_index in kf.split(patents):
                        counter_cv += 1
                        patents_train = patents.iloc[train_index, :]
                        patents_train.loc[: , 'train'] = 1
                        patents_train = patents.merge(patents_train, on=['patent'], how='outer')
                        patents_train['train'] = patents_train['train'].fillna(0)

                        df = data.reset_index().merge(patents_train, on=['patent'], how = 'inner').set_index('publn_claim_id')
                        X_train = df[df['train'] == 1]
                        X_test = df[df['train'] == 0]
                        y_train = df[classify][df['train'] == 1]
                        y_test = df[classify][df['train'] == 0]

                        pipe = model
                        pipe.fit(X_train, y_train)
                        X_test['prediction'] = pipe.predict(X_test)
                        X_test['correct'] = np.where(X_test['prediction'] == X_test[classify], 1, 0)

                        if counter_cv == 0:
                            predictions = X_test
                        else:
                            predictions = predictions.append(X_test)

                    for r in range(0, len(pipe.get_params()['features__transformer_list'])):
                        feature_list.append(pipe.get_params()['features__transformer_list'][r][0])

                    predictions = predictions[predictions['sic'] == int(sic_code)]
                    predictions['precision_process'] = round(precision_score(predictions[classify], predictions['prediction'], pos_label=1), 3)
                    predictions['recall_process'] = round(recall_score(predictions[classify], predictions['prediction'], pos_label=1), 3)
                    predictions['precision_product'] = round(precision_score(predictions[classify], predictions['prediction'], pos_label=0), 3)
                    predictions['recall_product'] = round(recall_score(predictions[classify], predictions['prediction'], pos_label=0), 3)
                    predictions['f1_score_process'] = round(f1_score(predictions[classify], predictions['prediction']), 3)
                    predictions['balanced_accuracy'] = round(balanced_accuracy_score(predictions[classify], predictions['prediction']), 3)
                    predictions['matthews_corrcoef'] = round(matthews_corrcoef(predictions[classify], predictions['prediction']), 3)

                    if predict_on == 'yes':
                        predictions.to_csv(str(sic_code) + '_predictions_pipe_' + str(counter) + '.csv')

                    predictions = predictions.reset_index()
                    predictions = predictions.groupby(['precision_process', 'recall_process', 'precision_product', 'recall_product', \
                                           'f1_score_process', 'balanced_accuracy', 'matthews_corrcoef', \
                                           'publn_claim_id'], as_index=False)[['correct']].mean().set_index('publn_claim_id')

                    predictions = predictions.merge(data, left_index=True, right_index=True)
                    predictions['classifier'] = classifier_list[iclf]
                    predictions['feature_list'] = ', '.join(feature_list)
                    predictions['sic_codes_for_prediction'] = ', '.join(sic_list)
                    predictions['pipe_number'] = counter
                    predictions['pipe_type_index'] = ipipe_type
                    predictions['classifier_index'] = iclf
                    predictions['select_from_model_index'] = iselect
                    predictions['model_index'] = imodel
                    predictions['obs_sic'] = predictions.shape[0]
                    predictions['obs_for_prediction'] = data.shape[0]
                    predictions['frac_obs_process_sic'] = round(predictions[classify].sum()/predictions['obs_sic'], 3)
                    predictions['feature_list'] = predictions['feature_list'].apply(lambda x: str(x).split(', '))
                    predictions['sic_codes_for_prediction'] = predictions['sic_codes_for_prediction'].apply(lambda x: str(x).split(', '))
                    predictions = predictions.iloc[[0]]
                    predictions = predictions[['sic', 'recall_process', 'precision_process', 'f1_score_process', 'balanced_accuracy', \
                                                'matthews_corrcoef', 'recall_product', 'precision_product', 'frac_obs_process_sic',  'obs_sic', \
                                                'obs_for_prediction', 'classifier', 'feature_list', 'sic_codes_for_prediction', \
                                                'pipe_number', 'pipe_type_index', 'classifier_index', 'select_from_model_index', \
                                                'model_index']]

                    predictions = predictions.set_index('pipe_number')

                    if counter == 0:
                        df2 = predictions
                    else:
                        df2 = df2.append(predictions)

    df2.to_csv(str(sic_code) + '_model_selection.csv')
    df3 = df2.iloc[[df2['matthews_corrcoef'].argmax()]]

    pipe_i1 = []
    clf_i1 = classifier_list[df3['classifier_index'].iloc[0]]
    feat_select = ('feat_select', SelectFromModel(clf_i1))

    for ifeature, feature in enumerate(df3['feature_list'].iloc[0]):
        if ifeature == 0:
            features_i1_initial = d[feature]
        else:
            features_i1_initial = features_i1_initial + d[feature]

    if not int(df3['select_from_model_index'].iloc[0]):
        for x in ['_count', '_stopword_frac', '_clean_avg_word_length']:
            pipe_i1.append(Pipeline([('features', FeatureUnion(features_i1_initial + d['publn_claims{0}'.format(x)])), \
                                     ('clf', clf_i1)]))


        pipe_i1.append(Pipeline([('features', FeatureUnion(features_i1_initial + d['publn_claims_clean_noun_frac'] + \
                                                          d['publn_claims_clean_verb_frac'] + d['publn_claims_clean_adj_frac'])), \
                                 ('clf', clf_i1)]))

        pipe_i1.append(Pipeline([('features', FeatureUnion(features_i1_initial + d['publn_claims_count'] + \
                                                           d['publn_claims_stopword_frac'] + \
                                                           d['publn_claims_clean_avg_word_length'] + \
                                                           d['publn_claims_clean_noun_frac'] + d['publn_claims_clean_verb_frac'] + \
                                                           d['publn_claims_clean_adj_frac'])), \
                                 ('clf', clf_i1)]))
        pipe_i1.append(Pipeline([('features', FeatureUnion(features_i1_initial + d['publn_claims_count'] + \
                                                   d['publn_claims_stopword_frac'] + \
                                                   d['publn_claims_clean_avg_word_length'] + \
                                                   d['publn_claims_clean_noun_frac'] + d['publn_claims_clean_verb_frac'] + \
                                                   d['publn_claims_clean_adj_frac'] + d['publn_claims_count_cpc4'] + \
                                                   d['publn_claims_stopword_frac_cpc4'] + \
                                                   d['publn_claims_clean_avg_word_length_cpc4'] + \
                                                   d['publn_claims_clean_noun_frac_cpc4'] + d['publn_claims_clean_verb_frac_cpc4'] + \
                                                   d['publn_claims_clean_adj_frac_cpc4'])), \
                         ('clf', clf_i1)]))
        pipe_i1.append(Pipeline([('features', FeatureUnion(features_i1_initial + d['publn_claims_count'] + \
                                                   d['publn_claims_stopword_frac'] + \
                                                   d['publn_claims_clean_avg_word_length'] + \
                                                   d['publn_claims_clean_noun_frac'] + d['publn_claims_clean_verb_frac'] + \
                                                   d['publn_claims_clean_adj_frac'] + d['publn_claims_count_gvkey'] + \
                                                   d['publn_claims_stopword_frac_gvkey'] + \
                                                   d['publn_claims_clean_avg_word_length_gvkey'] + \
                                                   d['publn_claims_clean_noun_frac_gvkey'] + d['publn_claims_clean_verb_frac_gvkey'] + \
                                                   d['publn_claims_clean_adj_frac_gvkey'])), \
                         ('clf', clf_i1)]))


    else:
        for x in ['_count', '_stopword_frac', '_clean_avg_word_length']:
            pipe_i1.append(Pipeline([('features', FeatureUnion(features_i1_initial + d['publn_claims{0}'.format(x)])), \
                                  feat_select, ('clf', clf_i1)]))


        pipe_i1.append(Pipeline([('features', FeatureUnion(features_i1_initial + d['publn_claims_clean_noun_frac'] + \
                                                          d['publn_claims_clean_verb_frac'] + d['publn_claims_clean_adj_frac'])), \
                                  feat_select, ('clf', clf_i1)]))

        pipe_i1.append(Pipeline([('features', FeatureUnion(features_i1_initial + d['publn_claims_count'] + \
                                                           d['publn_claims_stopword_frac'] + \
                                                           d['publn_claims_clean_avg_word_length'] + \
                                                           d['publn_claims_clean_noun_frac'] + d['publn_claims_clean_verb_frac'] + \
                                                           d['publn_claims_clean_adj_frac'])), feat_select, \
                                 ('clf', clf_i1)]))
        pipe_i1.append(Pipeline([('features', FeatureUnion(features_i1_initial + d['publn_claims_count'] + \
                                                       d['publn_claims_stopword_frac'] + \
                                                       d['publn_claims_clean_avg_word_length'] + \
                                                       d['publn_claims_clean_noun_frac'] + d['publn_claims_clean_verb_frac'] + \
                                                       d['publn_claims_clean_adj_frac'] + d['publn_claims_count_cpc4'] + \
                                                       d['publn_claims_stopword_frac_cpc4'] + \
                                                       d['publn_claims_clean_avg_word_length_cpc4'] + \
                                                       d['publn_claims_clean_noun_frac_cpc4'] + d['publn_claims_clean_verb_frac_cpc4'] + \
                                                       d['publn_claims_clean_adj_frac_cpc4'])), feat_select, \
                             ('clf', clf_i1)]))
        pipe_i1.append(Pipeline([('features', FeatureUnion(features_i1_initial + d['publn_claims_count'] + \
                                                   d['publn_claims_stopword_frac'] + \
                                                   d['publn_claims_clean_avg_word_length'] + \
                                                   d['publn_claims_clean_noun_frac'] + d['publn_claims_clean_verb_frac'] + \
                                                   d['publn_claims_clean_adj_frac'] + d['publn_claims_count_gvkey'] + \
                                                   d['publn_claims_stopword_frac_gvkey'] + \
                                                   d['publn_claims_clean_avg_word_length_gvkey'] + \
                                                   d['publn_claims_clean_noun_frac_gvkey'] + d['publn_claims_clean_verb_frac_gvkey'] + \
                                                   d['publn_claims_clean_adj_frac_gvkey'])), feat_select, \
                         ('clf', clf_i1)]))

    for imodel, model in enumerate(pipe_i1):
        counter += 1
        feature_list = []
        patents = pd.DataFrame(list(data['patent'].unique()), columns=['patent'])
        kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=5)
        counter_cv = -1

        for train_index, test_index in kf.split(patents):
            counter_cv += 1
            patents_train = patents.iloc[train_index, :]
            patents_train.loc[: , 'train'] = 1
            patents_train = patents.merge(patents_train, on=['patent'], how='outer')
            patents_train['train'] = patents_train['train'].fillna(0)

            df = data.reset_index().merge(patents_train, on=['patent'], how = 'inner').set_index('publn_claim_id')
            X_train = df[df['train'] == 1]
            X_test = df[df['train'] == 0]
            y_train = df[classify][df['train'] == 1]
            y_test = df[classify][df['train'] == 0]

            pipe = model
            pipe.fit(X_train, y_train)
            X_test['prediction'] = pipe.predict(X_test)
            X_test['correct'] = np.where(X_test['prediction'] == X_test[classify], 1, 0)

            if counter_cv == 0:
                predictions = X_test
            else:
                predictions = predictions.append(X_test)

        for r in range(0, len(pipe.get_params()['features__transformer_list'])):
            feature_list.append(pipe.get_params()['features__transformer_list'][r][0])

        predictions = predictions[predictions['sic'] == int(sic_code)]
        predictions['precision_process'] = round(precision_score(predictions[classify], predictions['prediction'], pos_label=1), 3)
        predictions['recall_process'] = round(recall_score(predictions[classify], predictions['prediction'], pos_label=1), 3)
        predictions['precision_product'] = round(precision_score(predictions[classify], predictions['prediction'], pos_label=0), 3)
        predictions['recall_product'] = round(recall_score(predictions[classify], predictions['prediction'], pos_label=0), 3)
        predictions['f1_score_process'] = round(f1_score(predictions[classify], predictions['prediction']), 3)
        predictions['balanced_accuracy'] = round(balanced_accuracy_score(predictions[classify], predictions['prediction']), 3)
        predictions['matthews_corrcoef'] = round(matthews_corrcoef(predictions[classify], predictions['prediction']), 3)

        if predict_on == 'yes':
            predictions.to_csv(str(sic_code) + '_predictions_pipe_' + str(counter) + '.csv')

        predictions = predictions.reset_index()
        predictions = predictions.groupby(['precision_process', 'recall_process', 'precision_product', 'recall_product', \
                                           'f1_score_process', 'balanced_accuracy', 'matthews_corrcoef', \
                                        'publn_claim_id'], as_index=False)[['correct']].mean().set_index('publn_claim_id')

        predictions = predictions.merge(data, left_index=True, right_index=True)
        predictions['classifier'] = clf_i1
        predictions['feature_list'] = ', '.join(feature_list)
        predictions['sic_codes_for_prediction'] = ', '.join(sic_list)
        predictions['pipe_number'] = counter
        predictions['pipe_type_index'] = int(df3['pipe_type_index'].iloc[0])
        predictions['classifier_index'] = int(df3['classifier_index'].iloc[0])
        predictions['select_from_model_index'] = int(df3['select_from_model_index'].iloc[0])
        predictions['model_index'] = imodel
        predictions['obs_sic'] = predictions.shape[0]
        predictions['obs_for_prediction'] = data.shape[0]
        predictions['frac_obs_process_sic'] = round(predictions[classify].sum()/predictions['obs_sic'], 3)
        predictions['feature_list'] = predictions['feature_list'].apply(lambda x: str(x).split(', '))
        predictions['sic_codes_for_prediction'] = predictions['sic_codes_for_prediction'].apply(lambda x: str(x).split(', '))
        predictions = predictions.iloc[[0]]
        predictions = predictions[['sic', 'recall_process', 'precision_process', 'f1_score_process', 'balanced_accuracy', \
                                   'matthews_corrcoef', 'recall_product', 'precision_product', 'frac_obs_process_sic',  'obs_sic', \
                                   'obs_for_prediction', 'classifier', 'feature_list', 'sic_codes_for_prediction', \
                                            'pipe_number', 'pipe_type_index', 'classifier_index', 'select_from_model_index', \
                                             'model_index']]

        predictions = predictions.set_index('pipe_number')
        df2 = df2.append(predictions)

    df2.to_csv(str(sic_code) + '_model_selection.csv')
    df3 = df2.iloc[[df2['matthews_corrcoef'].argmax()]]
    df4 = pd.read_csv(root + '//data//diagnostics.csv', index_col = 'sic')
    df3 = df3.set_index('sic')
    df4 = df4.reindex(columns=df4.columns.union(df3.columns))

    if df4['matthews_corrcoef'].iloc[df4.reset_index().index[df4.reset_index()['sic'] == sic_code]].isnull().iloc[0]:
        df4.update(df3)
        df4.reset_index(inplace=True)
        df4 = df4[['sic', 'sic_desc', 'patents', 'frac_patents', 'recall_process', 'precision_process', 'f1_score_process', \
                             'balanced_accuracy', 'matthews_corrcoef', 'recall_product', 'precision_product', 'frac_obs_process_sic', \
                             'obs_sic', 'obs_for_prediction', 'feature_list', 'classifier', \
                             'sic_codes_for_prediction', 'select_from_model_index']]
        df4.to_csv(root + '//data//diagnostics.csv', index=False)
        df4[df4['sic'].astype(str) == sic_code]

    elif list(df3['matthews_corrcoef'].iloc[df3.reset_index().index[df3.reset_index()['sic'] == sic_code]])[0] > \
            list(df4['matthews_corrcoef'].iloc[df4.reset_index().index[df4.reset_index()['sic'] == sic_code]])[0]:
        df4.update(df3)
        df4.reset_index(inplace=True)
        df4 = df4[['sic', 'sic_desc', 'patents', 'frac_patents', 'recall_process', 'precision_process', 'f1_score_process', \
                             'balanced_accuracy', 'matthews_corrcoef', 'recall_product', 'precision_product', 'frac_obs_process_sic', \
                             'obs_sic', 'obs_for_prediction', 'feature_list', 'classifier', \
                             'sic_codes_for_prediction', 'select_from_model_index']]
        df4.to_csv(root + '//data//diagnostics.csv', index=False)
        df4[df4['sic'].astype(str) == sic_code]

    if final_prediction_on == 'yes':
        clf_i1 = classifier_list[df3['classifier_index'].iloc[0]]
        feat_select = ('feat_select', SelectFromModel(clf_i1))

        for ifeature, feature in enumerate(df3['feature_list'].iloc[0]):
            if ifeature == 0:
                features_final = d[feature]
            else:
                features_final = features_final + d[feature]

        if not int(df3['select_from_model_index'].iloc[0]):
            pipe_final = Pipeline([('features', FeatureUnion(features_final)), ('clf', clf_i1)])

        else:
            pipe_final = Pipeline([('features', FeatureUnion(features_final)), feat_select, ('clf', clf_i1)])

        pipe_final.fit(data, data[classify])

        #unzip the file
        with zipfile.ZipFile(str(sic_code) + '_discern.zip', 'r') as my_zip:
            my_zip.extract(str(sic_code) + '_discern.csv')

        df_all = pd.read_csv(str(sic_code) + '_discern.csv', index_col=['publn_claim_id'], low_memory = False)
        df_hc = pd.read_csv(str(sic_code) + '_discern_hand_classified.csv', index_col=['publn_claim_id'], low_memory = False)
        
        #drop classification columns and those with no publn_claim data so that I can overwrite them with new predictions
        df_all = df_all.drop(columns=[classify, 'classified'], errors = 'ignore')
        df_nopub = df_all[(df_all['publn_claims_clean'].isnull() == 1) & (df_all['hand_classified'] == 0)]
        df_all = df_all.dropna(subset = ['publn_claims_clean'])
        df_unclass = df_all[df_all['hand_classified'] == 0]
    
        #make predictions
        df_unclass[classify] = pipe_final.predict(df_unclass)
        
        #append hand classified
        df_all = df_unclass.append(df_hc)
        df_all = df_all.append(df_nopub)

        #zip the up new file if losing no data, else zip up the old file
        if df_all.shape[0] == pd.read_csv(str(sic_code) + '_discern.csv', index_col=['publn_claim_id'], low_memory = False).shape[0]:
            
            df_all.to_csv(str(sic_code) + '_discern.csv')
            os.remove(str(sic_code) + '_discern.zip')

            with zipfile.ZipFile(str(sic_code) + '_discern.zip', 'w', compression = zipfile.ZIP_DEFLATED) as my_zip:
                my_zip.write(str(sic_code) + '_discern.csv')

        df_all = df_all[['patent', classify, 'cpc', 'cpc4', 'cpc3', 'ayear', 'pyear', \
                         'adate', 'pdate', 'fcites_nonfam', 'fcites_fam', 'bcites_nonfam', 'bcites_fam']]
        df_all.to_csv(str(sic_code) + '_discern_classified.csv')
        os.remove(str(sic_code) + '_discern.csv')


if __name__ ==  '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [[executor.submit(classify_sic, isic_code, sic_code, isic_list, sic_list) \
                    for isic_code, sic_code in enumerate(sic_list)] \
                    for isic_list, sic_list in enumerate(sic_code_list)]

        for f in concurrent.futures.as_completed(results[0]):
            print(f.result())

        for f in concurrent.futures.as_completed(results[1]):
            print(f.result())
