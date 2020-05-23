import os
import sys
import numpy as np
import pandas as pd


import nltk
from nltk.stem import PorterStemmer
# NLTK Stop words
from nltk.corpus import stopwords

import gensim
from gensim import corpora, models
from gensim.test.utils import  datapath
from gensim.corpora import MmCorpus, Dictionary
from gensim.test.utils import get_tmpfile

import pickle

DATA_DIR = "../Data"
TWEETS_PATH = os.path.join(DATA_DIR, 'tweets')
TREND_PATH = os.path.join(DATA_DIR, 'trends')
SAVE_PATH = os.path.join(DATA_DIR, 'save')
STATS_PATH = os.path.join(DATA_DIR, 'stats')
TOPICS_PATH = os.path.join(DATA_DIR, 'topics')

def semmatize_stop_words(w):
    ps = PorterStemmer()
    return ps.stem(w)

def get_stop_words():
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'rt'])
    stop_words_stem = [semmatize_stop_words(x) for x in stop_words]
    stop_words.extend(stop_words_stem)
    stop_words = list(dict.fromkeys(stop_words))

    return stop_words

def load_raw_datasets():

    dfs_train =  pd.read_csv(os.path.join(SAVE_PATH, "lda_train_data"), header=0, parse_dates=['trend_date'])
    dfsLDA_train = dfs_train.loc[:,["trend","text"]]
    dfsLDA_train.dropna(inplace=True)
    trend_docs = dfsLDA_train.groupby(['trend'])['text'].apply(lambda x: ','.join(x)).reset_index()
    print("LOADING RAW DATA TREND-TEXT, LENGTH: ", len(trend_docs))

    return dfs_train, trend_docs

def load_lda_datasets():

    corps = MmCorpus.load("../LDA/ldadata/corpus0")
    print("LOADING CORPUS, LENGTH: ", len(corps))

    dicts = Dictionary.load_from_text("../LDA/ldadata/dictionary0")
    print("LOADING DICTIONARY, LENGTH: ", len(dicts))


    file = open("../LDA/ldadata/prev-small/stemmed_data.pkl", "rb")
    obj = pickle.load(file)
    dataset = pd.DataFrame(obj)
    #

    # dataset = pd.read_pickle("../LDA/ldadata/stemmed_data.pkl")
    print("LOADING DATASET, LENGTH: ", len(dataset))

    return dataset, corps, dicts






dfs_train, trend_doc = load_raw_datasets()
stemmed_dataset, corpus, dictionary = load_lda_datasets()

def semmatize_text(text):
    ps = PorterStemmer()
    return [ps.stem(w)  for w in text if len(w)>3]


def tokanize_text(trend_doc):
    return trend_doc.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)


def remove_stopwords(texts):
    return [word for word in texts if word not in stop_words ]


def process_lda_format(trend_doc):
    tokenized_df = tokanize_text(trend_doc)
    stemmed_dataset = tokenized_df.apply(semmatize_text)
    stemmed_dataset = stemmed_dataset.map(lambda x: remove_stopwords(x))
    return stemmed_dataset


def load_test_dataset():
    dfs_test =  pd.read_csv(os.path.join(SAVE_PATH, "lda_test_data"), header=0, parse_dates=['trend_date'])
    dfsLDA_test = dfs_test.loc[:,["trend","text"]]
    dfsLDA_test.dropna(inplace=True)
    test_doc = dfsLDA_test.groupby(['trend'])['text'].apply(lambda x: ','.join(x)).reset_index()
    stemmed_test = process_lda_format(test_doc)
    corpus_test = [dictionary.doc2bow(word) for word in stemmed_test]

    return test_doc, stemmed_test, corpus_test

stop_words = get_stop_words()
test_doc, stemmed_test, corpus_test = load_test_dataset()

def load_model(topic_num):
    # Get file path
    cwd = os.getcwd()
    temp_file = datapath(os.path.join(cwd, "../LDA/models/lda_model_"+str(topic_num)))
    # Load a potentially pretrained model from disk.
    lda_test = models.ldamodel.LdaModel.load(temp_file)
    return lda_test


# with open('mypickle.pickle', 'wb') as f:
#     pickle.dump(some_obj, f)



# dfs_train =  pd.read_csv(os.path.join(SAVE_PATH, "lda_train_data"), header=0, parse_dates=['trend_date'])
# dfsLDA_train = dfs_train.loc[:,["trend","text"]]
# dfsLDA_train.dropna(inplace=True)
# trend_docs = dfsLDA_train.groupby(['trend'])['text'].apply(lambda x: ','.join(x)).reset_index()
# print("LOADING RAW DATA TREND-TEXT, LENGTH: ", len(trend_docs))
#
# stop_words = get_stop_words()
# # Create data structures to be used in LDA
# stemmed_dataset = process_lda_format(trend_docs)
# stemmed_dataset.to_pickle('./ldadata/stemmed_data_new.pkl')
