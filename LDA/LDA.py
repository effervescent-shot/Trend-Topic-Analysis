import os
import re
import sys
import numpy as np
import pandas as pd
import string
import re

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import common_corpus, common_dictionary, datapath
import gensim


import preprocessor as p
from preprocessor.api import clean
from wordcloud import WordCloud
import matplotlib.pyplot as plt


DATA_DIR = "../Data"
TWEETS_PATH = os.path.join(DATA_DIR, 'tweets')
TREND_PATH = os.path.join(DATA_DIR, 'trends')
SAVE_PATH = os.path.join(DATA_DIR, 'save')
STATS_PATH = os.path.join(DATA_DIR, 'stats')
TOPICS_PATH = os.path.join(DATA_DIR, 'topics')

def semmatize_text(text):
    ps = PorterStemmer()
    return [ps.stem(w)  for w in text if len(w)>3]

def tokanize_text(trend_doc):
    return trend_doc.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)

def process_lda_format(trend_doc):
    tokenized_df = tokanize_text(trend_doc)
    stemmed_dataset = tokenized_df.apply(semmatize_text)

    return stemmed_dataset

def initialize_corpus_and_dictionary(stemmed_dataset):
    dictionary_of_words = gensim.corpora.Dictionary(stemmed_dataset)
    word_corpus = [dictionary_of_words.doc2bow(word) for word in stemmed_dataset]

    return word_corpus, dictionary_of_words

def lda_datasets(trend_doc):
    stemmed_dataset = process_lda_format(trend_doc)
    corpus, dictionary = initialize_corpus_and_dictionary(stemmed_dataset)

    return stemmed_dataset, corpus, dictionary


def run_lda():
    # Model with the best coherence_value
    lda_model_16 = models.ldamodel.LdaModel(corpus = corpus, id2word=dictionary, num_topics=16,
                                            random_state=1, update_every=1, chunksize=100,
                                            passes=50, alpha='auto', per_word_topics=True)

    cwd = os.getcwd()
    temp_file = datapath(os.path.join(cwd, "models/lda_model_16"))
    print('Model is saving... at', temp_file)
    lda_model_16.save(temp_file)

    # Compute Perplexity Score
    print('Perplexity Score: ', lda_model_16.log_perplexity(corpus))

    # Compute Coherence Score
    cohr_val = CoherenceModel(model=lda_model_16, texts=stemmed_dataset, dictionary=dictionary,
                                      coherence='c_v').get_coherence()

    print('Coherence Score: ', cohr_val)



if __name__ == '__main__':
    dfs_train = pd.read_csv(os.path.join(SAVE_PATH, "lda_train_data"), header=0, parse_dates=['trend_date'])
    dfs_test = pd.read_csv(os.path.join(SAVE_PATH, "lda_test_data"), header=0, parse_dates=['trend_date'])

    print("Train data size {}".format(dfs_train.shape[0]))
    print("Test data size {}".format(dfs_test.shape[0]))

    dfsLDA = dfs_train.loc[:,["trend", "text"]]
    dfsLDA.dropna(inplace=True)
    trend_doc = dfsLDA.groupby(['trend'])['text'].apply(lambda x: ','.join(x)).reset_index()

    stemmed_dataset, corpus, dictionary = lda_datasets(trend_doc)
    run_lda()

    # Load a potentially pretrained model from disk.
    cwd = os.getcwd()
    temp_file = datapath(os.path.join(cwd, "models/lda_model_16"))
    lda = models.ldamodel.LdaModel.load(temp_file)

    words_match = re.compile(r'\"\w+\"')
    for idx, topic in lda.print_topics(-1):
        topic_file = open(os.path.join(TOPICS_PATH, "topic-"+str(idx)+".txt"), "w+")
        words = re.findall( words_match, topic)
        topic_file.write( str(words))
        topic_file.close()

    print("Topics are saved under topics.")
