import os
import re
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp



from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import common_corpus, common_dictionary, datapath
import gensim


from gensim.corpora import MmCorpus, Dictionary
from gensim.test.utils import get_tmpfile


# NLTK Stop words
from nltk.corpus import stopwords

DATA_DIR = "../Data"
TWEETS_PATH = os.path.join(DATA_DIR, 'tweets')
TREND_PATH = os.path.join(DATA_DIR, 'trends')
SAVE_PATH = os.path.join(DATA_DIR, 'save')
STATS_PATH = os.path.join(DATA_DIR, 'stats')
TOPICS_PATH = os.path.join(DATA_DIR, 'topics')




def load_lda_datasets():

    corps = MmCorpus.load("./ldadata/corpus0")
    print("LOADING CORPUS, LENGTH: ", len(corps))

    dicts = Dictionary.load_from_text("./ldadata/dictionary0")
    print("LOADING DICTIONARY, LENGTH: ", len(dicts))

    dataset = pd.read_csv("./ldadata/stemmed_data.zip")
    print("LOADING DATASET, LENGTH: ", len(dataset))

    return dataset, corps, dicts

def run_lda(topic_num):
    # Model with the best coherence_value
    lda_model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_num,
                                            random_state=1, update_every=1, chunksize=100,
                                            passes=50, alpha='auto', per_word_topics=True)

    cwd = os.getcwd()
    temp_file = datapath(os.path.join(cwd, "models/lda_model_"+str(topic_num)))
    print('Model is saving... at', temp_file)
    lda_model.save(temp_file)

    return lda_model


def models_run():

    pool = mp.Pool(mp.cpu_count() - 2)

    for topic_number in range(10, 17, 3):
        pool.apply_async(run_lda, args=(topic_number))

    pool.close()
    pool.join()


def models_check(topic_num):

    cwd = os.getcwd()
    temp_file = datapath(os.path.join(cwd, "models/lda_model_"+str(topic_num)))
    lda_model =  models.ldamodel.LdaModel.load(temp_file)
    print("Topic number = ", topic_num)

    # Compute Perplexity Score
    print('Perplexity Score: ', lda_model.log_perplexity(corpus))

    # Compute Coherence Score
    cohr_val = CoherenceModel(model=lda_model, texts=stemmed_dataset, dictionary=dictionary, coherence='c_v').get_coherence()
    print('Coherence Score: ', cohr_val)



if __name__ == '__main__':

    print("HRE WE GO!")
    stemmed_dataset, corpus, dictionary = load_lda_datasets()

