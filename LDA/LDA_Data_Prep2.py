import os
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
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


def semmatize_text(text):
    ps = PorterStemmer()
    return [ps.stem(w) for w in text if len(w) > 3]


def tokanize_text(trend_doc):
    return trend_doc.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)


def remove_stopwords(texts):
    return [word for word in texts if word not in stop_words]


def process_lda_format(trend_docs):
    tokenized_df = tokanize_text(trend_docs)
    stemmed_dataset = tokenized_df.apply(semmatize_text)
    stemmed_dataset = stemmed_dataset.map(lambda x: remove_stopwords(x))
    return stemmed_dataset


def initialize_corpus_and_dictionary(stemmed_dataset):
    dictionary_of_words = gensim.corpora.Dictionary(stemmed_dataset)
    word_corpus = [dictionary_of_words.doc2bow(word) for word in stemmed_dataset]

    return word_corpus, dictionary_of_words


if __name__ == '__main__':

    dfs_train = pd.read_csv(os.path.join(SAVE_PATH, "lda_train_data"), header=0, parse_dates=['trend_date'])
    # dfs_test = pd.read_csv(os.path.join(SAVE_PATH, "lda_test_data"), header=0, parse_dates=['trend_date'])

    print("Train data size {}".format(dfs_train.shape[0]))
    # print("Test data size {}".format(dfs_test.shape[0]))

    # Create trend-doc from train_data
    dfsLDA = dfs_train.loc[:, ["trend", "text"]]
    dfsLDA.dropna(inplace=True)
    trend_docs = dfsLDA.groupby(['trend'])['text'].apply(lambda x: ','.join(x)).reset_index()

    # Get stop words
    stop_words = get_stop_words()

    # Create data structures to be used in LDA
    stemmed_dataset = process_lda_format(trend_docs)
    corpus, dictionary = initialize_corpus_and_dictionary(stemmed_dataset)


    # SAVE DATA
    #
    output_fname = get_tmpfile("corpus0.mm")
    MmCorpus.serialize(output_fname, corpus)
    mm = MmCorpus(output_fname)
    mm.save("./ldadata/corpus0")

    dictionary.save_as_text("./ldadata/dictionary0")

    stemmed_dataset.to_pickle('./ldadata/stemmed_data.pkl')

    print("FINISHED SAVING")

    print("CHECKING INTEGRITY")
    corps = MmCorpus.load("./ldadata/corpus0")
    print("CORPUS: ", len(corpus), len(corps))

    dicts = Dictionary.load_from_text("./ldadata/dictionary0")
    print("DICTIONARY: ", len(dictionary), len(dicts))

    dataset = pd.read_pickle("./ldadata/stemmed_data.pkl")
    print("DATASET: ", len(stemmed_dataset), len(dataset))


