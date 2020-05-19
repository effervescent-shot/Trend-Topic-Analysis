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
from gensim.test.utils import common_corpus, common_dictionary
import gensim
import preprocessor as p
from preprocessor.api import clean
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_DIR = "../Data"
TWEETS_PATH = os.path.join(DATA_DIR, 'tweets')
TREND_PATH = os.path.join(DATA_DIR, 'trends')
SAVE_PATH = os.path.join(DATA_DIR, 'save')
STATS_PATH = os.path.join(DATA_DIR, 'stats')


# os.listdir(SAVE_PATH)[:5]

def collect_stats(dfs):
    """
    count how many tweet a trend has on per day
    count how many trend a tweet has on per day
    """
    trend_by = dfs.groupby(["trend_date", "trend"]).agg({"id": "nunique"}).reset_index()
    #     per_day = trend_by.groupby('trend_date')['id'].mean()
    trend_by.to_csv(os.path.join(STATS_PATH, "tweetMatch_per_trend.txt"), index=True)

    tweet_by = dfs.groupby(["trend_date", "id"]).agg({"trend": "nunique"}).reset_index()
    #     per_day = tweet_by.groupby('trend_date')['id'].mean()
    tweet_by.to_csv(os.path.join(STATS_PATH, "trendMatch_per_tweet.txt"), index=True)


def clear_text(dfs):
    """
    clean the digits, punctuation, non-ascii characters
    """
    remove_digits = str.maketrans('', '', string.digits)
    exclude = '[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
    non_ascii = re.compile(r'[^\x00-\x7F]+')

    dfs['trend'] = dfs['trend'].map(lambda x: x.lower())
    dfs['trend'] = dfs['trend'].map(lambda x: x.translate(remove_digits))
    dfs['trend'] = dfs['trend'].map(lambda x: re.sub(str(exclude), '', x))

    dfs['text'] = dfs['text'].map(lambda x: x.lower())
    dfs['text'] = dfs['text'].map(lambda x: clean(x))
    dfs['text'] = dfs['text'].map(lambda x: x.translate(remove_digits))
    dfs['text'] = dfs['text'].map(lambda x: re.sub(str(exclude), '', x))
    dfs['text'] = dfs['text'].map(lambda x: re.sub(non_ascii, '', x))

    dfs.drop_duplicates(inplace=True)


def plotting_stats(dfs):
    """
    Plotting stats
    #of tweet per topic per day
    #of author per topic per day
    """
    tweet_by = dfs.groupby(["trend_date", "trend"]).agg({"id": "nunique"}).reset_index()
    author_by = dfs.groupby(["trend_date", "trend"]).agg({"author_id": "nunique"}).reset_index()

    tweet_by.to_csv(os.path.join(STATS_PATH, "tweetCount_per_trend.txt"), index=True)
    author_by.to_csv(os.path.join(STATS_PATH, "authorCount_per_trend.txt"), index=True)


def prepare_data():
    save_folder = SAVE_PATH

    files = os.listdir(save_folder)
    files = [file for file in files if file >= start and file <= end and 'csv' in file]
    dfs = []

    for i, file in enumerate(files):
        date = file.split('_')[0]
        print('%d / %d - %s - date: %s' % (i, len(files), file, str(date)))

        df = pd.read_csv(os.path.join(save_folder, file), header=0, usecols=list(range(10)),
                         parse_dates=['trend_date'])
        df = df[df.lang == "en"]
        df.drop(["Unnamed: 0", "lang", "created_at", "match", "match_rule"], inplace=True, axis=1)
        df.dropna(inplace=True)
        dfs.append(df)

    dfs = pd.concat(dfs)

    collect_stats(dfs)
    clear_text(dfs)
    plotting_stats(dfs)

    dfs_train, dfs_test = train_test_split(dfs, test_size=0.00001)
    dfs_train.to_csv(os.path.join(SAVE_PATH, "lda_train_data"), index=False)
    dfs_test.to_csv(os.path.join(SAVE_PATH, "lda_test_data"), index=False)

    return dfs_train, dfs_test
