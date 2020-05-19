import pandas as pd
import random
import os
import numpy as np
from datetime import datetime
import multiprocessing as mp
import time
import re
from collections import namedtuple
import sys

DATA_DIR = "../Data"
TWEETS_PATH = os.path.join(DATA_DIR, 'tweets')
TREND_PATH = os.path.join(DATA_DIR, 'trends')
SAVE_PATH = os.path.join(DATA_DIR, 'save')
STATS_PATH = os.path.join(DATA_DIR, 'stats')

RuleMatch = namedtuple('RuleMatch', 'trend rule')
RuleMatchStats = namedtuple('RuleMatchStats', 'tweetSize matchedSize')
TrendMatch = namedtuple('TrendMatch', 'trend match matchRule')
stats = []


def camel_case_split(trend_topic):
    """
    param: trend_topic is a single word trend topic

    Check if a trend word is in the form of camel case without #
    If so split the camel case to its words

    return: list of all words in camel case format
    """
    trend_topic = re.sub('#', '', trend_topic)
    match_list = []
    #     for identifier in trend_topic:
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', trend_topic)
    match_list += [m.group(0) for m in matches]

    if (len(match_list)) == 1:
        return []
    return match_list


def onegram_augment(trend_topic):
    """
    param: trend_topic is a single word trend topic

    From a trend in onegram set create augmented set of the trend by
    apply upper-lower case transformation
    split the hash and rejoin
    split the camel case and rejoin
    write a rule for every augmentation

    return: set of augmented-trend-topic, set of augmeted-trend-topic if it was also camelcase
    """

    onegram = set([RuleMatch(trend=trend_topic, rule='simple')])
    onegram_up = set([RuleMatch(trend=gram.trend.upper(), rule='simple-upper') for gram in onegram])
    onegram_lower = set([RuleMatch(trend=gram.trend.lower(), rule='simple-lower') for gram in onegram])

    nohash = set()
    nohash_up = set()
    nohash_lower = set()

    if '#' in trend_topic:
        nohash = set([RuleMatch(trend=re.sub('#', '', trend_topic), rule='no-hashtag')])
        nohash_up = set([RuleMatch(trend=gram.trend.upper(), rule='no-hashtag-upper') for gram in nohash])
        nohash_lower = set([RuleMatch(trend=gram.trend.lower(), rule='no-hashtag-lower') for gram in nohash])

    camelCase = camel_case_split(trend_topic)
    camelSplit = set()
    if len(camelCase) != 0:
        cc = [RuleMatch(trend=gram, rule='camel') for gram in camelCase]
        cc_up = [RuleMatch(trend=gram.upper(), rule='camel-upper') for gram in camelCase]
        cc_lower = [RuleMatch(trend=gram.lower(), rule='camel-lower') for gram in camelCase]

        ccHashed = set([RuleMatch(trend='#' + gram, rule='camel-hashtag') for gram in camelCase])
        ccHashed_up = set([RuleMatch(trend='#' + gram.trend, rule='camel-upper-hashtag') for gram in cc_up])
        ccHashed_lower = set([RuleMatch(trend='#' + gram.trend, rule='camel-lower-hashtag') for gram in cc_lower])

        ccHashJoined = set([RuleMatch(trend='#'.join(camelCase), rule='camel-join-hashtag')])
        ccHashJoined_up = set([RuleMatch(trend='#'.join(camelCase).upper(), rule='camel-upper-join-hashtag')])
        ccHashJoined_lower = set([RuleMatch(trend='#'.join(camelCase).lower(), rule='camel-lower-join-hashtag')])

        """
        This part is to add it later to nongrams
        """
        if len(camelCase) > 1:
            camelSplit = camelSplit.union(set([RuleMatch(trend=' '.join(camelCase), rule='camel-join')]),
                                          set([RuleMatch(trend=' '.join(camelCase).upper(), rule='camel-join-upper')]),
                                          set([RuleMatch(trend=' '.join(camelCase).lower(), rule='camel-join-lower')]))

        camelCase = set().union(ccHashed, ccHashed_up, ccHashed_lower, ccHashJoined_up, ccHashJoined_lower, ccHashJoined)

    onegram = onegram.union(onegram_up, onegram_lower, nohash, nohash_up, nohash_lower, camelCase)

    return onegram, camelSplit


def nonegram_augment(trend_topic):
    """
    param: trend_topic is a multiple words trend topic

    Nongram means a trend consits of more than a single word
    transform each word to its upper-lower case
    augment with the original set

    return: set of augmented-trend-topic

    """
    nonegram = set([RuleMatch(trend=trend_topic, rule='simple')])
    nonegram_up = set([RuleMatch(trend=trend_topic.upper(), rule='simple-upper')])
    nonegram_lower = set([RuleMatch(trend=trend_topic.lower(), rule='simple-lower')])

    return nonegram.union(nonegram_up, nonegram_lower)


def index_trends(text, onegram_trend_dict, nonegram_trend_dict):
    """
    param: text, twitter text
    param: onegram_trend_dict, augmented trend dictionary of onegram trends
    param: nonegram_trend_dict, augmented trend dictionary of nonegram trends

    For each tweet text, go through augmented trend dictionaries, if there is a match in augmented version
    crate a TrendMatch, a namedtuple, which consist of the actual trend, matching version and matching rule

    return: set of TrendMatch tuples
    """

    try:
        tokens = set(text.split(' '))
        trend_set = set()


        ####### Match not only the onegram but with augmented set of it  #########
        for onegram, onegram_aug in onegram_trend_dict.items():
            if len(onegram_aug) != 0 :
                rm = RuleMatch(*zip(*onegram_aug))
                aug_list = list(rm.trend)
                rules = list(rm.rule)

                for ind, aug_trend in enumerate(aug_list):
                    if aug_trend in tokens:
                        trend_set.add(TrendMatch(trend=onegram, match=aug_trend, matchRule=rules[ind]))


        ###### Match not only the nonegram but with augmented set of it  #########
        for nonegram, nonegram_aug in nonegram_trend_dict.items():
            if len(nonegram_aug) != 0 :
                rm = RuleMatch(*zip(*nonegram_aug))
                aug_list = list(rm.trend)
                rules = list(rm.rule)

                for ind, aug_trend in enumerate(aug_list):
                    if aug_trend in " "+text+" ":
                        trend_set.add(TrendMatch(trend=nonegram, match=aug_trend, matchRule=rules[ind]))


        ###### Collect Statistics #####
        if len(trend_set) != 0 :
            # tm = TrendMatch(*zip(*trend_set))
            # unique_trends = set(tm.trend)
            # stats.append(len(unique_trends))
            return trend_set

        else:
            return None

    except:
        print("Text could not be processed: ",  text)
        return set()


def expand_trend_set(df, trend_col):
    non_list_cols = [col for col in (df.columns) if col != trend_col]
    df2 = pd.DataFrame(df[trend_col].tolist(), index=[df[col] for col in non_list_cols]) \
        .stack() \
        .reset_index(name=trend_col)[non_list_cols + [trend_col]]
    return df2


def prepare_data_trend_date_indexed_function(file, candidates):

    tweets_folder = TWEETS_PATH
    save_folder = SAVE_PATH

    df = pd.read_csv('%s/%s' % (tweets_folder, file))
    dfs = []
    total_tweet = 0

    for candidate in candidates:
        df_that_day = pd.DataFrame(df)
        trends_that_day = set(trends[trends.date == candidate]['name'])

        # Stats collection
        total_tweet += df_that_day.shape[0]
        stats = []

        if (len(trends_that_day) == 0):
            print('trends for %s not found!' % candidate)
            continue

        ################################### AUGMENT TREND SETS ############################################
        trends_that_day_onegrams = set([trend for trend in trends_that_day if len(trend.split(' ')) == 1])
        onegram_trend_dict = dict()
        camel_split_dict = dict()
        for k in trends_that_day_onegrams:
            v1, v2 = onegram_augment(k)
            onegram_trend_dict[k] = v1
            if len(v2)!= 0:
                camel_split_dict[k] = v2


        trends_that_day_nonegrams = trends_that_day - trends_that_day_onegrams
        nonegram_trend_dict = dict((k, nonegram_augment(k)) for k in trends_that_day_nonegrams)
        nonegram_trend_dict.update(camel_split_dict)

        ################################### APPLY TREND INDEX #############################################

        df_that_day['TrendMatch'] = df_that_day.text.apply( lambda x:
                                    index_trends(x, onegram_trend_dict, nonegram_trend_dict))
        df_that_day.dropna(subset=['TrendMatch'], inplace=True)

        df_that_day = expand_trend_set(df_that_day, 'TrendMatch')
        df_that_day['trend'] = df_that_day['TrendMatch'].apply(lambda x: x.trend)
        df_that_day['match'] = df_that_day['TrendMatch'].apply(lambda x: x.match)
        df_that_day['match_rule'] = df_that_day['TrendMatch'].apply(lambda x: x.matchRule)

        df_that_day.drop(['TrendMatch'], axis=1, inplace=True)
        ###################################################################################################

        df_that_day['trend_date'] = candidate
        dfs.append(df_that_day)


    dfs = pd.concat(dfs)
    new_file = file.split('_')[0] + "_trends.csv"
    dfs.to_csv('%s/%s' % (save_folder, new_file), index=False)
    statsFile = open(os.path.join(STATS_PATH, "stats.txt"), "a+")
    statsFile.write(str(total_tweet) + "," + str(dfs.shape[0]) + "\n")
    statsFile.close()



def prepare_data_trend_date_indexed_parallelized():
    tweets_folder = TWEETS_PATH
    save_folder = SAVE_PATH

    files = os.listdir(tweets_folder)
    files = [file for file in files if file >= start and file <= end and 'csv' in file]
    pool = mp.Pool(mp.cpu_count() - 2)

    for i, file in enumerate(files):
        print('%d / %d - %s' % (i, len(files), file))
        date = file.split('_')[0]
        that_day = pd.Timestamp(date).date()
        one_day_before = that_day - pd.Timedelta(days=1)
        one_day_after = that_day + pd.Timedelta(days=1)
        candidates = [str(that_day), str(one_day_before), str(one_day_after)]
        pool.apply_async(prepare_data_trend_date_indexed_function, args=(file, candidates))

    pool.close()
    pool.join()


def trend_date_parser(d):
    format_in = "%Y-%m-%d %X"
    format_out = "%Y-%m-%d"

    d = datetime.strptime(d, format_in)
    return d.strftime(format_out)





if __name__ == '__main__':

    # start = sys.argv[1]
    # end = sys.argv[2]

    start ="2019-06-30"
    end = "2019-08-01"
    print("Hello, tweets from {} to {} will be considered.".format(start, end))
    trends = pd.read_csv(os.path.join(TREND_PATH, 'all_trends_world.csv'),
                         parse_dates=['date'], date_parser=trend_date_parser)

    prepare_data_trend_date_indexed_parallelized()

