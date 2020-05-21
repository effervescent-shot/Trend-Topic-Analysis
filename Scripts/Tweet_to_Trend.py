

import pandas as pd
import random
import os
import numpy as np
from datetime import datetime
import multiprocessing as mp
import time
import re


# In[2]:


DATA_DIR = "../Data"
TWEETS_PATH = os.path.join(DATA_DIR, 'tweets')
TREND_PATH = os.path.join(DATA_DIR, 'trends')
SAVE_PATH = os.path.join(DATA_DIR, 'save')
#os.listdir(DATA_DIR)


# In[3]:


def camel_case_split(onegram):
    match_list = []
    for identifier in set(onegram):    
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        match_list += [m.group(0) for m in matches]
    
    return match_list
        
def onegram_augment(onegram):
    
    onegram = set(onegram)
    onegram_up = set([gram.upper() for gram in onegram])
    onegram_lower = set([gram.lower() for gram in onegram])
    
    nohash = set([re.sub('#','', gram) for gram in onegram])
    nohash_up = set([gram.upper() for gram in nohash])
    nohash_lower = set([gram.lower() for gram in nohash])
    
    camelCase = camel_case_split(nohash) 
    camelSplit = set()
    if len(camelCase) !=0 :
        cc_up = [gram.upper() for gram in camelCase]
        cc_lower = [gram.lower() for gram in camelCase]

        ccHashed = set(['#'+gram for gram in camelCase])
        ccHashed_up = set(['#'+gram for gram in cc_up])
        ccHashed_lower = set(['#'+gram for gram in cc_lower])
        
        ccHashJoined = set(['#'.join(camelCase)])
        ccHashJoined_up = set(['#'.join(cc_up)])
        ccHashJoined_lower = set(['#'.join(cc_lower)])
        
        
        if len(camelCase) > 1:
            camelSplit = camelSplit.union(set([' '.join(camelCase)]), 
                                          set([' '.join(cc_up)]), set([' '.join(cc_lower)]))
        
        camelCase = set().union(ccHashed, ccHashed_up, ccHashed_lower,
                        ccHashJoined_up, ccHashJoined_lower, ccHashJoined)
    
    return (onegram.union(onegram_up,onegram_lower, nohash,nohash_up,nohash_lower,camelCase), camelSplit)


def nonegram_augment(nonegram):
    nonegram = set(nonegram)
#     nonegram_up = set([gram.upper() for gram in nonegram])
#     nonegram_lower = set([gram.lower() for gram in nonegram])
    
    return nonegram #nonegram.union(nonegram_up, nonegram_lower)



def index_trends(text, onegram_trend_set, nonegram_trend_set):
    try:
        tokens = text.split(' ')
        trend_set = set()
        
        ####### Match not only the onegram but with augmented set of it  #########
        for onegram, onegram_aug in onegram_trend_set.items():
            onegram_match = set(tokens).intersection(onegram_aug) 
#             print(onegram_augmented)
#             print(tokens)
#             print(camel_split)
            if len(onegram_match)!= 0:
                trend_set.add(onegram)
        
        ####### Match not only the nonegram but with augmented set of it  #########
        for nonegram, nonegram_aug in nonegram_trend_set.items():
            others = set([other for other in nonegram_aug if (" " + other + " ") in (" " + text +" ")])
            if len(others)!=0:
                trend_set.add(nonegram)

        return trend_set
    
    except:
        print(text)
        return set()


def expand_trend_set(df, trend_col):
    
    non_list_cols = [col for col in (df.columns) if col != trend_col ]
    df2 = pd.DataFrame(df[trend_col].tolist(), index=[df[col] for col in non_list_cols]).stack().reset_index(name=trend_col)[non_list_cols+[trend_col]]
    return df2



def prepare_data_trend_date_indexed_function(file, candidates):
    
    tweets_folder =  TWEETS_PATH
    save_folder = SAVE_PATH
    
    df = pd.read_csv('%s/%s' % (tweets_folder, file))
    dfs = []
    
    for candidate in candidates:
        df_that_day = pd.DataFrame(df)
        trends_that_day = set(trends[trends.date == candidate]['name'])
        
        if (len(trends_that_day) == 0):
            print('trends for %s not found!' % candidate)
            continue

        ##################################################################################################
        trends_that_day_onegrams = set([trend for trend in trends_that_day if len(trend.split(' ')) == 1])
        onegram_trend_dict = dict()
        camel_split_dict = dict()
        
        for k in trends_that_day_onegrams:
            v1, v2 = onegram_augment([k])
            onegram_trend_dict[k] = v1
            if len(v2)!=0:
                camel_split_dict[k] = v2


        trends_that_day_nonegrams = trends_that_day - trends_that_day_onegrams
        nonegram_trend_dict = dict((k, nonegram_augment([k])) for k in trends_that_day_nonegrams)
        nonegram_trend_dict.update(camel_split_dict)        
        
        ################################### APPLY TREND INDEX #############################################
        
        df_that_day['trend'] = df_that_day.text.apply( lambda x: 
                                            index_trends(x, onegram_trend_dict, nonegram_trend_dict))
        df_that_day = expand_trend_set(df_that_day, 'trend')
        
        ##################################################################################################
        
        df_that_day['trend_date'] = candidate
        dfs.append(df_that_day)
        
    dfs = pd.concat(dfs)
#     dfs = dfs[['text','trends','trend_date']]\
#         .groupby(['trends','trend_date'])['text']\
#         .apply(lambda x: ','.join(x))\
#         .reset_index()
    
    new_file = file.split('_')[0] + "_trends.csv"
    dfs.to_csv('%s/%s' % (save_folder, new_file), index=False)



def prepare_data_trend_date_indexed_parallelized():
    
    tweets_folder =  TWEETS_PATH
    save_folder = SAVE_PATH

    files = os.listdir(tweets_folder)
    files = [file for file in files if file >= '2019-07-01' and 'csv' in file] # trends only available after this date
    pool = mp.Pool(mp.cpu_count() - 2)
    
    for i, file in enumerate(files):
        print('%d / %d - %s' % (i, len(files), file))
        date = file.split('_')[0]
        that_day = pd.Timestamp(date).date()
        one_day_before = that_day - pd.Timedelta(days= 1)
        one_day_after = that_day + pd.Timedelta(days= 1)
        candidates = [str(that_day), str(one_day_before), str(one_day_after)]
        pool.apply_async(prepare_data_trend_date_indexed_function, args=(file, candidates))

    pool.close()
    pool.join()
    


# In[9]:


def trend_date_parser(d):
    format_in =  "%Y-%m-%d %X"
    format_out = "%Y-%m-%d"
 
    d = datetime.strptime(d, format_in)
    return d.strftime(format_out)

trend_date_parser("2013-07-07 23:36:32")


# In[10]:

if __name__ == '__main__':
    
    trends = pd.read_csv( os.path.join(TREND_PATH, 'all_trends_world.csv'),
                         parse_dates=['date'], date_parser=trend_date_parser)
    prepare_data_trend_date_indexed_parallelized()






