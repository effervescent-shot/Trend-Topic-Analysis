import pandas as pd
import random
import os
import numpy as np
from datetime import datetime
import multiprocessing as mp
trends = pd.read_csv('all_trends_world.csv')


def expand_list(df, list_column, new_column):
    lens_of_lists = df[list_column].apply(len)
    origin_rows = range(df.shape[0])
    destination_rows = np.repeat(origin_rows, lens_of_lists)
    non_list_cols = (
      [idx for idx, col in enumerate(df.columns)
       if col != list_column]
    )
    expanded_df = df.iloc[destination_rows, non_list_cols].copy()
    expanded_df[new_column] = (
      [item for items in df[list_column] for item in items]
      )
    expanded_df.reset_index(inplace=True, drop=True)
    return expanded_df

def index_trends(text, onegram_trend_set, others):
    try:
        tokens = text.split(' ')
        onegrams = set(tokens).intersection(onegram_trend_set)
        others = set([other for other in others if (" " + other + " ") in (" " + text +" ")])
        return onegrams.union(others)

    except:
        print(text)
        return set()

def prepare_data_trend_date_indexed_function(file, candidates, tweets_folder, save_folder):
    df = pd.read_csv('%s/%s' % (tweets_folder, file))
    dfs = []
    for candidate in candidates:
        df_that_day = pd.DataFrame(df)
        trends_that_day = set(trends[trends.date == candidate]['name'])
        if (len(trends_that_day) == 0):
            print('%s not found!' % candidate)
            continue

        trends_that_day_onegrams = set([trend for trend in trends_that_day if len(trend.split(' ')) == 1])
        trends_that_day_nonegrams = trends_that_day - trends_that_day_onegrams

        df_that_day['trends'] = df_that_day.text.apply(
            lambda x: index_trends(x, trends_that_day_onegrams, trends_that_day_nonegrams))
        df_that_day = expand_list(df_that_day, 'trends', 'trend')
        df_that_day['trend_date'] = candidate
        dfs.append(df_that_day)

    dfs = pd.concat(dfs)
    new_file = file.split('_')[0] + "_trends.csv"
    dfs.to_csv('%s/%s' % (save_folder, new_file), index=False)

def prepare_data_trend_date_indexed_parallelized(main_folder = '/./', tweets_folder = 'ezgi_en', save_folder = 'ezgi_trends'):
    tweets_folder = main_folder + tweets_folder
    save_folder = main_folder + save_folder

    files = os.listdir(tweets_folder)
    files = [file for file in files if file >= '2013-07-07' and 'csv' in file] # trends only available after this date

    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count() - 2)

    for i, file in enumerate(files):
        print('%d / %d' % (i, len(files)))
        date = file.split('_')[0]
        that_day = pd.Timestamp(date).date()
        one_day_before = that_day - pd.Timedelta(days = 1)
        one_day_after = that_day + pd.Timedelta(days = 1)
        candidates = [str(that_day), str(one_day_before), str(one_day_after)]
        pool.apply_async(prepare_data_trend_date_indexed_function, args=(file, candidates, tweets_folder, save_folder))

    pool.close()
    pool.join()

if __name__ == '__main__':
    prepare_data_trend_date_indexed_parallelized(trends)