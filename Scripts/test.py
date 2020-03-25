import os
import json
import nltk

import pandas as pd
import numpy as np

rootdir = '../data/'
# dirs = os.listdir(rootdir)
#
# for file in dirs:
# 	print(file)



march_01 =  pd.read_csv(rootdir+'2019-03-01_tweetswithlinks_tr.csv', header=None, usecols = np.arange(11),
                        names=['tid','time','uid','language','text','sname','ttype','RT/Q_uid','RT/Q_sname','Q_text','Q_link'], parse_dates = ['time'],

                        dtype={'tid':np.long, 'uid':np.long, 'language':str,'text':str,'sname':str,'ttype':str,'RT/Q_sname':str,'Q_text':str,'Q_link':str  } )

march_01 = march_01[march_01['language'] == 'tr']
march_01.drop(['language'],axis=1, inplace=True)
march_01.dropna(inplace=True)
march_01['RT/Q_uid'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()

print(len(march_01))
print(march_01.head(5))
print(march_01.columns)
print(march_01.dtypes)