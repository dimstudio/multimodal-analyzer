# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:09:35 2018

@author: ddm
"""

import numpy as np
import pandas as pd

import seaborn as sns
import json
from pandas.io.json import json_normalize    
import zipfile
import os
import operator

from sklearn.svm import SVC
from sklearn.feature_selection import RFE


session_path = '../2018-7-16-15H19M43S524_annotated.zip'

dfALL = pd.DataFrame()#Summary data frame

#1. reading data
with zipfile.ZipFile(session_path) as z:
    # First look for annotation.json
    for filename in z.namelist():
        if not os.path.isdir(filename):
            if '.json' in filename:
                with z.open(filename) as f:
                     data = json.load(f) 
                if 'intervals' in data:
                    df = pd.concat([pd.DataFrame(data), 
                        json_normalize(data['intervals'])], 
                        axis=1).drop('intervals', 1)#
                    if (df.applicationName.all()=='ManualAnnotations' or df.applicationName.all()=='AutomaticAnnotations'):
                        df = df.apply(pd.to_numeric, errors='ignore')
                        df.start = pd.to_datetime(df.start, unit='s')   
                        df.end = pd.to_datetime(df.end, unit='s')  
                        df['duration'] = (df.end-df.start) / np.timedelta64(1, 's')                           
                        df2 = df.copy()                  

                elif 'frames' in data:

                    df = pd.concat([pd.DataFrame(data), 
                        json_normalize(data['frames'])], 
                        axis=1).drop('frames', 1)
                    df['frameStamp'] = pd.to_datetime(df['frameStamp'], format='%H:%M:%S.%f')
                    df.columns = df.columns.str.replace("frameAttributes", df.applicationName.all())
                    df = df.set_index('frameStamp').iloc[:,2:]
                    df = df[~df.index.duplicated(keep='first')]
                    dfALL = dfALL.merge(df,how='outer',left_index=True,right_index=True)
                    
# Cast to float 
dfALL =  dfALL.apply(pd.to_numeric, errors='ignore').fillna(method='bfill')

# Drop the empty attributes
to_exclude = ['Ankle']
df1 = dfALL.loc[:,dfALL.sum(axis=0)!=0]
# Select only float dTypes
df1 = df1.select_dtypes(include=['float64'])
# Exclude features
for el in to_exclude:
    df1 = df1[[col for col in df1.columns if el not in col]]

df1.index = df1.index + pd.DateOffset(years=70)
masked_df = [
    df1[(df2_start <= df1.index) & (df1.index <= df2_end)]
    for df2_start, df2_end in zip(df2['start'], df2['end'])
]
# Feature aggregators
aggregations = ['max','min','std','mean']
features = []
for key in df1.columns.values:
    for a in aggregations:
        fname = key+'.'+a
        features.append(fname)
        if a == 'max':
            df2[fname] = [np.max(dt[key]) if not dt.empty else None for dt in masked_df]
        elif a == 'min':
            df2[fname] = [np.min(dt[key]) if not dt.empty else None for dt in masked_df]
        elif a == 'std':    
            df2[fname] = [np.std(dt[key]) if not dt.empty else None for dt in masked_df]
        elif a == 'mean':    
            df2[fname] = [np.mean(dt[key]) if not dt.empty else None for dt in masked_df]


#Feauture ranking
X = df2[features].values
y = df2['classRelease'].values
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
importance = dict(zip(df2[features].columns, rfe.ranking_))
importance = sorted(importance.items(), key=operator.itemgetter(1))
print importance

#Example plot correlation
#sns.pairplot(df2[features+['duration','classRate']],hue='classRate',palette=sns.color_palette('BuGn', 2))

