# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:09:35 2018

@author: ddm
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import json
from pandas.io.json import json_normalize    
import zipfile
import os
import operator
import re

from sklearn.svm import SVC
from sklearn import preprocessing

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sessions = ['OKDDM1_2018-7-24-11H57M49S371_annotated.zip',
'OKDDM2_2018-7-24-12H36M20S340_annotated.zip',
'OKDDM9_2018-7-24-13H32M20S73_annotated.zip',
'OKDDM10_2018-7-24-13H35M10S20_annotated.zip',
'OKDDM11_2018-7-24-13H37M54S38_annotated.zip']

dfALL = pd.DataFrame()#Summary data frame
dfAnn = pd.DataFrame()
for s in sessions:
    session_abs_time = pd.to_datetime(os.path.getmtime(s),unit='s')

    #1. reading data
    with zipfile.ZipFile(s) as z:
        # First look for annotation.json
        for filename in z.namelist():
            if not os.path.isdir(filename):
                if '.json' in filename:
                    with z.open(filename) as f:
                         data = json.load(f) 
                    if 'intervals' in data:
                        df2 = pd.concat([pd.DataFrame(data), 
                            json_normalize(data['intervals'])], 
                            axis=1).drop('intervals', 1)#
                        if (df2.applicationName.all()=='ManualAnnotations' or df2.applicationName.all()=='AutomaticAnnotations'):
                            df2 = df2.apply(pd.to_numeric, errors='ignore')
                            df2.columns = df2.columns.str.replace("annotations.", "")
                            df2.start = pd.to_datetime(df2.start, format='%H:%M:%S.%f') 
                            df2.start = pd.to_timedelta(df2.start.dt.strftime('%H:%M:%S.%f')) + session_abs_time
                            df2.end = pd.to_datetime(df2.end, format='%H:%M:%S.%f')  
                            df2.end = pd.to_timedelta(df2.end.dt.strftime('%H:%M:%S.%f')) + session_abs_time
                            df2['duration'] = (df2.end-df2.start) / np.timedelta64(1, 's')   
                            dfAnn = dfAnn.append(df2)                   
                    elif 'frames' in data:
                        df = pd.concat([pd.DataFrame(data), 
                            json_normalize(data['frames'])], 
                            axis=1).drop('frames', 1)
                        df.columns = df.columns.str.replace("_", "")
                        df['frameStamp'] = pd.to_datetime(df['frameStamp'], format='%H:%M:%S.%f')
                        df['frameStamp']= pd.to_timedelta(df['frameStamp'].dt.strftime('%H:%M:%S.%f')) + session_abs_time
                        appName = df.applicationName.all()
                        df.columns = df.columns.str.replace("frameAttributes", df.applicationName.all())
                        df = df.set_index('frameStamp').iloc[:,2:]
                        df = df[~df.index.duplicated(keep='first')]
                        df = df.apply(pd.to_numeric, errors='ignore')
                        df = df.select_dtypes(include=['float64','int64'])# Cast to float 
                        df = df.loc[:, (df.sum(axis=0) != 0)]
                        df.rename(columns=lambda x: re.sub('Kinect.\d','Kinect.',x),inplace=True)
                        #dfALL = dfALL.merge(df,how='outer',left_index=True,right_index=True)
                        dfALL = pd.concat([dfALL, df], ignore_index=False,sort=False).sort_index()
                   
df1 =  dfALL.apply(pd.to_numeric).fillna(method='bfill')

to_exclude = ['Ankle']
# Exclude features
for el in to_exclude:
    df1 = df1[[col for col in df1.columns if el not in col]]

masked_df = [
    df1[(df2_start <= df1.index) & (df1.index <= df2_end)]
    for df2_start, df2_end in zip(dfAnn['start'], dfAnn['end'])
]

# Feature aggregators
aggregations = ['max','min','std','mean','var','median']
features = []
for key in df1.columns.values:
    for a in aggregations:
        fname = key+'.'+a
        features.append(fname)
        if a == 'max':
            dfAnn[fname] = [np.max(dt[key]) if not dt.empty else None for dt in masked_df]
        elif a == 'min':
            dfAnn[fname] = [np.min(dt[key]) if not dt.empty else None for dt in masked_df]
        elif a == 'std':    
            dfAnn[fname] = [np.std(dt[key]) if not dt.empty else None for dt in masked_df]
        elif a == 'mean':    
            dfAnn[fname] = [np.mean(dt[key]) if not dt.empty else None for dt in masked_df]
        elif a == 'var':
            dfAnn[fname] = [np.var(dt[key]) if not dt.empty else None for dt in masked_df]
        elif a == 'median':  
            dfAnn[fname] = [np.median(dt[key]) if not dt.empty else None for dt in masked_df]
   
target_features = ['classRate','classDepth','classRelease']

for target in target_features:
    #Feauture ranking
    min_max_scaler = preprocessing.MinMaxScaler()
    X = dfAnn[features].values
    X = min_max_scaler.fit_transform(X)
    y = dfAnn[target].values.ravel()
    ETF = ExtraTreesClassifier()
    ETF.fit(X, y)
    
    importance = dict(zip(dfAnn[features].columns, ETF.feature_importances_))
    importance = sorted(importance.items(), key=operator.itemgetter(1),reverse=True)
    
    svc = SVC(kernel="linear", C=1)
    accuracies = []
    for n in range(1,len(importance)):
        training_feautres = []
        for el in importance[:n]:
            training_feautres.append(el[0])
        dfAnn[training_feautres+[target]]
        X = dfAnn[training_feautres].values
        X = min_max_scaler.fit_transform(X)
        y = dfAnn[target].values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=72)
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    fig = plt.figure()
    plt.plot(accuracies)
    fig.suptitle('SVM '+target+' N='+str(np.shape(dfAnn)[0]), fontsize=20)
    plt.xlabel('No. features', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)
    
'''
#Example plot correlation
import seaborn as sns
#sns.pairplot(dfAnn[features+['duration','classRate']],hue='classRate',palette=sns.color_palette('BuGn', 2))

#FFT
#sp = np.fft.rfft(signal)
#freq = np.fft.rfftfreq(signal.shape[-1]) # time sloth of histogram is 1 hour
#lt.plot(freq, np.log10(np.abs(sp) ** 2))
'''