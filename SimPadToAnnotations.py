# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:09:35 2018

@author: ddm
"""

#import numpy as np
import pandas as pd
import numpy as np
import zipfile
import json
import xml.etree.ElementTree as ET

session_zip = 'bls_session_Ddm_2018-07-16-16-36-18.ssx'
json_file_output = '_' + session_zip.replace('.ssx','.json')

event_file = 'CPR/CPREvents.xml'
colnames = ['startTime','endTime','compDepth','compPeakTime','compInstantaneousPeriod','compAbsolutePeakDepth','compLeaningDepth','compReleaseDepth','compHandposXiphoid','compHandposError','compIncompleteRelease','compMeanRate']
df = pd.DataFrame(columns=colnames)
#pd.options.display.float_format = lambda x : '{:.3f}'.format(x) if int(x) == x else '{:,.3f}'.format(x)


df.style.format({'start': '{:.3f}', 'end':'{:.3f}'})
#1. reading data
with zipfile.ZipFile(session_zip) as z:
    
    f = z.open(event_file)
    e = ET.parse(f).getroot()
    
    for evt in e.findall('evt'):
        #print evt
        r = np.empty(0) 
        if evt.get('type')=='CprCompEvent':
            r = np.append(r,[evt.get('msecs')])
            for param in evt.findall('param'):
                r = np.append(r,[param.get('value')]) 
            #print np.size(r)
            df.loc[len(df)] = r
    df[['startTime','endTime']] = df[['startTime','endTime']].astype(dtype='float')/1000
    df[['startTime','endTime']] = df[['startTime','endTime']].astype(dtype='str')
    df.rename(columns={'startTime':'start','endTime':'end'}, inplace=True)
    df[['compPeakTime','compInstantaneousPeriod','compHandposXiphoid','compHandposXiphoid','compHandposError','compIncompleteRelease','compMeanRate']].astype(dtype='int')   
    df[['compDepth','compAbsolutePeakDepth','compLeaningDepth','compReleaseDepth']].astype(dtype='float')   
    #df[['startTime','endTime','compDepth','compReleaseDepth','compMeanRate']]
    
    df['classRelease'] = np.where((df['compDepth']>='50') & (df['compDepth']<='60'), '1', '0')
    df['classDepth'] = np.where(df['compReleaseDepth']>'5', '1', '0')
    df['classRate'] = np.where((df['compMeanRate']>='100') & (df['compMeanRate']<='120'), '1', '0') 
    
    newdf = df[['start','end','compDepth','compReleaseDepth','compMeanRate','classRelease','classDepth','classRate']]
    df_json_pretty = json.loads(newdf.to_json(orient='records'))
    metadata = {"recordingID": session_zip.replace('.ssx',''),"applicationName":"AutomaticAnnotations","intervals":df_json_pretty}
    with open(json_file_output, 'w') as outfile:
        json.dump(metadata, outfile,indent=4)
