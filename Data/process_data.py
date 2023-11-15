# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:22:16 2023

@author: JF
"""

import pandas as pd
#Import data
df = pd.read_csv('./Telco data TC fix.csv')

##Convert 'Yes'/'No' columns to binary 1 or 0. Get dummies for others
to_bin = []
to_dum =[]
for col in df.select_dtypes(object).columns:
    if 'Yes' in df[col].unique() and 'No' in df[col].unique():
        to_bin.append(col)
    else:
        to_dum.append(col)

for col in to_bin:        
    df[col].replace('Yes',1, inplace = True, regex = True)
    df[col].replace('No',0, inplace = True, regex = True)
    
dums = pd.get_dummies(df[to_dum], drop_first=True)

#append to main dataframe
df.drop(to_dum, axis =1, inplace = True)
df = pd.concat((dums, df), axis =1)

##Prepare numeric variables for scaling in the pipe line.
def scaleCont(df):
    cols_to_scale = []
    cols_not_scale = []
    for col, item in df.iteritems():
        if df[col].max() > 1:
            cols_to_scale.append(col)
        else:
            cols_not_scale.append(col)
   
    return cols_to_scale , cols_not_scale

cols_to_scale , cols_not_scale = scaleCont(df.drop('Churn', axis=1))

#reorder data frame to conform with mapper output
reordered_cols = cols_to_scale.copy()
reordered_cols.extend(cols_not_scale)
reordered_cols.extend(['Churn'])

df = df[reordered_cols]

#export
df.to_csv(r'Telco_ML_ready.csv', index=False)

