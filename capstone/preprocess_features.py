# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 09:04:24 2015

@author: Saad Alam
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

#Convert training features into matrix and labels into array
df_feat = pd.read_csv('features_training.csv')
df_labels = pd.read_csv('labels_training.csv')
labels = np.array(df_labels.status_group)

#Modfiy any dataframe columns as needed
df_feat.loc[df_feat['construction_year'] == 0, 'construction_year'] = 1987 #This line changes all construction years that are equal to 0 to 1987
#df_feat.loc[df_feat['population'] == 0, 'population'] = 282

categories = ['extraction_type', 'waterpoint_type', 'region_code', 'source_type', 
              'water_quality', 'permit', 'public_meeting', 'management', 'management_group', 
              'payment', 'source_class', 'basin', 'scheme_management',
              'district_code']
categories_large = ['funder', 'lga', 'ward', 'subvillage']
#continuous_feat = ['construction_year', 'population', 'amount_tsh', 'gps_height']
continuous_feat = ['construction_year']

def binarize_cat(feature):
    le = preprocessing.LabelEncoder()
    enc = preprocessing.OneHotEncoder()
    arr = le.fit_transform(df_feat[feature].fillna('unknown'))
    arr = enc.fit_transform(arr.reshape((len(arr),1))).toarray()
    return arr

def minmaxscale(feature):
    minmax = preprocessing.MinMaxScaler(copy=False)
    arr = minmax.fit_transform(np.array(df_feat[feature]).astype(float, copy=False))
    return arr
    
def binarize(feature):
    minmax = preprocessing.Binarizer(copy=False)
    minmax.set_params(threshold=(df_feat[feature].mean()))
    arr = minmax.fit_transform(np.array(df_feat[feature]).astype(float, copy=False))
    return arr.T
    
def binarize_largecat(feature):
    arr = le.fit_transform(df_feat[feature].fillna('unknown')).astype(int, copy=False)
    num_bits = str(len(format(max(arr), 'b')))
    new = []
    for i in list(range(len(arr))):
        temp = format(arr[i],'0' + num_bits + 'b')
        new.append([float(i) for i in temp])
    return new
    
#Preprocess label values into numbers
le = preprocessing.LabelEncoder()
le.fit(labels)
labels=le.transform(labels)

#Get categorical features
features = binarize_cat('quantity')
for feature in categories:
    features = np.column_stack((features, binarize_cat(feature)))
    
#for feature in categories_large:
#    features = np.column_stack((features, binarize_largecat(feature)))

#Get all non categorical features of interest
for feature in continuous_feat:
    features = np.column_stack((features, binarize(feature)))
