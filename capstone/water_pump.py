# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 20:55:41 2015

@author: Saad Alam
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from preprocess_features import labels, features

clf = DecisionTreeClassifier()

#Generate KFold for cross validation
#kf = cross_validation.KFold(len(labels), n_folds=10)
#for train_index, test_index in kf:
#    features_train, features_test, labels_train, labels_test = \
#        features[train_index], features[test_index], labels[train_index], labels[test_index]
#    clf = clf.fit(features_train, labels_train)
#    pred = clf.predict(features_test)
#    print accuracy_score(labels_test, pred)
    

#Separate data into training and test set to test model
features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features, labels, test_size=0.33, random_state=15)

#Try out simple NBC using gps_height feature
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print accuracy_score(labels_test, pred)