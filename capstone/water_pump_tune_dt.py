# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 20:55:41 2015

@author: Saad Alam
"""

import csv
from sklearn import cross_validation
#from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from preprocess_features import labels, features

#Open up an output file in which to store the results
resultFile = open('output_dt.csv', 'wb')
wr = csv.writer(resultFile, dialect='excel')
wr.writerow(['min_sample_split', 'min_samples_leaf', 'Accuracy'])

#Arrays of C and Gamma that will be experimented with
min_splits = [2, 4, 8, 16, 32, 64, 128]
min_leafs = [1, 10, 100, 1000]

#Initialize rbf kernel
clf = DecisionTreeClassifier()

#Shuffle and split data 5 times using 10% of data to train and 3.3% of data to test each time
rs = cross_validation.ShuffleSplit(len(labels), n_iter=5, train_size=0.3, test_size=0.09)
for train_index, test_index in rs:
    features_train, features_test, labels_train, labels_test = \
        features[train_index], features[test_index], labels[train_index], labels[test_index]
    for c in min_splits:
        g = 1
        clf.set_params(min_samples_split=c)
        clf.set_params(min_samples_leaf=g)
        clf = clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        wr.writerow([c, g, accuracy_score(labels_test, pred)])
    for g in min_leafs:
        c = 2
        clf.set_params(min_samples_split=c)
        clf.set_params(min_samples_leaf=g)
        clf = clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        wr.writerow([c, g, accuracy_score(labels_test, pred)])
        
resultFile.close()
