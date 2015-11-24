# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 20:55:41 2015

@author: Saad Alam
"""

import csv
from sklearn import cross_validation
from sklearn import svm
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from preprocess_features import labels, features

#Open up an output file in which to store the results
resultFile = open('output_svm.csv', 'wb')
wr = csv.writer(resultFile, dialect='excel')
wr.writerow(['C', 'gamma', 'Accuracy'])

#Arrays of C and Gamma that will be experimented with
C_vals = [50, 100, 200, 300, 400, 500]
gamma_vals = [0.005, 0.007813, 0.01, 0.02, 0.03]

#Initialize rbf kernel
clf = svm.SVC(kernel='rbf')

#Shuffle and split data 5 times using 10% of data to train and 3.3% of data to test each time
rs = cross_validation.ShuffleSplit(len(labels), n_iter=5, train_size=0.3, test_size=0.09)
for train_index, test_index in rs:
    features_train, features_test, labels_train, labels_test = \
        features[train_index], features[test_index], labels[train_index], labels[test_index]
    for c in C_vals:
        g = 1./196
        clf.set_params(C=c)
        clf.set_params(gamma=g)
        clf = clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        wr.writerow([c, g, accuracy_score(labels_test, pred)])
    for g in gamma_vals:
        c = 1
        clf.set_params(C=c)
        clf.set_params(gamma=g)
        clf = clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        wr.writerow([c, g, accuracy_score(labels_test, pred)])
        
resultFile.close()
