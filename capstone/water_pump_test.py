# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 09:11:49 2015

@author: Saad
"""

from sklearn import cross_validation
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from preprocess_features import labels, features

#Initialize rbf kernel
#clf = svm.SVC(kernel='linear', C=100, gamma=2**-7)
clf = DecisionTreeClassifier()


##Separate data into training and test set to test model
features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features, labels, test_size=0.33, random_state=15)

#Try out simple NBC using gps_height feature
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print accuracy_score(labels_test, pred)