#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import numpy as np
from sklearn import tree


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels

#function to create and train classifier
def create_clf(ft_train, lb_train, min_samples):
    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples)
    t0 = time()
    clf = clf.fit(ft_train, lb_train)
    t = round(time()-t0, 3)
    return clf, t
    
for pct in [10, 1]:    
    features_train, features_test, labels_train, labels_test = preprocess(pct)
    
    #########################################################
    ### your code goes here ###
    print "Select Percentile = ", pct
    clf_dt_40, t_train = create_clf(features_train, labels_train, 40)
    print "The number of features in the data: ", len(features_train[0])
    print "Accuracy of this decision tree classifier with min_samples = 40: ", clf_dt_40.score(features_test, labels_test)
    print "Amount of time necessary to train data: ", t_train

#########################################################


print "SelectPercentile appears to be setting the number of features in the dataset that are used to classify. A larger value for \
the percentile leads to more features which in turn leads to a more complex decision tree."