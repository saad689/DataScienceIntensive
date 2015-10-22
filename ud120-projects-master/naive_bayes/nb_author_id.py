#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import numpy as np

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
gauss_nb = GaussianNB()
multi_nb = MultinomialNB()
berno_nb = BernoulliNB()

t0 = time()
gauss_nb.fit(features_train, labels_train)
print "Gaussian Training Time: ", round(time()-t0, 3), " s"

t0 = time()
multi_nb.fit(features_train, labels_train)
print "Multinomial Training Time: ", round(time()-t0, 3), " s"

t0 = time()
berno_nb.fit(features_train, labels_train)
print "Bernoulli Training Time: ", round(time()-t0, 3), " s"

t0 = time()
gauss_pred = gauss_nb.predict(features_test)
print "Gaussian Predict Time: ", round(time()-t0, 3), " s"
print "Gaussian Accuracy: ", gauss_nb.score(features_test, np.array(labels_test))

t0 = time()
multi_pred = multi_nb.predict(features_test)
print "Multinomial Predict Time: ", round(time()-t0, 3), " s" 
print "Multinomial Accuracy: ", multi_nb.score(features_test, np.array(labels_test))

t0 = time()
berno_pred = berno_nb.predict(features_test)
print "Bernoulli Predict Time: ", round(time()-t0, 3), " s"
print "Bernoulli Accuracy: ", berno_nb.score(features_test, np.array(labels_test))
#########################################################
#Results:
