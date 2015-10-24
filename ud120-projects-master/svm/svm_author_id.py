#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
svm_lin = svm.SVC(kernel='linear')

t0 = time()
svm_lin.fit(features_train, labels_train)
print "Time to train linear SVM: ", round(time()-t0, 3)

t0 = time()
score_svm_lin = svm_lin.score(features_test, labels_test)
print "Time to predict with linear SVM: ", round(time()-t0, 3)
print "Accuracy of Linear Kernel SVM: ", score_svm_lin

#Create smaller data and run SVM with linear kernel
features_train_small = features_train[:len(features_train)/100]
labels_train_small = labels_train[:len(labels_train)/100]


svm_lin_sm = svm.SVC(kernel='linear')

t0 = time()
svm_lin_sm.fit(features_train_small, labels_train_small)
print "Time to train linear SVM with smaller training sample: ", round(time()-t0, 3)

t0 = time()
score_svm_lin_small = svm_lin_sm.score(features_test, labels_test)
print "Time to predict with linear SVM with small training set: ", round(time()-t0, 3)
print "Accuracy of Linear Kernel SVM trained with smaller size: ", score_svm_lin_small

#Set SVM with a kernel of 'rbf' with smaller training data
svm_lin_sm.set_params(kernel='rbf')
t0 = time()
svm_lin_sm.fit(features_train_small, labels_train_small)
print "Time to train RBF Kernel SVM with smaller training sample: ", round(time()-t0, 3)

t0 = time()
score_svm_lin_small = svm_lin_sm.score(features_test, labels_test)
print "Time to predict with RBF SVM with small training set: ", round(time()-t0, 3)
print "Accuracy of RBF Kernel SVM trained with smaller size: ", score_svm_lin_small

#Check accuracy performance for different values of C
for c in [10, 100, 1000, 10000]:
    svm_lin_sm.set_params(C=c)
    svm_lin_sm.fit(features_train_small, labels_train_small)
    score_svm_lin_small = svm_lin_sm.score(features_test, labels_test)
    print "C = ", c, " Accuracy of RBF Kernel SVM trained with smaller size: ", score_svm_lin_small

print "The largest value of C gives the best accuracy. This corresponds to a more \
    complex decision boundary as it sacrifices a smooth decision surface for greater\
    accuracy."

#With the optimized C parameter of C=10000 now train on the full set with an RBF kernel
svm_lin_sm.fit(features_train, labels_train)
score_svm_rbf = svm_lin_sm.score(features_test, labels_test)
pred_svm_rbf = svm_lin_sm.predict(features_test)
print "Accuracy of RBF Kernel with optimized C of 10000: ", score_svm_rbf

print "Element 10 is classified as ", pred_svm_rbf[10]
print "Element 26 is classified as ", pred_svm_rbf[26]
print "Element 50 is classified as ", pred_svm_rbf[50]

print "\nThe number of emails predicted as Chris with an RBF Kernel \
    and C = 10000 is ", pred_svm_rbf.sum()
#########################################################


