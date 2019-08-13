# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:26:38 2019

@author: Shivangi Sharma
"""

import pandas as pd
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
import pickle


#load data
dataset=pd.read_csv("project.csv")

#hange columns

dataset.columns = [' Diagnosis',' Forced vital capacity',' forced expiration',
               ' Performance status','Pain before surgery',' Haemoptysis before surgery',
               ' Dyspnoea before surgery ',' Cough before surgery',
               ' Weakness before surgery',' T in clinical TNM',
               ' Type 2 DM',' MI up to 6 months ','PAD',' Smoking',' Asthma','AGE','Risk1Y']


#featutres,labels


array = dataset.values
features = array[:,0:16]
labels = array[:,16]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features,
labels, test_size = 0.25, random_state = 0)

smt = SMOTE()
features_train, labels_train= smt.fit_sample( features_train, labels_train)


seed=7
num_trees = 100
from sklearn.ensemble import RandomForestClassifier
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, features_train, labels_train, cv=kfold)
print(results.mean())
model.fit(features_train, labels_train)  

labels_pred = model.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)


from sklearn.metrics import accuracy_score
accuracy_score(labels_pred,labels_test)


pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))














