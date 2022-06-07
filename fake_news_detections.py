# -*- coding: utf-8 -*-

import pandas as pd
import keras
import numpy as np

x='/content/drive/MyDrive/fake-news/train.csv'
data=pd.read_csv(x)
data

dataframe=data[['text','label']]
dataframe1=dataframe.dropna()
dataframe1

dataframe1.isnull().sum()

import re
def preProcess_data(text):
   text = text.lower()
   new_text = re.sub('[^a-zA-z0-9\s]','',text)
   new_text = re.sub('rt', '', new_text)
   return new_text

dataframe1['text'] = dataframe1['text'].apply(preProcess_data)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataframe1.text,dataframe1.label,test_size=0.3,random_state=42)
print("shape of X_train",X_train.shape)
print("shape of X_test",X_test.shape)
print("shape of y_train",y_train.shape)
print("shape of X_test",y_test.shape)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X_train=cv.fit_transform(X_train)

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(X_train,y_train)

X_test=cv.transform(X_test)
nb.score(X_test,y_test)

print(dataframe1['text'].values)
print(dataframe1['label'].values)

