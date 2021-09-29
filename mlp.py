import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')
from sklearn import preprocessing

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import six
from abc import ABCMeta
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers import Convolution1D
from keras import backend as K

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
vectorizer = TfidfVectorizer()
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2
max_features = 8000
batch_size = 32
nb_classes = 2
train = pd.read_csv( "liwc_input.csv", delimiter = ",")
test= pd.read_csv( "liwc_test.csv", delimiter = ",")
#vectorizer = vectorizer.fit((train["text"].values.astype("U"))
unigrams = vectorizer.fit_transform(train["text"].values.astype("U")).toarray()
liwc_scaler = preprocessing.StandardScaler()
unigrams_t = vectorizer.transform(test["text"].values.astype("U")).toarray()
liwc = liwc_scaler.fit_transform(train.loc[:, "WC":"OtherP"])

liwc_t = liwc_scaler.transform(test.loc[:, "WC":"OtherP"])
X_train = np.hstack((unigrams,liwc))
X_test = np.hstack((unigrams_t,liwc_t))


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(train['rating'])
y_test = np.array(test['rating'])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

input_dim = X_train.shape[1]
#pre-processing: divide by max and substract mean
model = Sequential()
# model.add(Dense(256, input_dim=input_dim))
# model.add(Activation('relu'))
# model.add(Dropout(0.4))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))
#
# # we'll use categorical xent for the loss, and RMSprop as the optimizer
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', show_accuracy=True)
#
# print("Training...")
# model.fit(X_train, Y_train, nb_epoch=5, batch_size=16, validation_split=0.1)
#
# print("Generating test predictions...")
# preds = model.predict_classes(X_test, verbose=0)
# print('prediction 6 accuracy: ', accuracy_score(test['HS'], preds))
model.add(Embedding(max_features, 128, dropout=0.2))
# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))

def max_1d(X):
    return K.max(X, axis=1)

model.add(Lambda(max_1d, output_shape=(nb_filter,)))
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Train...')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,
          validation_data=(X_test, Y_test))
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)
print('prediction 8 accuracy: ', accuracy_score(test['rating'], preds))
