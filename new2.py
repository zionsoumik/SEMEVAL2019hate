import re
import nltk
from sklearn import preprocessing

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

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
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
#from keras import backend as K

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
def review_to_wordlist( review, remove_stopwords=True):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()

    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (True by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    b=[]
    stemmer = english_stemmer #PorterStemmer()
    for word in words:
        b.append(stemmer.stem(word))

    # 5. Return a list of words
    return(b)
train = pd.read_csv( "train.csv", delimiter = ",")
test= pd.read_csv( "dev.csv", delimiter = ",")
clean_train_reviews = []
for review in train['text']:
    clean_train_reviews.append(" ".join(review_to_wordlist(review)))

clean_test_reviews = []
for review in test['text']:
    clean_test_reviews.append(" ".join(review_to_wordlist(review)))
#vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 200000, ngram_range = ( 1, 4 ),
                              #sublinear_tf = True )

#vectorizer = vectorizer.fit(clean_train_reviews)
#train_features = vectorizer.transform(clean_train_reviews)

#test_features = vectorizer.transform(clean_test_reviews)
max_features = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
maxlen = 80
batch_size = 32
nb_classes = 2
vectorizer = TfidfVectorizer()

#vectorizer = vectorizer.fit((train["text"].values.astype("U"))
# train_features = vectorizer.fit_transform(train["text"].values.astype("U"))
#
# test_features = vectorizer.transform(test["text"].values.astype("U"))
# X_train = train_features.toarray()
# X_test = test_features.toarray()
#
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)
y_train = np.array(train['HS'])
y_test = np.array(test['HS'])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


#pre-processing: divide by max and substract mean
# scale = np.max(X_train)
# X_train /= scale
# X_test /= scale
#
# mean = np.mean(X_train)
# X_train -= mean
# X_test -= mean

#input_dim = X_train.shape[1]
tokenizer = Tokenizer(nb_words=max_features)
tokenizer.fit_on_texts(train['text'])
liwc_scaler = preprocessing.StandardScaler()
liwc = liwc_scaler.fit_transform(train.ix[:, "TR":"OtherP"])
liwc_t = liwc_scaler.transform(test.ix[:, "TR":"OtherP"])
sequences_train = tokenizer.texts_to_sequences(train['text'])
sequences_test = tokenizer.texts_to_sequences(test['text'])
print('Pad sequences (samples x time)')
unigrams=sequence.pad_sequences(sequences_train, maxlen=maxlen)
unigrams_t=sequence.pad_sequences(sequences_test, maxlen=maxlen)
X_train = np.hstack((unigrams,liwc))
X_test = np.hstack((unigrams_t,liwc_t))

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training...")
#model.compile(optimizer='adam', loss='categorical_crossentropy', )
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,
          validation_data=(X_test, Y_test))
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)
print('prediction 7 accuracy: ', accuracy_score(test['HS'], preds))
