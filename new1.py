# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 14:31:40 2017
Authors: K132047 | Sahir

Data Science Project
Code:   Presents a Comparision of Different Classifiers and
        Applies Multi-Layer Perceptron Classifier on the UCI
        Poker Hand Data Set
"""
# -------------------------------------------------------------------------
# All the Libraries:
# -------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import ensemble
from gensim.sklearn_api.phrases import PhrasesTransformer
import pandas as pd
from xgboost import XGBClassifier
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

# ----------------------------------------------------------------
# Read the Training and Testing Data:
# ----------------------------------------------------------------
array_train = pd.read_csv(filepath_or_buffer="train_es.csv", sep=',')
array_test = pd.read_csv(filepath_or_buffer="test_es.csv", sep=',')
# ----------------------------------------------------------------
# Print it's Shape to get an idea of the data set:
# ----------------------------------------------------------------
#print(data_train.shape)
#print(data_test.shape)
# ----------------------------------------------------------------
# Prepare the Data for Training and Testing:
# ----------------------------------------------------------------
# Ready the Train Data
#array_train = data_train.values
def convert(texts):
    m = []
    for t in texts:
        s = ""
        for w in t:
            s = s + w + " "
        m.append(s)
    return m
data_train = array_train.iloc[:, 5:]
label_train = array_train.iloc[:, 4]
# Ready the Test Data
#array_test = data_test.values
data_test = array_test.iloc[:, 2:]
label_test = array_test.iloc[:, 4]
# ----------------------------------------------------------------
# Scaling the Data for our Main Model
# ----------------------------------------------------------------
# Scale the Data to Make the NN easier to converge
liwc_scaler = preprocessing.StandardScaler()
vectorizer = feature_extraction.text.TfidfVectorizer()
#liwc_scaler = preprocessing.StandardScaler()
phr=PhrasesTransformer(min_count=1, threshold=3)
phrases=phr.fit_transform(array_train["text"].values.astype("U").tolist())

unigrams = vectorizer.fit_transform(array_train["text"].values.astype("U")).toarray()
print(unigrams.shape)
liwc = liwc_scaler.fit_transform(data_train.ix[:, "WC":"OtherP"])
phrases_t=phr.transform(array_test["text"].values.astype("U").tolist())
# Fit only to the training data
unigrams_t = vectorizer.transform(array_test["text"].values.astype("U")).toarray()
print(unigrams_t.shape)
# Transform the training and testing data
liwc = liwc_scaler.fit_transform(data_train)
liwc_t= liwc_scaler.transform(data_test)
data_train = np.hstack((unigrams,liwc))
data_test = np.hstack((unigrams_t,liwc_t))
#data_train = scaler.transform(data_train)
#data_test = scaler.transform(data_test)
# ----------------------------------------------------------------
# Apply the MLPClassifier:
# ----------------------------------------------------------------
# acc_array = [0] * 5
# for s in range(1, 6):
#     # Init MLPClassifier
#     clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 64),
#                         activation='tanh', learning_rate_init=0.02, max_iter=2000, random_state=s)
#     # Fit the Model
#     result = clf.fit(data_train, label_train)
#     # Predict
#     prediction = clf.predict(data_test)
#     # Get Accuracy
#     acc = accuracy_score(label_test, prediction)
#     # Store in the Array
#     acc_array[s - 1] = acc
#     # ----------------------------------------------------------------
#     # Fetch & Print the Results:
#     # ----------------------------------------------------------------
#     print(classification_report(label_test, prediction))
#     print("Accuracy using MLPClassifier and Random Seed:", s, ":", str(acc))
#     print(confusion_matrix(label_test, prediction))
# print("Mean Accuracy using MLPClassifier Classifier: ", np.array(acc_array).mean())
# ----------------------------------------------------------------
# Init the Models for Comparision
# ----------------------------------------------------------------
models = [
    svm.SVC(kernel="linear",C=1)]#, OutputCodeClassifier(BaggingClassifier()),
          # OneVsRestClassifier(svm.SVC(kernel='linear'))]
# clf1=ensemble.GradientBoostingClassifier(n_estimators = 900, learning_rate = 0.0008, loss = 'exponential', min_samples_split = 3, min_samples_leaf = 2, max_features ='sqrt', max_depth = 3,  random_state = 42, verbose = 1)
# clf2= LogisticRegression()
# clf3= svm.SVC(kernel='linear', C=1, probability=True)
# model_names = ["Votinghard", "Voting soft"]
# eclf1 = VotingClassifier(estimators=[('gbm', clf1), ('lr', clf2), ('svc', clf3)], voting='hard')
# eclf2 = VotingClassifier(estimators=[('gbm', clf1), ('lr', clf2), ('svc', clf3)], voting='soft')
# models = [eclf1,eclf2]

model_names = ["svm"]
# ----------------------------------------------------------------
# Run Each Model
# ----------------------------------------------------------------
for model, name in zip(models, model_names):
    model.fit(data_train, label_train)
    # Display the relative importance of each attribute
    if name == "Random Forest":
        print(model.feature_importances_)
        # Predict
    prediction = model.predict(data_test)
    # Print Accuracy
    numpy.savetxt("foo5.csv", prediction, delimiter=",")
    #acc = accuracy_score(label_test, prediction)
    #print("Accuracy Using", name, ": " + str(acc) + '\n')
    #print(classification_report(label_test, prediction))
    #print(confusion_matrix(label_test, prediction))
