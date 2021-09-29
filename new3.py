import pandas
import numpy as np
from gensim import models
import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from gensim import matutils
#from gensim.models.ldamodel import LdaModel
from sklearn import cross_validation
from sklearn import dummy

from sklearn import feature_extraction
from sklearn import grid_search
from sklearn import linear_model
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm as SVC
import xgboost

from sklearn.decomposition import LatentDirichletAllocation as LDA
data = pandas.read_csv("train.csv")
trainX = data.iloc[:, 1:]
yTrain = data.iloc[:, 0]
# print(yTrain)
#X = data.iloc[:, 2:]
#Y = data.iloc[:, 0]
#print(X.shape)
#print(Y.shape)
runBaseline = True
test=pandas.read_csv("dev.csv")
testX = test.iloc[:, 1:]
yTest = test.iloc[:, 0]
# print(yTest)
#trainX, testX, yTrain, yTest = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)

vectorizer = feature_extraction.text.TfidfVectorizer()
liwc_scaler = preprocessing.StandardScaler()
unigrams = vectorizer.fit_transform(trainX["text"].values.astype("U")).toarray()
# vectorizer1 = feature_extraction.text.TfidfVectorizer()
# synst=vectorizer1.fit_transform(trainX["synset"].values.astype('U')).toarray()
# tf_vectorizer =feature_extraction.text.CountVectorizer()
# tf = tf_vectorizer.fit_transform(trainX["text"]).toarray()
# tf_feature_names = tf_vectorizer.get_feature_names()
# lda = LDA(n_topics=10, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
# lda_train = lda.transform(tf)
liwc = liwc_scaler.fit_transform(trainX.ix[:, "TR":"OtherP"])
allf = np.hstack((unigrams,liwc))

unigrams_t = vectorizer.transform(testX["text"].values.astype("U")).toarray()
# tf_t = tf_vectorizer.transform(testX["text"]).toarray()
# lda_test = lda.transform(tf_t)
liwc_t = liwc_scaler.transform(testX.ix[:, "TR":"OtherP"])
# synst_t = vectorizer1.transform(testX["synset"].values.astype('U')).toarray()
allf_t = np.hstack((unigrams_t,liwc_t))

features = {"All_f_without_synset": (allf, allf_t)}
for f in features:
    xTrain = features[f][0]
    xTest = features[f][1]

    #print(indent("Features: ", 4), f)

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(XTrain, yTrain)
        predictions = grid.predict(xTest)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        print(indent("Test Accuracy: ", 8), metrics.accuracy_score(yTest, predictions))
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
        print()