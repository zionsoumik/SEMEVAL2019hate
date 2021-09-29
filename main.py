import pandas
from sklearn import cross_validation
#from sklearn.model_selection import train_test_split
import numpy as np
from gensim import models
import pandas
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
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
data=pandas.read_csv("train.csv")
tknzr = TweetTokenizer()
#tfidf_transformer = TfidfTransformer()
def dummy_fun(doc):
    return doc

def indent(lines, amount, ch=' '):
    padding = amount * ch
    return padding + ('\n'+padding).join(lines.split('\n'))

models = [
          naive_bayes.GaussianNB(),
          linear_model.LogisticRegression(random_state=0),
          svm.SVC(random_state=0, kernel='linear'),
         ]

clf_hyp = [
           dict(),
        dict(),
        dict(),
        #dict(clf__C=[.00001, .0001, .001, .01, .1, 1., 10.]),
          ]

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)
runBaseline = True
tweetx=data["text"].apply(tknzr.tokenize)


listoftokensx=list(tweetx)
#tfidf.fit(listoftokensx)
trainX = data.iloc[:, 1:]
yTrain = data.iloc[:, 0]
#trainX, testX, yTrain, yTest = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)  #test train split
unigrams = tfidf.fit_transform(listoftokensx).toarray()
#gnb = GaussianNB()
allf = np.hstack((unigrams,))




test=pandas.read_csv("dev.csv")
testX= test.iloc[:, 1:]
yTest = test.iloc[:, 0]

tweety=test["text"].apply(tknzr.tokenize)
#listoftokensy=list(tweety)
#tfidf.fit(listoftokensy)
unigrams_t = tfidf.transform(testX["text"]).toarray()

allf_t = np.hstack((unigrams_t,))

features = {"All_f_without_synset":(allf,allf_t)}

for f in features:
    xTrain = features[f][0]
    xTest = features[f][1]

    if runBaseline:
        baseline = dummy.DummyClassifier(strategy='most_frequent', random_state=0)
        baseline.fit(xTrain, yTrain)
        predictions = baseline.predict(xTest)

        print(indent("Baseline: ", 4))
        print(indent("Test Accuracy: ", 4), metrics.accuracy_score(yTest, predictions))
        print(indent(metrics.classification_report(yTest, predictions), 4))
        print()
        runBaseline = False

    print(indent("Features: ", 4), f)

    for m, model in enumerate(models):
        hyp = clf_hyp[m]
        pipe = pipeline.Pipeline([('clf', model)])

        if len(hyp) > 0:
            grid = grid_search.GridSearchCV(pipe, hyp, n_jobs=6)  # grid search for best hyperparameters
            grid.fit(xTrain, yTrain)
            predictions = grid.predict(xTest)

            print(indent(type(model).__name__, 6))
            print(indent("Best hyperparameters: ", 8), grid.best_params_)
            print(indent("Validation Accuracy: ", 8), grid.best_score_)
            print(indent("Test Accuracy: ", 8), metrics.accuracy_score(yTest, predictions))
            print(indent(metrics.classification_report(yTest, predictions), 8))

        else:
            grid = model
            grid.fit(xTrain, yTrain)
            predictions = grid.predict(xTest)

            print(indent(type(model).__name__, 6))
            print(indent("Test Accuracy: ", 8), metrics.accuracy_score(yTest, predictions))
            print(indent(metrics.classification_report(yTest, predictions), 8))

    print()
print()