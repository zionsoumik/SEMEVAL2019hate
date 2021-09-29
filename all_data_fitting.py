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
from sklearn import svm
import xgboost
from sklearn.decomposition import LatentDirichletAllocation as LDA
def indent(lines, amount, ch=' '):
    padding = amount * ch
    return padding + ('\n'+padding).join(lines.split('\n'))
models = [
          #naive_bayes.GaussianNB(),
          xgboost.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',silent=True, nthread=1),
          #linear_model.LogisticRegression(random_state=0),
          #svm.SVC(random_state=0, kernel='linear'),
         ]

clf_hyp = [
           dict(),
        #dict(clf__C=[10.]),
        #dict(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    #silent=True, nthread=1),
    dict(),
        #dict(clf__C=[.00001, .0001, .001, .01, .1, 1., 10.]),
          ]

results = {}
pred=[]
def main():
    data = pandas.read_csv("tweets_user.csv")

    #trainX = data.iloc[:, 1:]
    #yTrain = data.iloc[:, 0]
    #print(yTrain)
    X=data.iloc[:,2:]
    Y=data.iloc[:,0]
    print(X.shape)
    print(Y.shape)
    runBaseline = True
    #test=pandas.read_csv("dev_es.csv")
    #testX = test.iloc[:, 1:]
    #yTest = test.iloc[:, 0]
    #print(yTest)
    trainX, testX, yTrain, yTest = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)  #test train split

    vectorizer = feature_extraction.text.TfidfVectorizer()
    liwc_scaler = preprocessing.StandardScaler()
    unigrams = vectorizer.fit_transform(trainX["text"].values.astype("U")).toarray()
    #vectorizer1 = feature_extraction.text.TfidfVectorizer()
    #synst=vectorizer1.fit_transform(trainX["synset"].values.astype('U')).toarray()
    #tf_vectorizer =feature_extraction.text.CountVectorizer()
    #tf = tf_vectorizer.fit_transform(trainX["text"]).toarray()
    #tf_feature_names = tf_vectorizer.get_feature_names()
    #lda = LDA(n_topics=10, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    #lda_train = lda.transform(tf)
    liwc = liwc_scaler.fit_transform(trainX.ix[:, "TR":"OtherP"])
    allf = np.hstack((unigrams,))



    unigrams_t = vectorizer.transform(testX["text"].values.astype("U")).toarray()
    #tf_t = tf_vectorizer.transform(testX["text"]).toarray()
    #lda_test = lda.transform(tf_t)
    liwc_t = liwc_scaler.transform(testX.ix[:, "TR":"OtherP"])
    #synst_t = vectorizer1.transform(testX["synset"].values.astype('U')).toarray()
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
                grid = GridSearchCV(pipe, hyp, cv=10, n_jobs=6)  #grid search for best hyperparameters
                #print("hello")
                #print(sorted(grid.cv_results_.keys()))
                grid.fit(xTrain, yTrain)
                predictions = grid.predict(xTest)
                pred.append(predictions)
                print(indent(type(model).__name__, 6))
                print(indent("Best hyperparameters: ", 8), grid.best_params_)
                print(indent("Validation Accuracy: ", 8), grid.best_score_)
                print(indent("Test Accuracy: ", 8), metrics.accuracy_score(yTest, predictions))
                print(indent(metrics.classification_report(yTest, predictions), 8))

            else:
                grid = model
                grid.fit(xTrain, yTrain)
                #predictions = grid.predict(xTest)
                #predictions = grid.predict(xTest)
                pred.append(predictions)
                print(indent(type(model).__name__, 6))
                print(indent("Test Accuracy: ", 8), metrics.accuracy_score(yTest, predictions))
                print(indent(metrics.classification_report(yTest, predictions), 8))

        print()
    print()


#print(pred)
file=open("output.csv","w")
file.writelines(["%s\n" % item for item in pred])
if __name__ == '__main__':
    main()