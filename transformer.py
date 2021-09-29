import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction
import numpy as np
import xgboost
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
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import gensim

#import scikitplot.plotters as skplt

import nltk

#from xgboost import XGBClassifier

import os

# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM
# from keras.utils.np_utils import to_categorical
# from keras.callbacks import ModelCheckpoint
# from keras.models import load_model
# from keras.optimizers import Adam
class MySentences(object):
    """MySentences is a generator to produce a list of tokenized sentences

    Takes a list of numpy arrays containing documents.

    Args:
        arrays: List of arrays, where each element in the array contains a document.
    """

    def __init__(self, *arrays):
        self.arrays = arrays

    def __iter__(self):
        for array in self.arrays:
            for document in array:
                for sent in nltk.sent_tokenize(document):
                    yield nltk.word_tokenize(sent)


def get_word2vec(sentences, location):
    """Returns trained word2vec

    Args:
        sentences: iterator for sentences

        location (str): Path to save/load word2vec
    """
    if os.path.exists(location):
        print('Found {}'.format(location))
        model = gensim.models.Word2Vec.load(location)
        return model

    print('{} not found. training model'.format(location))
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    print('Model done training. Saving to disk')
    model.save(location)
    return model
df_train = pd.read_csv(filepath_or_buffer="train.csv", sep=',')
df_test = pd.read_csv(filepath_or_buffer="dev.csv", sep=',')
w2vec = get_word2vec(
    MySentences(
        df_train['text'].values,
        #df_test['Text'].values  Commented for Kaggle limits
    ),
    'w2vmodel'
)
class MyTokenizer:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)

    def fit_transform(self, X, y=None):
        return self.transform(X)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.wv.syn0[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)

        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def fit_transform(self, X, y=None):
        return self.transform(X)
mean_embedding_vectorizer = MeanEmbeddingVectorizer(w2vec)
mean_embedded = mean_embedding_vectorizer.fit_transform(df_train['text'])
mean_embedded_t=mean_embedding_vectorizer.transform(df_test["text"])
print(mean_embedded)
liwc_scaler = preprocessing.StandardScaler()
liwc = liwc_scaler.fit_transform(df_train.ix[:, "TR":"OtherP"])
liwc_t = liwc_scaler.transform(df_test.ix[:, "TR":"OtherP"])
data_train = np.hstack((mean_embedded,liwc))
data_test = np.hstack((mean_embedded_t,liwc_t))
label_train = df_train.iloc[:, 0]
label_test = df_test.iloc[:, 0]

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
models = [BaggingClassifier(), tree.DecisionTreeClassifier(),LogisticRegression(),xgboost.XGBClassifier(),
          svm.SVC(kernel='linear', C=1), OutputCodeClassifier(BaggingClassifier()),
          OneVsRestClassifier(svm.SVC(kernel='linear'))]

model_names = ["Bagging with DT", "Decision Tree","LogisticRegression","XGB",
               "Linear SVM", "OutputCodeClassifier with Linear SVM", "OneVsRestClassifier with Linear SVM"]
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
    acc = accuracy_score(label_test, prediction)
    print("Accuracy Using", name, ": " + str(acc) + '\n')
    print(classification_report(label_test, prediction))
    print(confusion_matrix(label_test, prediction))

