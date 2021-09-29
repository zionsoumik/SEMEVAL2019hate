from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

X = ["Good morning", "Sweet Dreams", "Stay Awake"]
Y = ["Good morning", "Sweet Dreams", "Stay Awake"]

vectorizer = TfidfVectorizer().fit(X)

tfidf_vector_X = vectorizer.transform(X).toarray()  #//shape - (3,6)
tfidf_vector_Y = vectorizer.transform(Y).toarray() #//shape - (3,6)
tfidf_vector_X = tfidf_vector_X[:, :, None] #//shape - (3,6,1)
tfidf_vector_Y = tfidf_vector_Y[:, :, None] #//shape - (3,6,1)

X_train, X_test, y_train, y_test = train_test_split(tfidf_vector_X, tfidf_vector_Y, test_size = 0.2, random_state = 1)

from keras import Sequential
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(units=6, input_shape = X_train.shape[1:], return_sequences = True))
model.add(LSTM(units=6, return_sequences=True))
model.add(LSTM(units=6, return_sequences=True))
model.add(LSTM(units=1, return_sequences=True, name='output'))
model.compile(loss='cosine_proximity', optimizer='sgd', metrics = ['accuracy'])

print(model.summary())

model.fit(X_train, y_train, epochs=1, verbose=1)