"""
This module contains classes and helper functions
for running various types of neural networks.
TODO: n-grams?
"""

import sqlite3
import pickle
from sqlalchemy import create_engine

import numpy as np
import pandas as pd
from keras.initializers import Constant
from keras.layers import (
    Activation,
    Conv1D,
    CuDNNLSTM,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    MaxPooling1D,
)
from keras.models import Model, Sequential, load_model
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sqlalchemy import create_engine
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

# Notes:
# Word sense disambiguation
# Stanford CoreNLP
# Wordnet - Word Sense
# SentiWordNet
# LDA / TopicRNN


INPUT_GLOVE = "C:/Users/John/Downloads/glove.6B.100d.txt"


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def import_data(datasets):
    """
    This function accepts an array of file paths to datasets and
    returns a DataFrame of thsoe datasets, appended if multiple.
    """
    df = pd.DataFrame()
    db_name = ""
    assert type(datasets).__name__ == "list", "Please enter a list"
    for dataset in datasets:
        data_type = dataset[dataset.rfind(".") :]
        db_name_temporary = dataset[dataset.rfind("/") + 1 : dataset.rfind(".")]
        if data_type == ".db":
            cnx = sqlite3.connect(dataset)
            df_temporary = pd.read_sql_query(f"SELECT * FROM {db_name_temporary}", cnx)
        elif data_type == ".csv":
            df_temporary = pd.read_csv(dataset)
        print(db_name_temporary, df_temporary.shape)
        df = df.append(df_temporary, ignore_index=True, sort=False)
        db_name += db_name_temporary + "_"
    print(db_name[:-1], df.shape)
    df.text2 = df.text2.astype(str)
    return df, db_name[:-1]


def convert_data(datasets):
    assert type(datasets).__name__ == "list", "Please enter a list"
    for dataset in datasets:
        data_type = dataset[dataset.rfind(".") :]
        db_name_temporary = dataset[dataset.rfind("/") + 1 : dataset.rfind(".")]
        if data_type == ".db":
            cnx = sqlite3.connect(dataset)
            df_temporary = pd.read_sql_query(f"SELECT * FROM {db_name_temporary}", cnx)
            df_temporary.to_csv(f"{db_name_temporary}.csv")
        elif data_type == ".csv":
            df_temporary = pd.read_csv(dataset)
            df_temporary.to_sql(
                db_name_temporary,
                con=create_engine(f"sqlite:///{db_name_temporary}.db", echo=False),
                if_exists="replace",
            )


# MLP


class MLP:
    def __init__(self, datasets, split=0.2, batch_size=128, epochs=10, num_splits=1):
        self.datasets = datasets
        self.split = split
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_splits = num_splits

    def clean_data(self, df, sparse=True):
        # X, y = df["text2"], df["label"]
        # tokenizer = Tokenizer(num_words=5000)
        # tokenizer.fit_on_texts(X)
        # X_counts = tokenizer.texts_to_matrix(X, mode="tfidf")
        # if not sparse:
        #     other_vars = df[
        #         df.columns.difference(
        #             ["index", "text", "text1", "text2", "level_0", "Unnamed: 0"]
        #         )
        #     ]
        #     X_counts = pd.DataFrame.to_dense(X_counts)
        #     X_counts = pd.DataFrame(X_counts)
        #     X_counts = pd.concat((X_counts, other_vars), axis=1)
            # X_counts = pd.DataFrame.to_sparse(X_counts)
        X, y = df[df.columns.difference(["index", "text", "text1", "text2", "level_0", "Unnamed: 0"])], df["label"]
        output_shape = len(y.unique())
        print(output_shape)
        if output_shape > 2:
            y_cats = to_categorical(y, output_shape)
        X_train, X_test, y_train, y_test = train_test_split(
            # X_counts,
            X,
            y_cats if output_shape > 2 else y,
            test_size=self.split,
            random_state=42,
        )

        input_shape = X_train.shape[1]
        print(input_shape, output_shape)
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            input_shape,
            output_shape,
            0,
            # tokenizer,
            sparse,
        )

    def mlp(self, num_layer, num_node, num_dropout, input_shape, output_counts):
        model = Sequential()
        model.add(Dense(num_node, input_shape=(input_shape,), activation="relu"))
        model.add(Dropout(num_dropout))
        for layer in range(num_layer):
            model.add(Dense(num_node, activation="relu"))
            model.add(Dropout(num_dropout))

        if output_counts > 2:
            model.add(Dense(output_counts, activation="softmax"))
            model.compile(
                loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
        else:
            model.add(Dense(1, activation="sigmoid"))
            model.compile(
                loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
            )

        return model

    def train(self, model, X_train, y_train, X_test, y_test, k_folds):
        if k_folds:
            history = model.fit(
                X_train, y_train, epochs=self.epochs, batch_size=self.batch_size
            )
        else:
            # tensorboard = TensorBoard(log_dir=f"./logs/{name}")
            early_stopping = EarlyStopping(monitor="val_loss")
            # reduce = ReduceLROnPlateau(monitor='val_loss', factor=.2, patience=3, min_lr=.001)
            history = model.fit(
                X_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.split,
                callbacks=[early_stopping],
            )
        score = model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print(score)
        return history, score

    def mlp_wrapper(self):
        df, db_name = import_data(self.datasets)
        X_train, X_test, y_train, y_test, input_shape, output_shape, tokenizer, sparse = self.clean_data(
            df, sparse=False
        )

        num_layers = [1, 2]
        num_nodes = [32, 64]
        num_dropouts = [0, 20]
        count = 1
        total = len(num_layers) * len(num_nodes) * len(num_dropouts)
        best_score = 0
        best_model = ""
        best_model_name = ""

        for num_layer in num_layers:
            for num_node in num_nodes:
                for num_dropout in num_dropouts:
                    name = f"{num_layer}_layer_{num_node}_node_{num_dropout}%_dropout_{db_name}"
                    print(name, f"{count} out of {total}")
                    count += 1
                    if self.num_splits > 1:
                        skf = StratifiedKFold(n_splits=self.num_splits, shuffle=True)
                        for i, (train_indices, val_indices) in enumerate(
                            skf.split(X_train, y_train)
                        ):
                            print(train_indices, val_indices, type(train_indices))
                            print(f"Training on fold {i + 1} out of {self.num_splits}")
                            X_train_split, X_test_split = (
                                X_train[train_indices],
                                X_train[val_indices],
                            )
                            y_train_split, y_test_split = (
                                y_train[train_indices],
                                y_train[val_indices],
                            )

                            model = None
                            model = self.mlp(
                                num_layer,
                                num_node,
                                num_dropout,
                                input_shape,
                                output_shape,
                            )

                            history, score = self.train(
                                model,
                                X_train_split,
                                y_train_split,
                                X_test_split,
                                y_test_split,
                                True,
                            )

                            if score[1] > best_score:
                                best_score = score[1]
                                best_model = model
                                best_model_name = name
                    else:
                        model = self.mlp(
                            num_layer, num_node, num_dropout, input_shape, output_shape
                        )

                        history, score = self.train(
                            model, X_train, y_train, X_test, y_test, False
                        )

                        if score[1] > best_score:
                            best_score = score[1]
                            best_model = model
                            best_model_name = name

        if sparse:
            best_model.save(
                f"C:/Users/John/Downloads/models/{best_model_name}_{round(best_score, 3)}_sparse.h5"
            )
        else:
            best_model.save(
                f"C:/Users/John/Downloads/models/{best_model_name}_{round(best_score, 3)}_dense.h5"
            )
        with open(
            f"C:/Users/John/Downloads/models/tokenizer_{db_name}.pickle", "wb"
        ) as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"{best_model_name}_{round(best_score, 3)}")

    def test_mlp(self, model_loc, tokenizer_loc, datasets):
        df, _ = import_data(datasets)
        model = load_model(model_loc)
        tokenizer = Tokenizer()
        with open(tokenizer_loc, "rb") as handle:
            tokenizer = pickle.load(handle)
        X, y = df[df.columns.difference(["index", "text", "text1", "text2", "level_0", "Unnamed: 0"])], df["label"]
        # X, y = df["text2"], df["label"]
        # X_counts = tokenizer.texts_to_matrix(X, mode="tfidf")
        if len(y.unique()) > 2:
            y_cats = to_categorical(y, len(y.unique()))
        # print(model.predict(X_counts))
        print(
            model.evaluate(
                X_counts,
                y if len(y.unique()) <= 2 else y_cats,
                batch_size=self.batch_size,
            )
        )


"""
GloVe
"""


class Glove:
    def __init__(self, split=0.2, batch_size=32, epochs=5, embedding_dim=100):
        self.split = split
        self.batch_size = batch_size
        self.epochs = epochs
        self.embedding_dim = embedding_dim

    def clean_data(self, df, sent=False):
        X, y = df["text"], df["label"]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)
        X_counts = tokenizer.texts_to_sequences(X)
        X_counts = pad_sequences(X_counts, maxlen=100)
        print(len(X_counts[1]), len(X_counts[0]))

        if sent:
            sentiment = df["sentiment"]
            X_counts = pd.DataFrame.to_dense(X_counts)
            X_counts = pd.DataFrame(X_counts)
            X_counts = pd.concat((X_counts, sentiment), axis=1)
            # X_counts = pd.DataFrame.to_sparse(X_counts)

        value_counts = len(y.unique())
        if value_counts > 2:
            y_cats = to_categorical(y, value_counts)
        X_train, X_test, y_train, y_test = train_test_split(
            X_counts,
            y_cats if value_counts > 2 else y,
            test_size=split,
            random_state=42,
        )

        word_index = tokenizer.word_index
        input_shape = len(word_index)
        print(X_counts.shape, y_cats.shape if value_counts > 2 else y.shape)
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            input_shape,
            tokenizer,
            value_counts,
            sent,
            word_index,
        )

    def embedding(self):
        with open("C:/Users/John/Downloads/embeddings_index.pickle", "rb") as handle:
            embeddings_index = pickle.load(handle)
        df, db_name = import_multiple(import_stream(), import_russian(1))
        X_train, X_test, y_train, y_test, input_shape, tokenizer, value_counts, sent, word_index = self.clean_data(
            df
        )
        early_stopping = EarlyStopping(monitor="val_loss", patience=1)

        embedding_matrix = np.zeros((input_shape, embedding_dim))
        for word, i in word_index.items():
            try:
                embedding_vector = embeddings_index.get(bytes(word, "utf-8"))
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            except:
                pass

        embedding_layer = Embedding(
            input_shape,
            embedding_dim,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=100,
            trainable=False,
        )

        sequence_input = Input(shape=(100,), dtype="int32")
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation="relu")(embedded_sequences)
        x = Dropout(0.2)(x)
        x = MaxPooling1D(5)(x)
        # x = Conv1D(128, 5, activation="relu")(x)
        # x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation="relu")(x)
        preds = Dense(
            value_counts if value_counts > 2 else 1,
            activation="softmax" if value_counts > 2 else "sigmoid",
        )(x)

        model = Model(sequence_input, preds)
        model.compile(
            loss="categorical_crossentropy"
            if value_counts > 2
            else "binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.split,
            callbacks=[early_stopping],
        )
        score = model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print(score[1])
        model.save("C:/Users/John/Downloads/models/glove_embeddings_test.h5")

    def glove(self, INPUT_GLOVE):
        embeddings_index = {}
        with open(INPUT_GLOVE, "rb") as handle:
            for line in handle:
                try:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype="float32")
                    embeddings_index[word] = coefs
                except:
                    pass
        print(f"Loaded {len(embeddings_index)} word vectors.")
        with open(
            "C:/Users/John/Documents/GitHub/233bot/embeddings_index.txt", "wb"
        ) as handle:
            pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    mlp = MLP(["./tweets_users_liwc.csv", "./reddit_posts_liwc.csv"])
    mlp.mlp_wrapper()