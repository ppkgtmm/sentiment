import pandas as pd
import re
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pickle
import keras
import tensorflow as tf

class PolarityModel:

    reserved_words = ["EMAIL", "NUMBER", "MENTION", "URL"]
    tokenizer = None
    model = None

    def __init__(self, **args):
        self.max_features = args.get("max_features", 50000)
        self.max_len = args.get("max_len", 2700)
        self.train_size = args.get("train_size", 0.8)
        self.seed = args.get("seed", 42)
        self.epoch = args.get("epoch", 4)
        self.batch_size = args.get("batch_size", 128)

    def read_data(self, **args):

        columns = args.get("columns", None)

        data = pd.read_csv(
            filepath_or_buffer=args.get("file_path"),
            names=columns,
            encoding=args.get("encoding", "utf-8"),
        )

        if args.get("filter", None) is not None:
            data = data[args.get("filter")]

        return data

    def substitute(self, doc):
        doc = re.sub(r"<br />", " ", doc)
        doc = re.sub(r"\S+@\S+", " EMAIL ", doc)
        doc = re.sub(r"@\S+", " MENTION ", doc)
        doc = re.sub(r"https?:\S+|http?:\S+", " URL ", doc)
        doc = re.sub(r"(\d+\-\d+)|\d+", " NUMBER ", doc)
        doc = re.sub(r"[^A-Za-z']", " ", doc)

        return doc

    def expand(self, token):
        token_dict = {
            "ca": "can",
            "wo": "will",
            "sha": "shall",
            "'ve": "have",
            "'ll": "will",
            "'m": "am",
            "n't": "not",
            "'re": "are",
        }
        word_dict = {
            "cant": "can not",
            "couldnt": "could not",
            "wont": "will not",
            "pls": "please",
            "plz": "please",
            "youre": "you are",
            "theyre": "they are",
            "ive": "I have",
            "havent": "have not",
            "hasnt": "has not",
            "hadnt": "had not",
            "im": "I am",
            "didnt": "did not",
            "dont": "do not",
            "doesnt": "does not",
            "gotta": "got to",
            "wanna": "want to",
            "gonna": "going to",
            "wannabe": "want to be",
            "cannot": "can not",
        }
        if token in self.reserved_words:
            return token
        token = token.lower()
        if token.lower().strip() == "let's":
            return "let us"

        if token.lower().strip() == "'twas":
            return "it was"

        tokens = word_tokenize(token)
        if len(tokens) == 1:
            return word_dict.get(tokens[0], tokens[0])
        for i in range(len(tokens)):
            tokens[i] = token_dict.get(tokens[i], tokens[i])

        return " ".join(tokens)

    def preprocess(self, doc):
        doc = str(doc)
        doc = self.substitute(doc)

        tokens = doc.split()
        doc = " ".join([self.expand(w) for w in tokens])

        tokens = doc.split()
        tokens = [word for word in tokens if word.isalpha()]

        tokens = [
            word.lower() if word not in self.reserved_words else word for word in tokens
        ]

        return " ".join(tokens)

    def encode(self, data, drop=False):
        return pd.get_dummies(data, drop_first=drop)

    def split_data(self, X, Y, **args):
        return train_test_split(
            X, Y,
            train_size=args.get("train_size", self.train_size),
            stratify=Y,
            random_state=args.get("seed", self.seed)
        )

    def create_model(
            self,
            embedding_dim=100,
            lstm_units=128,
            optimizer='adam',
            init='glorot_uniform'
        ):
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(1,), dtype=tf.string))
        model.add(TextVectorization(
                        max_tokens=self.max_features,
                        output_mode='int',
                        output_sequence_length=self.max_len
                    )
                )
        model.add(Embedding(
                            self.max_features,
                            embedding_dim,
                            mask_zero=True
                    )
                )
        model.add(LSTM(lstm_units))
        model.add(Dense(2, activation='softmax', kernel_initializer=init))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        return model

    def train(self, **args):
        data = self.read_data(**args)

        if args.get("is_cleaned", False):
            data["text"].apply(self.preprocess)

        # import done, create model done
        # supply raw text, OH labels to fit
        self.model = KerasClassifier(
            build_fn=self.create_model,
            epochs=self.epoch,
            verbose=1
        )
        optimizers = ['rmsprop', 'adam']
        init = ['glorot_uniform', 'normal', 'uniform']
        batches = [32, 64]
        param_grid = dict(optimizer=optimizers, batch_size=batches, init=init)
        grid = GridSearchCV(estimator=self.model, param_grid=param_grid)
        x = tf.strings.as_string(data['text'].astype(str))
        grid_result = grid.fit(x, self.encode(data['target']))

    def predict(self, text, **config):
        
        result = self.model.predict(self.preprocess(text), verbose=1)[0] 
        if np.argmax(result) == 0:
            polarity =  'negative'
        else:
            polarity = 'positive'
        return {
            "result": result,
            "polarity": polarity
        }
        
    def load(self, model, tokenizer):
        self.model = load_model(model)
        self.tokenizer = pickle.load(open(tokenizer, 'rb'))

    def dump_tokenizer(self, path):
        pickle.dump(self.tokenizer, open(path, "wb"))