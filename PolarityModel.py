import pandas as pd
import re
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM
from tensorflow.keras import regularizers
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pickle
import keras


class PolarityModel:

    reserved_words = ["EMAIL", "NUMBER", "MENTION", "URL"]
    tokenizer = None
    model = None

    def __init__(self, **args):
        self.max_features = args.get("max_features", 50000)
        self.max_len = args.get("max_len", 3200)
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

    def make_tokenizer(self, text, max_features):
        self.tokenizer = Tokenizer(num_words=max_features)
        self.tokenizer.fit_on_texts(list(text))

    def text_to_sequence(self, text):
        return self.tokenizer.texts_to_sequences(text)

    def pad(self,x, max_len):
        return sequence.pad_sequences(x, maxlen=max_len)

    def encode(self, data, drop=False):
        return pd.get_dummies(data, drop_first=drop)

    def split_data(self, X, Y, **args):
        return train_test_split(
            X, Y,
            train_size=args.get("train_size", self.train_size),
            stratify=Y,
            random_state=args.get("seed", self.seed)
        )

    def get_architecture(self, model, **config):
        model.add(
                    Embedding(
                        self.max_features,
                        config.get("emb_size", 200),
                        input_length=self.max_len
                    )
                )
        model.add(
                    Conv1D(
                        filters=config.get("filters", 32),
                        kernel_size=config.get("kernel", 3),
                        padding='same',
                        activation='relu',
                        kernel_regularizer=config.get("conv_reg", None),
                        bias_regularizer=config.get("conv_reg", None)
                    )
                )
        model.add(MaxPooling1D(pool_size=config.get("pool_size",2)))
        model.add(
                    Conv1D(
                        filters=config.get("filters", 32),
                        kernel_size=config.get("kernel", 3),
                        padding='same',
                        activation='relu',
                        kernel_regularizer=config.get("conv_reg", None),
                        bias_regularizer=config.get("conv_reg", None)
                    )
                )
        model.add(MaxPooling1D(pool_size=config.get("pool_size",2)))

        model.add(
                    Conv1D(
                        filters=config.get("filters", 32),
                        kernel_size=config.get("kernel", 3),
                        padding='same',
                        activation='relu',
                        kernel_regularizer=config.get("conv_reg", None),
                        bias_regularizer=config.get("conv_reg", None)
                    )
                )
        model.add(MaxPooling1D(pool_size=config.get("pool_size",2)))
        model.add(
                    Conv1D(
                        filters=config.get("filters", 32),
                        kernel_size=config.get("kernel", 3),
                        padding='same',
                        activation='relu',
                        kernel_regularizer=config.get("conv_reg", None),
                        bias_regularizer=config.get("conv_reg", None)
                    )
                )
        model.add(MaxPooling1D(pool_size=config.get("pool_size",2)))


        model.add(
                    Bidirectional(LSTM(
                        config.get("ls_units", 200),
                        dropout=0.2,
                        recurrent_dropout=0.2,
                        kernel_regularizer=config.get("ls_reg", None),
                        recurrent_regularizer=config.get("ls_reg", None),
                        bias_regularizer=config.get("ls_reg", None),
                        return_sequences=True
                    ))
                )
        model.add(
                    Bidirectional(LSTM(
                        config.get("ls_units", 100),
                        dropout=0.2,
                        recurrent_dropout=0.2,
                        kernel_regularizer=config.get("ls_reg", None),
                        recurrent_regularizer=config.get("ls_reg", None),
                        bias_regularizer=config.get("ls_reg", None),
                    ))
                )
        model.add(
                Dense(
                        config.get("out_units", 2),
                        activation=config.get("out_activation", "softmax")
                    )
                )
        model.compile(
                        loss=config.get("loss", "categorical_crossentropy"),
                        optimizer=config.get("optimizer",  "adam"),
                        metrics= [
                                  keras.metrics.TruePositives(name='tp'),
                                  keras.metrics.FalsePositives(name='fp'),
                                  keras.metrics.TrueNegatives(name='tn'),
                                  keras.metrics.FalseNegatives(name='fn'), 
                                  keras.metrics.CategoricalAccuracy(name='accuracy'),
                                  keras.metrics.Precision(name='precision'),
                                  keras.metrics.Recall(name='recall'),
                                  keras.metrics.AUC(name='auc')
                                ]
                    )
        model.summary()
        return model

    def get_model(self, **config):
        K.clear_session()
        if config.get("use_pre", False):
            for layer in self.model.layers:
                layer.trainable = False
                layer._name += str("_old")
                self.model.pop()
                self.model.pop()
                model = self.model

        else:
            model = Sequential()
        return self.get_architecture(model, **config)


    def train(self, **args):
        data = self.read_data(**args)

        if args.get("is_cleaned", False):
            data["text"].apply(self.preprocess)

        if args.get("use_pre", False) and args.get("tk_path", None) is not None and args.get("model_path", None) is not None:
          self.load(args.get("model_path"), args.get("tk_path"))
        else:
          self.make_tokenizer(data["text"], self.max_features)

        x = self.pad(self.text_to_sequence(data["text"]), self.max_len)
        y = self.encode(data["target"])

        x, x_val, y, y_val = self.split_data(x, y)
        self.model = self.get_model(**args)
        mc = ModelCheckpoint(
                        "model.h5", monitor='val_loss',
                        mode='min', verbose=1, save_best_only=True
                    )

        self.model.fit(
                    x,
                    y,
                    epochs=self.epoch,
                    validation_data=(x_val, y_val),
                    batch_size=self.batch_size,
                    verbose=1,
                    callbacks=[mc]
                )

    def predict(self, text,  **config):
        text = self.preprocess(text)
        x_test = self.pad(np.asarray(self.text_to_sequence([text])), config.get("max_len", self.max_len))

        result = self.model.predict(x_test, verbose=1)[0] 
        if np.argmax(result) == 0:
            polarity =  'negative'
        else:
            polarity = 'positive'
        return {
            "result": result,
            "polarity": polarity
        }


    def evaluate(self, **config):
        data = self.read_data(**config)
        data["text"] = data.text.apply(self.preprocess)
        x = self.pad(self.text_to_sequence(data["text"]), config.get("max_len", self.max_len))
        y = self.encode(data["target"])
        acc = self.model.evaluate(
                    x,
                    y
                )[1]
        return acc
        
    def load(self, model, tokenizer):
        self.model = load_model(model)
        self.tokenizer = pickle.load(open(tokenizer, 'rb'))

    def dump_tokenizer(self, path):
        pickle.dump(self.tokenizer, open(path, "wb"))