import pandas as pd
import numpy as np
from keras.preprocessing.text import  Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.metrics import Precision, Recall, FalseNegatives, \
FalsePositives, TrueNegatives, TruePositives
from sklearn.preprocessing import OneHotEncoder
from kerastuner.tuners import Hyperband
from kerastuner import HyperParameters
from keras.optimizers import Adam, Adadelta, Adagrad, RMSprop
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report, \
confusion_matrix
import matplotlib.pyplot as plt

seed = 123456
OH_encoder = None
metrics = [
            'accuracy',
            Precision(), 
            Recall(),
            TruePositives(), 
            TrueNegatives(), 
            FalsePositives(),
            FalseNegatives()
        ]

def read_data(data_path, cols):
    if cols:
        return pd.read_csv(data_path)[cols]

    return pd.read_csv(data_path)

def split_data(data, test_size=0.15, stratify='target'):
    return train_test_split(data, test_size=0.15, stratify=data[stratify], random_state=seed)

def OH_fit_transform(col):
    OH_fit(col)
    return OH_transform(col)

def OH_fit(col):
    global OH_encoder
    OH_encoder = OneHotEncoder(sparse=False)
    OH_encoder.fit(col.values.reshape(-1, 1))

def OH_transform(col):
    return OH_encoder.transform(col.values.reshape(-1, 1))
     
def get_optimizer(key):
    map = {
      'Adam': Adam(),
      'Adadelta': Adadelta(),
      'Adagrad': Adagrad(),
      'RMSprop': RMSprop()
    }
    return map.get(key)

def get_tuner(build_func,directory, project, hp, max_epochs=4,executions_per_trial=2):
    return Hyperband(
        build_func,
        max_epochs=max_epochs,
        objective='val_loss',
        executions_per_trial=executions_per_trial,
        directory=directory,
        project_name=project,
        hyperparameters=hp,
    )

def dump(obj, path):
    return pickle.dump(obj, open(path, 'wb'))

def load(path):
    return pickle.load(open(path, 'rb'))

def get_callbacks(file_path):
    return [
          EarlyStopping(monitor='val_accuracy', patience=5, mode='max', \
                        restore_best_weights=True),
          ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, \
                          save_best_only=True,  mode='max')
    ]
def get_sequences(tokenizer, max_len, texts):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len)

def get_model_from_config(base_line, optimizer, mtr=metrics):
    model = Sequential().from_config(base_line.get_config())
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics= mtr
    )
    return model


def model_evaluate(model, X_test, y_test, batch_size=32):
    # predict class with test set
    y_pred_test =  model.predict_classes(X_test, batch_size=batch_size, verbose=1)
    print('Accuracy:\t{:0.1f}%'.format(accuracy_score(np.argmax(y_test,axis=1),y_pred_test)*100))
    
    #classification report
    print('\n')
    print(classification_report(np.argmax(y_test,axis=1), y_pred_test))

    #confusion matrix
    confmat = confusion_matrix(np.argmax(y_test,axis=1), y_pred_test)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(confmat, cmap=plt.cm.Purples, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()