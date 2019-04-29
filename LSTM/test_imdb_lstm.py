'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import json
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import numpy as np
from pprint import pprint
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

#max_features = 1378 #227 # num of unique words in 192 (pos and neg) emails
max_features = 1293 # num of unique words in 192 (pos and neg) emails
#maxlen = 2777 #700  # cut texts after this number of words in each email document
maxlen = 500  # cut texts after this number of words in each email document
batch_size = 16
nb_classes = 3

print('Loading data...')

index_label_tuples = pd.read_pickle('index_label_tuple.pkl')
#with open ('index_label_tuples.json') as fh:
#    index_label_tuples = json.load (fh)'

train_size = int(0.9 * len(index_label_tuples))
y = [index_label_tuples[i][1] for i in range(len(index_label_tuples))]

x_train = np.array([index_label_tuples[i][0] for i in range(train_size)])
y_train = np.array(y[:train_size])

print(y_train)
x_test = np.array([index_label_tuples[i][0] for i in range(train_size,len(index_label_tuples))])
y_test = np.array(y[train_size:])

# y_train = np_utils.to_categorical(y_train, nb_classes)
# y_test = np_utils.to_categorical(y_test, nb_classes)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(y_train.shape)
print(y_test.shape)
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(3, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
y_train_cat = to_categorical(y_train, 3)
print(y_train_cat[0:3])
y_test_cat = to_categorical(y_test, 3)
model.fit(x_train, y_train_cat,
          batch_size=batch_size,
          nb_epoch=10,
          validation_split = 0.2)
score, acc = model.evaluate(x_test, y_test_cat,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

print('Model prediction')
y_pred = model.predict_classes(x_test, batch_size = batch_size)
D1 = pd.DataFrame(confusion_matrix(y_test, y_pred), index = ['neutra', 'positive', 'negative'], columns = ['neutra', 'positive', 'negative'])
D2= pd.DataFrame(np.array(precision_recall_fscore_support(y_test, y_pred)), columns = ['neutra', 'positive', 'negative'], index= ['precision', 'recall', 'fscore', 'avg'])
print(D1);print(D2);
