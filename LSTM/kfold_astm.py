from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import numpy as np
from pprint import pprint
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

max_features = 15000# num of unique words in 192 (pos and neg) emails
maxlen = 500  # cut texts after this number of words in each email document
batch_size = 32
nb_classes = 3

print('Loading data...')
index_label_tuples = pd.read_pickle('index_label_tuple_4530.pkl')
#with open ('index_label_tuples.json') as fh:
#    index_label_tuples = json.load (fh)'

train_size = int(1* len(index_label_tuples))
y = np.array([index_label_tuples[i][1] for i in range(len(index_label_tuples))])
x = np.array([index_label_tuples[i][0] for i in range(train_size)])
print('before padding', x.shape)
print('Pad sequences (samples x time)')
x = sequence.pad_sequences(x, maxlen=maxlen)
print('x shape:', x.shape)


print('Build model...')
#k-flod cross validation on model
kfold = StratifiedKFold(n_splits=2, shuffle=True)
kfold.get_n_splits(x,y)
cvscores= []
print(kfold)
D1 = pd.DataFrame([])
D2 = pd.DataFrame([])
for train, test in kfold.split(x,y):
    y_test = y[test];
    model = Sequential()
    model.add(Embedding(max_features, 128, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(3, activation='softmax'))

# try using different optimizers and different optimizer configs
    #D2 = D2.append(pd.DataFrame(y[test]))
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',#ROOTmns
              metrics=['accuracy'])

    print('Train...');ycat = to_categorical(y,3)
    model.fit(x[train], ycat[train],
              batch_size=batch_size,
              nb_epoch=1,
              validation_data=None)
    score, acc = model.evaluate(x[test], ycat[test],
                                batch_size=batch_size)
    print('Test score:', score);print('Test accuracy:', acc)
    cvscores.append(acc*100)
    
    print('Model prediction')
    y_pred = model.predict_classes(x[test], batch_size = batch_size)
    D1 = D1.append(pd.DataFrame(confusion_matrix(y_test, y_pred)))#, index = ['neutra', 'positive', 'negative'], columns = ['neutra', 'positive', 'negative']))
    print(D1)
    pr = np.array(precision_recall_fscore_support(y_test, y_pred))
    print(pr);pr = np.transpose(pr)
    D2= D2.append(pd.DataFrame(pr))#, index = ['neutra', 'positive', 'negative'], columns = ['precision', 'recall', 'fscore', 'avg']))
    print(D2)
print('Mean:', np.mean(cvscores), 'Std:', np.std(cvscores))
D1.to_csv('pr.csv')
D2.to_csv('ts.csv')

