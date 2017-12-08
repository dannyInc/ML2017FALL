import re
import sys
import csv
import math
import string
import numpy as np
import pandas as pd
import gensim, logging
from gensim.models import Word2Vec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, GRU, Bidirectional
from keras.layers import Flatten, Activation
from keras.layers.convolutional import Conv1D
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils
from keras.datasets import imdb

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('model_25_history_no@@.png')

def get_semi_data(self, name, label, threshold, loss_function):
    # if th==0.3, will pick label>0.7 and label<0.3
    label = np.squeeze(label)
    index = (label > 1 - threshold) + (label < threshold)
    semi_X = self.data[name][0]
    semi_Y = np.greater(label, 0.5).astype(np.int32)
    if loss_function == 'binary_crossentropy':
        return semi_X[index, :], semi_Y[index]
    elif loss_function == 'categorical_crossentropy':
        return semi_X[index, :], np_utils.to_categorical(semi_Y[index])
    else:
        raise Exception('Unknown loss function : %s' % loss_function)


train_label_path = sys.argv[1]
train_nolabel_path = sys.argv[2]

puns = set(string.punctuation)
version = 28
#train_label_path = 'training_label.txt'
#train_nolabel_path = 'training_nolabel.txt'
#test_path = 'testing_data.txt'
#result_path = "output_" + str(version) + "_w256_no@@.csv"

# read_file
with open(train_label_path, "r", encoding='UTF-8') as ins:
    x_train_label = []
    y_train_label = []
    for line in ins:
        line_2 = line.split(' +++$+++ ')
        line_2[1] = line_2[1][:-1]
        y_train_label.append(line_2.pop(0))
        line_2 = ''.join([c for c in line_2[0] if c not in puns])
        line_2 = line_2.split(' ')  # for word2vec
        x_train_label.append(line_2)
"""
with open(train_nolabel_path, "r", encoding='UTF-8') as ins_1:
    x_train_nolabel = []
    for line in ins_1:
        line = line.split(' ') # for word2vec
        x_train_nolabel.append(line)

with open(test_path, "r", encoding='UTF-8') as ins_2:
    x_test = []
    for line in ins_2:
        line_2 = line.split(',', 1)[1]
        line_2 = line_2[:-1]
        line_2 = ''.join([c for c in line_2 if c not in puns])
        line_2 = line_2.split(' ') # for word2vec
        x_test.append(line_2)
x_test.pop(0)
"""
# Tokenizer
"""
MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 25


tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=" ",
                      char_level=False)
tokenizer.fit_on_texts(x_train_label)
tokenizer.fit_on_texts(x_train_nolabel)
tokenizer.fit_on_texts(x_test)

sequences = tokenizer.texts_to_sequences(x_train_label)
sequences_nolabel = tokenizer.texts_to_sequences(x_train_nolabel)
sequences_test = tokenizer.texts_to_sequences(x_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
data_nolabel = pad_sequences(sequences_nolabel, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

labels = np_utils.to_categorical(np.asarray(y_train_label))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
"""
# Word2vec
#sentences = x_train_label + x_train_nolabel + x_test
fname_word2vec = 'w2v_Model_256'
#model_w2v = Word2Vec(sentences, size=256, window=5, min_count=10)
#model_w2v.save(fname_word2vec)
model_w2v = Word2Vec.load(fname_word2vec)

# embedding_matrix
EMBEDDING_DIM = 64

x_train_l_idx = []
x_train_nol_idx = []
x_test_idx = []

N = 25

for i,s in enumerate(x_train_label):
    x_train_label[i] = (s + N * ['.'])[:N]
"""
for i,s in enumerate(x_train_nolabel):
    x_train_nolabel[i] = (s + N * ['.'])[:N]

for i,s in enumerate(x_test):
    x_test[i] = (s + N * ['.'])[:N]
"""
for i,s in enumerate(x_train_label):
    tmp = []
    for j,w in enumerate(s):
        if w in model_w2v.wv:
            tmp.append(model_w2v.wv[w])
        else:
            tmp.append(model_w2v.wv['.'])
    tmp = np.array(tmp)
    x_train_l_idx.append(tmp)

"""
for i,s in enumerate(x_train_nolabel):
    tmp = []
    for j,w in enumerate(s):
        if w in model_w2v.wv:
            tmp.append(model_w2v.wv[w])
        else:
            tmp.append(model_w2v.wv['.'])
    x_train_nol_idx.append(np.array(tmp))

for i,s in enumerate(x_test):
    tmp = []
    for j,w in enumerate(s):
        if w in model_w2v.wv:
            tmp.append(model_w2v.wv[w])
        else:
            tmp.append(model_w2v.wv['.'])
    x_test_idx.append(np.array(tmp))
"""
x_train_l_idx = np.array(x_train_l_idx)
#x_train_nol_idx = np.array(x_train_nol_idx)
#x_test_idx = np.array(x_test_idx)
y_train_label = np.array(y_train_label)

for s in x_train_l_idx:
    s = s[0]
"""
for s in x_train_nol_idx:
    s = s[0]

for s in x_test_idx:
    s = s[0]
"""

#validation
VALIDATION_SPLIT = 0.1

indices = np.arange(len(x_train_l_idx))
np.random.shuffle(indices)
x_train_l_idx = x_train_l_idx[indices]
y_train_label = y_train_label[indices]
nb_validation_samples = int(VALIDATION_SPLIT * len(x_train_l_idx))

x_train = x_train_l_idx[:-nb_validation_samples]
y_train = y_train_label[:-nb_validation_samples]
x_val = x_train_l_idx[-nb_validation_samples:]
y_val = y_train_label[-nb_validation_samples:]

# embedding layer

# Build model
input_layer = Input(shape=(N, 256))
#embedded_sequences = embedding_layer(sequence_input)
# x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
x = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(input_layer)
x = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', dropout=0.5, recurrent_dropout=0.5)(x)

x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

#preds = Dense(2, activation='softmax')(x)
preds = Dense(1, activation='sigmoid')(x)

# optimizer
adam = Adam()
print ('compile model...')

# compile model
model = Model(input_layer, preds)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


#earlystopping
filepath = "weights_early_" + str(version) + ".hdf5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint2 = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
callbacks_list = [checkpoint1,checkpoint2]

#load model
#model = load_model('model_18_256.hdf5')

#print
model.summary()

BATCH_SIZE = 1024

#fit model for semi

train_history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                          epochs=25, batch_size=BATCH_SIZE)
#model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=BATCH_SIZE)
show_train_history(train_history, 'acc', 'val_acc')
"""
# semi
for i in range(6):
    # label the semi-data
    semi_pred = model.predict(x_train_nol_idx, batch_size=1024, verbose=True)
    semi_X, semi_Y = dm.get_semi_data('semi_data', semi_pred, 0.9, 'binary_crossentropy')
    semi_X = np.concatenate((semi_X, X))
    semi_Y = np.concatenate((semi_Y, Y))
    print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
    # train
    history = model.fit(semi_X, semi_Y,
                        validation_data=(x_val, y_val),
                        epochs=20,
                        batch_size=args.batch_size,
                        callbacks=[checkpoint, earlystopping] )

    if os.path.exists(save_path):
        print ('load model from %s' % save_path)
        model.load_weights(save_path)
    else:
        raise ValueError("Can't find the file %s" %path)

#predict for semi
y_nolabel = model.predict(data_nolabel, batch_size=BATCH_SIZE, verbose=1)

#semi-supervised
x_train = np.concatenate((x_train, data_nolabel), axis=0)
y_train = np.concatenate((y_train, y_nolabel), axis=0)

#fit model
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=BATCH_SIZE)
"""
#save model
model_name = "model_" + str(version) + "_w256.hdf5"
model.save(model_name)
"""
#predict
y_test = model.predict(x_test_idx, batch_size=BATCH_SIZE, verbose=1)
y_test = np.round(y_test)

#output
ans = []
for i in range(len(y_test)):
    ans.append([str(i)])
    ans[i].append(int(y_test[i]))

text = open(result_path, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()
"""