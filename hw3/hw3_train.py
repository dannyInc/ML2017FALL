import sys
import csv
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Activation, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils

#GPU usage
"""
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
sess = tf.Session(config = config)
"""

#path
#x_train_path = sys.argv[1]
#x_test_path = sys.argv[2]
#result_path = sys.argv[1]
train_data = sys.argv[1]

#plot
"""
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('cnn_history_best.png')
"""

#load data
"""
x_train = np.load('train_x.npy')
y_train = np.load('train_y.npy')
x_valid = np.load('val_x.npy')
y_valid = np.load('val_y.npy')
x_test = np.load('test_x.npy')
"""

#load data from train.csv
mytrain = pd.read_csv(train_data)

train_x = []
val_x = []
train_y = []
val_y = []

for i in range(len(mytrain)):
    if(i%20 == 7):
        val_x.append(np.array(mytrain.iloc[i,1].split()).reshape(48,48,1).astype(float)/256)
    else:
        temp = np.array(mytrain.iloc[i,1].split()).reshape(48,48,1).astype(float)
        temp /= 256
        train_x.append(temp)
        train_x.append(np.flip(temp,axis=1))
train_x = np.array(train_x)
val_x = np.array(val_x)

for i in range(len(mytrain)):
    if(i%20 == 7):
        val_y.append(int(mytrain.iloc[i,0]))
    else:
        train_y.append(int(mytrain.iloc[i,0]))
        train_y.append(int(mytrain.iloc[i,0]))
train_y = np.array(train_y)
train_y = np_utils.to_categorical(train_y, 7)
val_y = np.array(val_y)
val_y = np_utils.to_categorical(val_y, 7)

#parameters
epochs = 90

#ImageGenerator
datagen_train = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.5,
	rotation_range=13,
    horizontal_flip=False,
    fill_mode='nearest')

datagen_train.fit(train_x)

#CNN
model = Sequential()
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

for i in range(2):
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

model.add(Flatten())

#DNN:
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(units=7, activation='softmax'))

#optimizers
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#compile
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#earlystopping
filepath="weights_early_1.hdf5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint2 = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
callbacks_list = [checkpoint1,checkpoint2]

#print
model.summary()

#fit model
batch_sz = 200
train_history = model.fit_generator(datagen_train.flow(train_x,train_y, batch_size=batch_sz,shuffle=True),
                    steps_per_epoch=3*(math.floor(len(train_x)/batch_sz)), epochs=epochs,
                    validation_data=(val_x, val_y),
                    validation_steps=len(val_x)/batch_sz,
                    callbacks=callbacks_list, verbose=1)

#show_train_history(train_history, 'acc', 'val_acc')

#save model
model.save('model_5.hdf5')