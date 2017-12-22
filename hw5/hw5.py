import sys
import csv
import math
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras import regularizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, Dot
from keras.layers import Flatten, Activation, Merge, Add, Concatenate
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils

test_path = sys.argv[1]
output_path = sys.argv[2]
movie_path = sys.argv[3]
user_path = sys.argv[4]
#train_path = 'train.csv'

# parameters
k_factors = 10
VALIDATION_SPLIT = 0.05
TRAIN_OTHER_FEATURE = 0
BATCH_SIZE = 1024
EPOCHS = 40
version = 23

all_genre = ['Action','Adventure','Sci-Fi','Western','Documentary','Drama'
            ,'Animation','Children\'s','Comedy','Musical','Romance','Fantasy'
            ,'Crime','Film-Noir','Horror','Mystery','Thriller','War']

def DotModel(n_users, n_items, k_factors):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    #genres_input = Input(shape=[1])
    user_v = Embedding(n_users, k_factors, embeddings_initializer='random_normal')(user_input)
    user_v = Flatten()(user_v)
    item_v = Embedding(n_items, k_factors, embeddings_initializer='random_normal')(item_input)
    item_v = Flatten()(item_v)
    #genres_v = Embedding(n_items, k_factors, embeddings_initializer='random_normal')(genres_input)
    #genres_v = Flatten()(genres_v)
    user_b = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_b = Flatten()(user_b)
    item_b = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
    item_b = Flatten()(item_b)

    r_hat = Dot(axes=1)([user_v, item_v])
    #r_hat_2 = Dot(axes=1)([user_v, genres_v])
    r_hat = Add()([r_hat, user_b, item_b])
    model = Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer='adamax')
    model.summary()
    return model

def nnModel(n_users, n_items, k_factors):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_v = Embedding(n_users, k_factors, embeddings_initializer='random_normal')(user_input)
    user_v = Flatten()(user_v)
    item_v = Embedding(n_items, k_factors, embeddings_initializer='random_normal')(item_input)
    item_v = Flatten()(item_v)
    merge_v = Concatenate()([user_v, item_v])
    hidden = Dense(150, activation='relu')(merge_v)
    hidden = Dropout(0.3)(hidden)
    hidden = Dense(100, activation='relu')(hidden)
    hidden = Dropout(0.3)(hidden)
    output = Dense(1)(hidden)
    model = Model([user_input, item_input], output)
    model.compile(loss='mse', optimizer='adamax')
    model.summary()
    return model

def draw(x, y):
    y = np.array(y)
    x = np.array(x, dtype=np.float64)
    vis_data = TSNE(n_components=2).fit_transform(x)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(vis_x, vis_y, c=y, cmap=cm)
    plt.colorbar(sc)
    plt.savefig('tsne.png')

# load data
with open(user_path, "r", encoding='UTF-8', errors='ignore') as ins:
    users = []
    next(ins)
    for line in ins:
        line_2 = line.split('::')
        users.append(int(line_2[0]))

with open(movie_path, "r", encoding='UTF-8', errors='ignore') as ins:
    movies = []
    if (TRAIN_OTHER_FEATURE == 1):
        movie_genres = np.zeros([3962, 1])
    else:
        movie_genres = np.zeros([3962, 1])
    next(ins)
    for i,line in enumerate(ins):
        line = line.split('::')
        movies.append(int(line[0]))
        gen = line[2].split('|')
        gen[-1] = gen[-1][:-1]
        if(TRAIN_OTHER_FEATURE == 1):
            movie_genres[int(line[0])] = all_genre.index(gen[0])
            """
            for s in gen:
                movie_genres[int(line[0])][all_genre.index(s)] = 1
            """
        else:
            movie_genres[int(line[0])] = math.floor(all_genre.index(gen[0])/6)
"""
with open(train_path, "r", encoding='UTF-8', errors='ignore') as ins:
    train_user = []
    train_movie = []
    ratings = []
    next(ins)
    for line in ins:
        line_2 = line.split(',')
        line_2.pop(0)
        line_2[2] = line_2[2][:-1]
        train_user.append(int(line_2[0]))
        train_movie.append(int(line_2[1]))
        ratings.append(int(line_2[2]))
"""
with open(test_path, "r", encoding='UTF-8', errors='ignore') as ins:
    test_user = []
    test_movie = []
    next(ins)
    for line in ins:
        line_2 = line.split(',')
        line_2[2] = line_2[2][:-1]
        test_user.append(int(line_2[1]))
        test_movie.append(int(line_2[2]))

# other features
"""
if (TRAIN_OTHER_FEATURE == 1):
    train_movie_other = []
    test_movie_other = []
    for i in range(len(train_movie)):
        train_movie_other.append(movie_genres[train_movie[i]])
    train_movie_other = np.array(train_movie_other)
    for i in range(len(test_movie)):
        test_movie_other.append(movie_genres[test_movie[i]])
    test_movie_other = np.array(test_movie_other)

train_user = np.array(train_user)
train_movie = np.array(train_movie)
ratings = np.array(ratings)
"""
test_user = np.array(test_user)
test_movie = np.array(test_movie)
users = np.array(users)
movies = np.array(movies)
"""
indices = np.arange(len(train_user))
np.random.shuffle(indices)
train_user = train_user[indices]
train_movie = train_movie[indices]
ratings = ratings[indices]
"""
# normalization
"""
rating_avg = np.average(ratings)
rating_std = np.std(ratings)
ratings = (ratings-rating_avg)/rating_std
"""
rating_avg = 3.58171208604
rating_std = 1.11689766115

# build model
#model = DotModel(np.amax(users)+10, np.amax(movies)+10, k_factors)

# load model
model_name = "model_9.hdf5"
model = load_model(model_name)

# get embedding
user_emb = np.array(model.layers[2].get_weights()).squeeze()
print('user_emb shape:', user_emb.shape)
movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print('movie_emb shape:', movie_emb.shape)
#np.save('user_emb.npy', user_emb)
#np.save('movie_emb.npy', movie_emb)

#earlystopping
filepath = "w_early_" + str(version) + ".hdf5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint2 = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
callbacks_list = [checkpoint1, checkpoint2]

#print
model.summary()

#fit
"""
if(TRAIN_OTHER_FEATURE == 1):
    train_history = model.fit([train_user, train_movie, train_movie_other], ratings, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE)
else:
    train_history = model.fit([train_user, train_movie], ratings, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE)
"""
#save model
#model_name = "model_" + str(version) + ".hdf5"

#predict
#y_test = model.predict([test_user, test_movie, test_movie_other], verbose=1)
y_test = model.predict([test_user, test_movie], verbose=1)
y_test = y_test * rating_std + rating_avg

# tsne
#draw(movie_emb, movie_genres)

#output
ans = []
for i in range(len(y_test)):
    ans.append([str(i+1)])
    ans[i].append(y_test[i][0])

text = open(output_path, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["TestDataID", "Rating"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()