import sys
import csv
import math
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.models import Model

test_path = sys.argv[1]
output_path = sys.argv[2]
movie_path = sys.argv[3]
user_path = sys.argv[4]

# parameters
k_factors = 10
BATCH_SIZE = 1024

# load data
with open(test_path, "r", encoding='UTF-8', errors='ignore') as ins:
    test_user = []
    test_movie = []
    next(ins)
    for line in ins:
        line_2 = line.split(',')
        line_2[2] = line_2[2][:-1]
        test_user.append(int(line_2[1]))
        test_movie.append(int(line_2[2]))

test_user = np.array(test_user)
test_movie = np.array(test_movie)

# normalization
rating_avg = 3.58171208604
rating_std = 1.11689766115

# load model
model_name1 = "model_5.hdf5"
model_name2 = "model_7.hdf5"
model_name3 = "model_8.hdf5"
model_name4 = "model_9.hdf5"
model1 = load_model(model_name1)
model2 = load_model(model_name2)
model3 = load_model(model_name3)
model4 = load_model(model_name4)

#predict
#y_test = model.predict([test_user, test_movie, test_movie_other], verbose=1)
y1 = model1.predict([test_user, test_movie], verbose=1)
y2 = model2.predict([test_user, test_movie], verbose=1)
y3 = model3.predict([test_user, test_movie], verbose=1)
y4 = model4.predict([test_user, test_movie], verbose=1)
y = (y1+y2+y3+y4)/4
y_test = y * rating_std + rating_avg
y_test = np.clip(y_test, 1, 5)

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