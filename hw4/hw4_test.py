import sys
import csv
import math
import numpy as np
from keras.models import load_model
from gensim.models import Word2Vec

# path
test_path = sys.argv[1]
result_path = sys.argv[2]

batch_sz = 1024
N = 25

# Word2vec
fname_word2vec = 'w2v_Model_256'
model_w2v = Word2Vec.load(fname_word2vec)

# load test
with open(test_path, "r", encoding='UTF-8') as ins_2:
    x_test = []
    for line in ins_2:
        line_2 = line.split(',', 1)[1]
        line_2 = line_2[:-1]
        line_2 = line_2.split(' ')
        x_test.append(line_2)
x_test.pop(0)

x_test_idx = []

for i,s in enumerate(x_test):
    x_test[i] = (s + N * ['.'])[:N]

for i,s in enumerate(x_test):
    tmp = []
    for j,w in enumerate(s):
        if w in model_w2v.wv:
            tmp.append(model_w2v.wv[w])
        else:
            tmp.append(model_w2v.wv['.'])
    x_test_idx.append(np.array(tmp))

x_test_idx = np.array(x_test_idx)

for s in x_test_idx:
    s = s[0]

#load model
model = load_model('model_18_w256.hdf5')

#predict
y_test = model.predict(x_test_idx, batch_size=batch_sz, verbose=1)
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