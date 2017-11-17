import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import csv
import numpy as np
import pandas as pd
from keras.models import load_model

test_data = sys.argv[1]
result_path = sys.argv[2]
batch_sz = 200

#load data
mytest = pd.read_csv(test_data)

test_x = []
for i in mytest.iloc[:,1]:
    temp = np.array(i.split()).reshape(48,48,1).astype(float)
    temp /= 256
    test_x.append(temp)
test_x = np.array(test_x)

#load model
model = load_model('model_5.hdf5')
y_test = model.predict(test_x, batch_size=batch_sz, verbose=1)

#argmax
y_test = np.argmax(y_test, axis=1)

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