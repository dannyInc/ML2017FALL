import pandas as pd
import numpy as np
from numpy.linalg import inv
import math
import csv
import sys

test_path = sys.argv[1]
result_path = sys.argv[2]

data = pd.read_csv("./train.csv", encoding='big5')

#parameters
#item  = [9]
item  = [7,8,9,16]
quad = [8,9]

n_hours = 8
repeat = 10000
lr_rate = 10
lamda = 1

#preprocessing
cols = [0,1,2]
data = data.drop(data.columns[cols],axis=1)
data = data.as_matrix()
data[data == 'NR'] = 0
data = np.asfarray(data,float)

flat_data = data[0:18,:]

for i in range(1,240):
    flat_data = np.append(flat_data, data[18*i:18*(i+1),:], axis=1)


for i in range(len(quad)):
    flat_data = np.append(flat_data, [flat_data[quad[i]]**2], axis=0)
    item += [18+i]


x = []
y = []
for i in range(12):
    # 一個月取連續n_hours小時的data可以有480-n_hours筆
    for j in range(480-n_hours):
        x.append([])
        # specified items
        for t in item:
            # 連續n_hours小時
            for s in range(n_hours):
                x[(480-n_hours)*i+j].append(flat_data[t][480*i+j+s] )
        y.append(flat_data[9][480*i+j+n_hours])


x = np.array(x)
y = np.array(y)

#normalization
x_mean = np.array([np.mean(x, axis=0)])
x_std = np.array([np.std(x, axis=0)])
x = np.subtract(x, x_mean)
x = np.divide(x, x_std)

x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
"""
#regression

x_t = np.transpose(x)
w = np.zeros(len(x[0]))
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    gra += 2*lamda*w
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - lr_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))


# close form
#w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)+lamda*np.identity(len(x[0]))),x.transpose()),y)
"""
# save model
#np.save('hw1_best_model.npy',w)
# read model
w = np.load('hw1_best_w.npy')



#read test.csv
test_data = pd.read_csv(test_path, header=None, encoding = 'big5')
test_data = test_data.drop(test_data.columns[[0,1]], axis=1)
test_data = test_data.as_matrix()
test_data[test_data == 'NR'] = 0
test_data = np.asfarray(test_data, float)

t_x = []
for i in range(240):
    t_x.append([])
    # specified items
    for t in item:
        if t<18:
            for s in range(9-n_hours, 9):
                t_x[i].append(test_data[18*i+t][s])
        else:
            quad_array = test_data[18*i+quad[t-18]]**2
            for s in range(9-n_hours, 9):
                t_x[i].append(quad_array[s])


t_x = np.array(t_x)
t_x = np.subtract(t_x, x_mean)
t_x = np.divide(t_x, x_std)
t_x = np.concatenate((np.ones((t_x.shape[0],1)),t_x), axis=1)

#output
ans = []
for i in range(len(t_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w, t_x[i])
    ans[i].append(a)

text = open(result_path, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()
