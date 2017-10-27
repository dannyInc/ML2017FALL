import numpy as np
from numpy.linalg import inv
import csv
import sys

x_train_path = sys.argv[3]
y_train_path = sys.argv[4]
x_test_path = sys.argv[5]
result_path = sys.argv[6]

def sigmoid(z):
    res = 1/(1 + np.exp(-z))
    return np.clip(res, 1e-8, 1 - (1e-8))

x_train = np.loadtxt(x_train_path, delimiter=',', skiprows=1)
y_train = np.loadtxt(y_train_path, delimiter=',', skiprows=1)
x_test  = np.loadtxt(x_test_path, delimiter=',', skiprows=1)

# add quadratic term
quad = [0,4]

for i in range(len(quad)):
    x_train = np.append(x_train, x_train[:,quad[i]:(quad[i]+1)]**2, axis=1)

for i in range(len(quad)):
    x_test = np.append(x_test, x_test[:,quad[i]:(quad[i]+1)]**2, axis=1)

# normalization

x_train_mean = np.array([np.mean(x_train, axis=0)])
x_train_std = np.array([np.std(x_train, axis=0)])

x_train = np.subtract(x_train, x_train_mean)
x_train = np.divide(x_train, x_train_std)

x_test = np.subtract(x_test, x_train_mean)
x_test = np.divide(x_test, x_train_std)

# mean_0 mean_1
dim = len(x_train[0])
mean_0 = np.zeros(dim)
mean_1 = np.zeros(dim)
cnt_0 = 0
cnt_1 = 0

for i in range(len(x_train)):
    if y_train[i] == 0:
        mean_0 += x_train[i]
        cnt_0 += 1
    else:
        mean_1 += x_train[i]
        cnt_1 += 1

mean_0 /= cnt_0
mean_1 /= cnt_1

# sigma_0 sigma_1 -> shared_sigma
sigma_0 = np.zeros((dim, dim))
sigma_1 = np.zeros((dim, dim))

for i in range(len(x_train)):
    if y_train[i] == 0:
        X = [x_train[i] - mean_0]
        sigma_0 += np.dot(np.transpose(X), X)
    else:
        X = [x_train[i] - mean_1]
        sigma_1 += np.dot(np.transpose(X), X)

sigma_0 /= cnt_0
sigma_1 /= cnt_1

shared_sigma = (cnt_0/len(x_train)) * sigma_0 + (cnt_1/len(x_train)) * sigma_1

# w,b
w = np.dot([mean_0 - mean_1], inv(shared_sigma))
w = np.transpose(w)

b  = np.dot(np.dot([mean_1], inv(shared_sigma)), np.transpose([mean_1]))
b -= np.dot(np.dot([mean_0], inv(shared_sigma)), np.transpose([mean_0]))
b /= 2
b += np.log(float(cnt_0)/cnt_1)

w = np.concatenate((b, w), axis=0)
w = np.transpose(w)[0]
x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)

# save model
#np.save('model_generative.npy',w)
# read model
#w = np.load('hw1_best_w.npy')


# accuracy
hypo = np.dot(x_train, w)
hypo = sigmoid(-hypo)
IDcount = 0
hypo_round = np.round(hypo)
for j in range(len(hypo)):
    if hypo_round[j] == y_train[j]:
            IDcount = IDcount+1
accuracy = IDcount / len(y_train)
print('Accuracy: %f ' % (accuracy))

#output
x_test = np.concatenate((np.ones((x_test.shape[0],1)), x_test), axis=1)

ans = []
for i in range(len(x_test)):
    ans.append([str(i+1)])
    a = np.dot(w, x_test[i])
    a = round(sigmoid(-a))
    ans[i].append(int(a))

text = open(result_path, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()