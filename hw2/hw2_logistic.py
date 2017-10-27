import numpy as np
import csv
import sys

x_train_path = sys.argv[3]
y_train_path = sys.argv[4]
x_test_path = sys.argv[5]
result_path = sys.argv[6]

def sigmoid(z):
    res = 1 / (1 + np.exp(-z))
    return np.clip(res, 1e-8, 1 - (1e-8))

x_train = np.loadtxt(x_train_path, delimiter=',', skiprows=1)
y_train = np.loadtxt(y_train_path, delimiter=',', skiprows=1)
x_test  = np.loadtxt(x_test_path, delimiter=',', skiprows=1)

repeat = 2000
lr_rate = 1.1
quad = [0, 4]
lamda = 100

#add quadratic term
for i in range(len(quad)):
    x_train = np.append(x_train, x_train[:,quad[i]:(quad[i]+1)]**2, axis=1)

for i in range(len(quad)):
    x_test = np.append(x_test, x_test[:,quad[i]:(quad[i]+1)]**2, axis=1)

#normalization

x_train_mean = np.array([np.mean(x_train, axis=0)])
x_train_std = np.array([np.std(x_train, axis=0)])

x_train = np.subtract(x_train, x_train_mean)
x_train = np.divide(x_train, x_train_std)

x_test = np.subtract(x_test, x_train_mean)
x_test = np.divide(x_test, x_train_std)

#regression

x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)
x_test = np.concatenate((np.ones((x_test.shape[0],1)), x_test), axis=1)

x_t = np.transpose(x_train)
w = np.zeros(len(x_train[0]))
s_gra = np.zeros(len(x_train[0]))

for i in range(repeat):
    hypo = np.dot(x_train,w)
    hypo = sigmoid(hypo)
    loss = hypo - y_train

    IDcount = 0
    hypo_round = np.round(hypo)
    for j in range(len(hypo)):
        if hypo_round[j] == y_train[j]:
            IDcount = IDcount+1
    accuracy = IDcount / len(y_train)

    gra = np.dot(x_t,loss)
    #gra += 2*lamda*w
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - lr_rate * gra/ada
    print ('iteration: %d | Accuracy: %f  ' % ( i,accuracy))

# save model
#np.save('model.npy',w)
# read model
#w = np.load('hw1_best_w.npy')


#output
ans = []
for i in range(len(x_test)):
    ans.append([str(i+1)])
    a = np.dot(w, x_test[i])
    a = round(sigmoid(a))
    ans[i].append(int(a))

text = open(result_path, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()
