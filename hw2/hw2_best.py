from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv
import sys

csv_1 = sys.argv[1]
csv_2 = sys.argv[2]
x_train_path = sys.argv[3]
y_train_path = sys.argv[4]
x_test_path = sys.argv[5]
result_path = sys.argv[6]

x_train = np.loadtxt(x_train_path, delimiter=',', skiprows=1)
y_train = np.loadtxt(y_train_path, delimiter=',', skiprows=1)
x_test  = np.loadtxt(x_test_path, delimiter=',', skiprows=1)

clf = RandomForestClassifier(n_estimators=450, max_depth=13, n_jobs=-1)
clf.fit(x_train, y_train)
pdt = clf.predict(x_test)

ans = []
for i in range(len(x_test)):
    ans.append([str(i+1)])
    ans[i].append(int(pdt[i]))

text = open(result_path, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()
