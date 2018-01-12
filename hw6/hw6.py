import sys
import numpy as np
import pandas as pd

image_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]

clust = np.load('clust.npy')
X = np.load(image_path)
X = np.reshape(X, (-1, 28, 28))

numclust = {0, 5, 7, 11, 12, 15, 16, 17}
'''
for i, img in enumerate(X):
    if clust[i] == 19:
        plt.imshow(img)
        plt.show()
'''
f = pd.read_csv(test_path)
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])
o = open(output_path, 'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = clust[i1]
    p2 = clust[i2]
    if p1 in numclust and p2 in numclust:
        pred = 1  # two images in same cluster
    elif p1 not in numclust and p2 not in numclust:
        pred = 1  # two images in same cluster
    else:
        pred = 0  # two images not in same cluster
    o.write("{},{}\n".format(idx, pred))
o.close()

