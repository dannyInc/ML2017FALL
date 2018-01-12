from skimage import io
from skimage import transform
import sys
import numpy as np

imgs_path = sys.argv[1]
recon_path = sys.argv[2]

# imgs_path = './Aberdeen'
# recon_path = '216.jpg'
recon_num = int(recon_path.split('.')[0])
LOAD_IMG = 0
NUM_EIGEN = 4
# RECON = [5, 64, 112, 216]
RECON = [recon_num]
old_shape = (600, 600)
new_shape = (400, 400)

if(LOAD_IMG == 0):
    X = []
    for i in range(415):
        filename = imgs_path+'/'+str(i)+'.jpg'
        img = np.array(io.imread(filename))
        new_img = transform.resize(img, new_shape)
        new_img = new_img.flatten()
        X.append(new_img)
    X = np.array(X)
    # np.save('X.npy', X)
else:
    X = np.load('X.npy')

X_mean = np.average(X, axis=0)
U, s, V = np.linalg.svd(np.transpose(X - X_mean), full_matrices=False)
# S = np.diag(s)
# print(U.shape)

# mean face
# X_mean_img = np.reshape(X_mean, (400, 400, 3))
# X_mean_img = transform.resize(X_mean_img, old_shape)
# plt.imshow(X_mean_img) #Needs to be in row,col order
# print(X_mean_img)
# plt.savefig('X_mean.jpg')

def ModImg(M):
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M
# top 4 eigenfaces
"""
for i in range(NUM_EIGEN):
    M = ModImg(U[:, i])
    img = np.reshape(M, (400, 400, 3))
    img = transform.resize(img, old_shape)
    plt.imshow(img)
    plt.savefig('eigFace_'+str(i+1)+'.jpg')
"""
eigFace = np.transpose(U[:, :NUM_EIGEN])

# reconstruction
reconFace = []
for i in RECON:
    y = np.array([X[i]-X_mean])
    p = np.dot(eigFace, np.transpose(y))
    reImg = np.dot(np.transpose(p), eigFace)
    reImg += X_mean
    reImg = ModImg(reImg)
    reImg = np.reshape(reImg, (400, 400, 3))
    reImg = transform.resize(reImg, old_shape)
    io.imsave('reconstruction.jpg', reImg)
    #io.imsave('reconFace_'+str(NUM_EIGEN)+'_'+str(i)+'.jpg', reImg)
    #plt.savefig('reconFace_'+str(NUM_EIGEN)+'_'+str(i)+'.jpg')

