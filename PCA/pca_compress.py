import cv2
import numpy as np
import matplotlib.pyplot as plt

def rescale_img(img):
    m, n = img.shape
    max = img.max()
    min = img.min()
    if min < 0:
        for i in range(m):
            for j in range(n):
                img[i, j] += -min
        max = img.max()
        min = 0
    scale = 255 / (max - min)
    for i in range(m):
        for j in range(n):
            img[i, j] = int(np.floor(img[i, j] * scale))
    return img

def pca(X, k):
    m, n = X.shape
    mean = []
    for i in range(m):
        mean.append(np.mean(X[i]))
        for j in range(n):
            X[i, j] -= mean[i]
    
    C = 1 / n * np.matmul(X, X.T)
    eigen_val, eigen_vec = np.linalg.eig(C)
    idx = np.argsort(eigen_val)
    k_vec_idx = idx[:-(k+1):-1]
    P = eigen_vec[:, k_vec_idx]
    P = np.transpose(P)
    Y = np.matmul(P, X)
    Y = np.matmul(P.T, Y)
    for i in range(m):
        for j in range(n):
            Y[i, j] += mean[i]

    P = rescale_img(P)
    Y = rescale_img(Y)
    
    return P, Y.T

original = cv2.imread('data/data/imgs/270.png', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('data/data/imgs/270.png', cv2.IMREAD_GRAYSCALE)

m0, n0 = img.shape

k = 6
d = 12

original = original[0 : (m0 - m0 % d), 0 : ( n0 - n0 % d)]
img = img[0 : (m0 - m0 % d), 0 : ( n0 - n0 % d)]
m, n = img.shape
pm = int(m / d)
pn = int(n / d)

X = np.zeros((pm * pn, d * d))
idx = 0
for i in range(pm):
    for j in range(pn):
        vec = np.array(img[i*d : (i+1) * d, j*d : (j+1) * d])
        #cv2.imshow('part', cv2.resize(vec, (d*10, d*10)))
        X[idx] = vec.reshape(d * d)
        idx += 1

P, Y = pca(X.T, k)

for i in range(k):
    vec = P[i]
    vec = vec.reshape((d, d))
    vec = np.array(vec, dtype='uint8')
    #cv2.imshow('vec'+str(i), cv2.resize(vec, (d*10, d*10)))
    

result = np.zeros((m, n))
idx = 0
for i in range(pm):
    for j in range(pn):
        result[i*d : (i+1) * d, j*d : (j+1) * d] = Y[idx].reshape((d, d))
        idx += 1

cv2.imshow('original', original)
result = np.array(result, dtype='uint8')
cv2.imwrite('PCA/result/6D-result.jpg', result)
cv2.imshow('compressed', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
