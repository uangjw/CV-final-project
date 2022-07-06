import cv2
import numpy as np
from tqdm import trange

img = cv2.imread('data/data/imgs/970.png')
original = img.copy()
#img = cv2.imread('boat.jpg', cv2.IMREAD_GRAYSCALE)
gt_img = cv2.imread('data/data/gt/970.png', cv2.IMREAD_GRAYSCALE)

def gt_penalty(i, j, gt_img):
    if gt_img[i, j] == 0:
        return 0
    else:
        return 1000

def min_seam(img, gt_img):
    m, n = img.shape

    M = np.zeros((m, n), dtype=int)
    backtrack = np.zeros((m, n), dtype=int)

    for i in range(1, m):
        for j in range(0, n):
            if j == 0:
                del1 = abs(int(img[i, j+2]) - int(img[i, j]))
                del2 = abs(int(img[i-1, j+1]) - int(img[i, j]))
                del3 = abs(int(img[i-1, j+1]) - int(img[i, j+2]))
                v1 = M[i-1, j] + del1 + del2 + gt_penalty(i-1, j, gt_img)
                v2 = M[i-1, j+1] + del1 + gt_penalty(i-1, j+1, gt_img)
                v3 = M[i-1, j+2] + del1 + del3 + gt_penalty(i-1, j+2, gt_img)
                M[i, j] = min(v1, v2, v3)
                if v1 == min(v1, v2, v3):
                    backtrack[i, j] = j
                elif v2 == min(v1, v2, v3):
                    backtrack[i, j] = j+1
                else:
                    backtrack[i, j] = j+2
            elif j == n-1:
                del1 = abs(int(img[i, j]) - int(img[i, j-2]))
                del2 = abs(int(img[i-1, j-1]) - int(img[i, j-2]))
                del3 = abs(int(img[i-1, j-1]) - int(img[i, j]))
                v1 = M[i-1, j-2] + del1 + del2 + gt_penalty(i-1, j-2, gt_img)
                v2 = M[i-1, j-1] + del1 + gt_penalty(i-1, j-1, gt_img)
                v3 = M[i-1, j] + del1 + del3 + gt_penalty(i-1, j, gt_img)
                M[i, j] = min(v1, v2, v3)
                if v1 == min(v1, v2, v3):
                    backtrack[i, j] = j-2
                elif v2 == min(v1, v2, v3):
                    backtrack[i, j] = j-1
                else:
                    backtrack[i, j] = j
            else:
                del1 = abs(int(img[i, j+1]) - int(img[i, j-1]))
                del2 = abs(int(img[i-1, j]) - int(img[i, j-1]))
                del3 = abs(int(img[i-1, j]) - int(img[i, j+1]))
                v1 = M[i-1, j-1] + del1 + del2 + gt_penalty(i-1, j-1, gt_img)
                v2 = M[i-1, j] + del1 + gt_penalty(i-1, j, gt_img)
                v3 = M[i-1, j+1] + del1 + del3 + gt_penalty(i-1, j+1, gt_img)
                M[i, j] = min(v1, v2, v3)
                if v1 == min(v1, v2, v3):
                    backtrack[i, j] = j-1
                elif v2 == min(v1, v2, v3):
                    backtrack[i, j] = j
                else:
                    backtrack[i, j] = j+1
    return M, backtrack

def seam_carving(img, gt_img):
    m, n, _ = img.shape
    M, backtrack = min_seam(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), gt_img)
    mask = np.ones((m, n), dtype=bool)
    j = np.argmin(M[-1])
    for i in reversed(range(m)):
        mask[i, j] = False
        j = backtrack[i, j]
    gt_img = gt_img[mask].reshape((m, n-1))
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((m, n-1, 3))
    return img, gt_img

scale = 0.8

m, n, _ = img.shape

# cnt = 0

for i in trange(n - int(scale * n)):
    img, gt_img = seam_carving(img, gt_img)
    # file_name = 'SeamCarving/gif_frames/' + str(cnt) + '.jpg'
    # cur_m, cur_n, _ = img.shape
    # img0 = np.pad(img[:,:,0], (n-cur_m, n-cur_n), 'constant', constant_values=(0, 0))
    # img1 = np.pad(img[:,:,1], (n-cur_m, n-cur_n), 'constant', constant_values=(0, 0))
    # img2 = np.pad(img[:,:,2], (n-cur_m, n-cur_n), 'constant', constant_values=(0, 0))
    # cv2.imwrite(file_name, cv2.merge([img0, img1, img2]))
    # cnt += 1

img = cv2.transpose(img)
gt_img = cv2.transpose(gt_img)

for j in trange(m - int(scale * m)):
    img, gt_img = seam_carving(img, gt_img)
    # file_name = 'SeamCarving/gif_frames/' + str(cnt) + '.jpg'
    # img0 = np.pad(img[:,:,0], (n-cur_m, n-cur_n), 'constant', constant_values=(0, 0))
    # img1 = np.pad(img[:,:,1], (n-cur_m, n-cur_n), 'constant', constant_values=(0, 0))
    # img2 = np.pad(img[:,:,2], (n-cur_m, n-cur_n), 'constant', constant_values=(0, 0))
    # cv2.imwrite(file_name, cv2.transpose(cv2.merge([img0, img1, img2])))
    # cnt += 1

img = cv2.transpose(img)
gt_img = cv2.transpose(gt_img)

cv2.imwrite('SeamCarving/result/970-result.jpg', img)
cv2.imshow('image', original)
# cv2.imshow('gt mask', gt_img)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()