import cv2
from sklearn import cluster
from tqdm import trange
import numpy as np
import random
import os
import graph_based_img_seg as seg

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

img_path = 'data/data/imgs'
gt_path = 'data/data/gt'
images = os.listdir(img_path)
image_list = np.array([os.path.join(img_path, img) for img in images])
gt_imgs = os.listdir(gt_path)
gt_list = np.array([os.path.join(gt_path, img) for img in gt_imgs])
random_idx = random.sample(range(1000), 200)
random_idx = sorted(random_idx)

train_img_dirs = image_list[random_idx]
train_gt_dirs = gt_list[random_idx]

def calc_RGBHist(img, mask):
    m, n, _ = img.shape
    hist = np.zeros((8, 8, 8))
    for i in range(m):
        for j in range(n):
            if mask is None or mask[i, j] == 255:
                b = int(np.floor(img[i, j, 0] / 32))
                g = int(np.floor(img[i, j, 1] / 32))
                r = int(np.floor(img[i, j, 2] / 32))
                hist[b, g, r] += 1
    return hist

def gen_region_mask(img, djs, parent):
    m, n, _ = img.shape
    mask = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if djs.find_parent(i*n+j) == parent:
                mask[i, j] = 255
    return mask

def gen_region_features(img, djs):
    label = []
    processed = []
    region_features = []
    img_feature = calc_RGBHist(img, None)
    m, n, _ = img.shape
    for i in range(m):
        for j in range(n):
            parent = djs.find_parent(i*n+j)
            if parent not in processed:
                mask = gen_region_mask(img, djs, parent)
                # cv2.imshow('mask', mask)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                feature = calc_RGBHist(img, mask)
                region_features.append(np.hstack([feature, img_feature]))
                processed.append(parent)
                if djs.data[parent, 3] / djs.data[parent, 1] >= 0.5:
                    label.append(1)
                else:
                    label.append(0)
    return region_features, label

print("generating features: ")
features = []
gt_labels = []
for i in trange(train_img_dirs.size):
    img = cv2.imread(train_img_dirs[i])
    gt_img = cv2.imread(train_gt_dirs[i], cv2.IMREAD_GRAYSCALE)
    _, djs = seg.segmentation(img, gt_img, 80, 50)
    f, l = gen_region_features(img, djs)
    features.extend(f)
    gt_labels.extend(l)

print("conducting pca...")
pca = PCA(n_components=20)
pca_features = pca.fit_transform(features)

print("clustering word bags...")
kmeans = KMeans(n_clusters=50).fit(pca_features)
cluster_centers = kmeans.cluster_centers_
kmeans_features = np.matmul(pca_features, cluster_centers.T)

print("training SVM...")
features = np.hstack([pca_features, kmeans_features])
labels = gt_labels
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(features, labels)

print("testing...")

test_img_path = 'data/test/img'
test_gt_path = 'data/test/gt'
test_imgs = os.listdir(test_img_path)
test_img_list = np.array([os.path.join(test_img_path, img) for img in test_imgs])
test_gt_imgs = os.listdir(test_gt_path)
test_gt_list = np.array([os.path.join(test_gt_path, img) for img in test_gt_imgs])


t_features = []
t_gt_labels = []
for i in range(test_img_list.size):
    t_img = cv2.imread(test_img_list[i])
    t_gt_img = cv2.imread(test_gt_list[i], cv2.IMREAD_GRAYSCALE)
    _, djs = seg.segmentation(t_img, t_gt_img, 80, 50)
    f, l = gen_region_features(t_img, djs)
    t_features.extend(f)
    t_gt_labels.extend(l)

t_pca_features = pca.transform(t_features)
t_kmeans_features = np.matmul(t_pca_features, cluster_centers.T)
t_features = np.hstack([t_pca_features, t_kmeans_features])

true_predict = 0
predict_labels = clf.predict(t_features)
for i, label in enumerate(t_gt_labels):
    if predict_labels[i] == label:
        true_predict += 1

print("accuracy on test dataset: ", true_predict / len(t_gt_labels))
