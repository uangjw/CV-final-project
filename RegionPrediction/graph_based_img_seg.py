import cv2
import numpy as np
import os


class DisjointSet:
    def __init__(self, num_vertices):
        self.num_sets = num_vertices
        self.num_vertices = num_vertices
        self.data = np.zeros((num_vertices, 4), dtype=int)

        for i in range(num_vertices):
            self.data[i, 0] = 0 # rank，即以节点i为根的集合所表示的树的层数
            self.data[i, 1] = 1 # size，即以节点i为根的集合的节点数
            self.data[i, 2] = i # parent，初始化时每个parent都指向自己
            self.data[i, 3] = 0 # gt label，即以节点i为根的集合是否属于前景
    
    # 为节点idx找到父节点，同时压缩路径
    def find_parent(self, idx):
        parent = idx
        while parent != self.data[parent, 2]:
            parent = self.data[parent, 2]
        self.data[idx, 2] = parent
        return parent

    # 基于rank将以idx1与idx2为根的两集合合并
    def join(self, idx1, idx2):
        if self.data[idx1, 0] > self.data[idx2, 0]:
            # idx1作为新集合的根
            self.data[idx2, 2] = idx1
            self.data[idx1, 1] += self.data[idx2, 1]
        else:
            # idx2作为新集合的根
            self.data[idx1, 2] = idx2
            self.data[idx2, 1] += self.data[idx1, 1]
            if self.data[idx1, 0] == self.data[idx2, 0]:
                # 若idx1的集合与idx2的集合rank相等，则新集合rank需加一
                self.data[idx2, 0] += 1
        self.num_sets -= 1

# 计算两像素的不相似度
def calc_dist(p1, p2):
    diff = (p1 - p2) ** 2
    return np.sqrt(np.sum(diff))

def build_graph(img):
    m, n, _ = img.shape
    edges_v = []    # 边的顶点矩阵
    edges_dist = [] # 边的权值矩阵（不相似度）
    for i in range(m):
        for j in range(n):
            # 生成四连通的边
            if j < n - 1:
                edges_v.append(np.array([i*n+j, i*n+(j+1)]))
                edges_dist.append(calc_dist(img[i, j], img[i, j+1]))
            if i < m - 1:
                edges_v.append(np.array([i*n+j, (i+1)*n+j]))
                edges_dist.append(calc_dist(img[i, j], img[i+1, j]))
    # 调整edges_v形状，第一维为边数，第二维为两顶点
    edges_v = np.vstack(edges_v).astype(int)
    edges_dist = np.array(edges_dist).astype(float)
    idx = np.argsort(edges_dist)    # 将边按照权值排序
    return edges_v[idx], edges_dist[idx]

def segmentation(img, gt_img, k, min_pixels):
    m, n, _ = img.shape
    edges_v, edges_dist = build_graph(img)

    djs = DisjointSet(m * n)
    # 设置阈值
    threshold = np.zeros(m * n, dtype=float)
    for i in range(m * n):
        threshold[i] = k

    # 遍历每一条边进行集合合并
    for i in range(len(edges_v)):
        v1_parent = djs.find_parent(edges_v[i, 0])
        v2_parent = djs.find_parent(edges_v[i, 1])
        # 对两个不相交的集合，查看是否可以合并
        if (v1_parent != v2_parent):
            # 查看两集合是否足够相似
            if (edges_dist[i] <= threshold[v1_parent]) and (edges_dist[i] <= threshold[v2_parent]):
                djs.join(v1_parent, v2_parent)
                v1_parent = djs.find_parent(v1_parent)
                # 更新不相似度阈值
                threshold[v1_parent] = edges_dist[i] + k / djs.data[v1_parent, 1]
    while(True):
        flag = False
        # 遍历所有边，消除过小分割区域
        for i in range(len(edges_v)):
            v1_parent = djs.find_parent(edges_v[i, 0])
            v2_parent = djs.find_parent(edges_v[i, 1])
            # 如果这一边上两集合中有一个集合的像素过少，合并
            if (v1_parent != v2_parent) and ((djs.data[v1_parent, 1] < min_pixels) or (djs.data[v2_parent, 1] < min_pixels)):
                djs.join(v1_parent, v2_parent)
                flag = True
        if not flag:
            break

    # 对照前景蒙版gt_img标记出前景区域
    for i in range(m):
        for j in range(n):
            if gt_img[i, j] > 127:
                djs.data[djs.find_parent(i*n+j), 3] += 1
    
    predict = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            parent = djs.find_parent(i*n+j)
            # 如果有过半数像素都在前景中，认为这一区域属于前景区域
            if djs.data[parent, 3] / djs.data[parent, 1] >= 0.5:
                predict[i, j] = 255
    return predict, djs

img = cv2.imread('data/data/imgs/570.png')
gt_img = cv2.imread('data/data/gt/570.png', cv2.IMREAD_GRAYSCALE)

predict, djs = segmentation(img, gt_img, 80, 50)

m, n = gt_img.shape

intersect = 0
union = 0

for i in range(m):
    for j in range(n):
        if predict[i, j] > 127:
            union += 1
            if gt_img[i, j] >127:
                intersect += 1
        else:
            if gt_img[i, j] > 127:
                union += 1

print("IOU = ", float(intersect) / union)
# print("num of regions = ", num_sets)
cv2.imshow('predict', predict)
cv2.imshow('image', img)
cv2.imshow('gt', gt_img)
cv2.waitKey(0)
cv2.destroyAllWindows()