import numpy as np
import cv2
import pickle
import os
from skimage.feature import local_binary_pattern

class WeightedLBPH:
    def __init__(self, grid=(8, 8), weights=None, radius=1, n_points=8, method='uniform'):
        self.grid = grid
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.n_bins = 59 if method == 'uniform' else 2 ** n_points
        self.weights = weights if weights is not None else self._create_gaussian_weights(grid)
        self.labels = []          # 存储标签（整数）
        self.features = []        # 存储每个样本的特征向量
        self.label_names = []     # 标签对应的姓名
        self.label_thresholds = {} # 每个标签的自适应阈值
        self.global_threshold = 50.0

    def _create_gaussian_weights(self, grid):
        rows, cols = grid
        weights = np.zeros((rows, cols))
        center_r, center_c = rows // 2, cols // 2
        sigma = max(rows, cols) / 4
        for i in range(rows):
            for j in range(cols):
                dist2 = (i - center_r) ** 2 + (j - center_c) ** 2
                weights[i, j] = np.exp(-dist2 / (2 * sigma * sigma))
        weights = weights / np.mean(weights)
        return weights

    def _extract_features(self, face_gray):
        h, w = face_gray.shape
        cell_h = h // self.grid[0]
        cell_w = w // self.grid[1]
        hist = []
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < self.grid[0] - 1 else h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < self.grid[1] - 1 else w
                cell = face_gray[y_start:y_end, x_start:x_end]
                lbp = local_binary_pattern(cell, self.n_points, self.radius, method=self.method)
                hist_cell, _ = np.histogram(lbp.ravel(), bins=self.n_bins, range=(0, self.n_bins))
                hist_cell = hist_cell / (np.linalg.norm(hist_cell) + 1e-7)  # L2 归一化
                hist_cell = hist_cell * self.weights[i, j]
                hist.extend(hist_cell)
        return np.array(hist, dtype=np.float32)

    def train(self, images, labels, label_names=None):
        self.labels = []
        self.features = []
        for img, label in zip(images, labels):
            feat = self._extract_features(img)
            self.features.append(feat)
            self.labels.append(label)
        if label_names is not None:
            self.label_names = label_names

        # 计算每个标签的自适应阈值
        unique_labels = set(self.labels)
        all_thresholds = []
        for lbl in unique_labels:
            lbl_indices = [i for i, l in enumerate(self.labels) if l == lbl]
            lbl_features = [self.features[i] for i in lbl_indices]
            # 计算类内距离（所有样本两两之间的卡方距离）
            intra_distances = []
            for i in range(len(lbl_features)):
                for j in range(i+1, len(lbl_features)):
                    d = np.sum((lbl_features[i] - lbl_features[j])**2 / (lbl_features[i] + lbl_features[j] + 1e-7))
                    intra_distances.append(d)
            if len(intra_distances) == 0:
                # 只有一个样本，暂设为0，后面用全局阈值填充
                self.label_thresholds[lbl] = 0
            else:
                mean_dist = np.mean(intra_distances)
                std_dist = np.std(intra_distances)
                thr = mean_dist + 2.0 * std_dist   # 系数1.5可调
                self.label_thresholds[lbl] = thr
                all_thresholds.append(thr)
        # 计算全局阈值（用于单样本情况）
        if all_thresholds:
            self.global_threshold = np.mean(all_thresholds)
        else:
            self.global_threshold = 50.0
        # 将单样本的阈值设为全局阈值
        for lbl in self.label_thresholds:
            if self.label_thresholds[lbl] == 0:
                self.label_thresholds[lbl] = self.global_threshold
        return self

    def predict(self, face_gray):
        feat = self._extract_features(face_gray)
        distances = []
        for train_feat in self.features:
            d = np.sum((feat - train_feat)**2 / (feat + train_feat + 1e-7))
            distances.append(d)
        best_idx = np.argmin(distances)
        best_label = self.labels[best_idx]
        confidence = distances[best_idx]
        # 使用该标签的专属阈值
        threshold = self.label_thresholds.get(best_label, self.global_threshold)
        if confidence < threshold:
            return best_label, confidence
        else:
            return -1, confidence   # -1 表示陌生人

    def save(self, filename):
        data = {
            'grid': self.grid,
            'weights': self.weights,
            'radius': self.radius,
            'n_points': self.n_points,
            'method': self.method,
            'n_bins': self.n_bins,
            'labels': self.labels,
            'features': self.features,
            'label_names': self.label_names,
            'label_thresholds': self.label_thresholds,
            'global_threshold': self.global_threshold
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.grid = data['grid']
        self.weights = data['weights']
        self.radius = data['radius']
        self.n_points = data['n_points']
        self.method = data['method']
        self.n_bins = data['n_bins']
        self.labels = data['labels']
        self.features = data['features']
        self.label_names = data['label_names']
        self.label_thresholds = data.get('label_thresholds', {})
        self.global_threshold = data.get('global_threshold', 50.0)

def create_default_weights(grid_size=(8, 8)):
    rows, cols = grid_size
    weights = np.zeros((rows, cols))
    center_r, center_c = rows // 2, cols // 2
    sigma = max(rows, cols) / 4
    for i in range(rows):
        for j in range(cols):
            dist2 = (i - center_r) ** 2 + (j - center_c) ** 2
            weights[i, j] = np.exp(-dist2 / (2 * sigma * sigma))
    weights = weights / np.mean(weights)
    return weights