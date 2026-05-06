"""
测试加权 LBPH 人脸识别算法的核心功能
基于 weighted_lbph.py 中的 WeightedLBPH 类
"""
import pytest
import numpy as np
import cv2
import os
import pickle
from weighted_lbph import WeightedLBPH, create_default_weights


class TestWeightedLBPHInit:
    """测试模型初始化"""

    def test_default_init(self):
        """测试默认参数初始化"""
        model = WeightedLBPH()
        assert model.grid == (8, 8)
        assert model.radius == 1
        assert model.n_points == 8
        assert model.method == 'uniform'
        # uniform 模式有 59 个 bin
        assert model.n_bins == 59

    def test_custom_grid_init(self):
        """测试自定义网格参数初始化"""
        model = WeightedLBPH(grid=(10, 10))
        assert model.grid == (10, 10)
        # 权重矩阵形状应与 grid 一致
        assert model.weights.shape == (10, 10)

    def test_none_weights_init(self):
        """测试不传 weights 时自动生成高斯权重"""
        model = WeightedLBPH(grid=(8, 8), weights=None)
        assert model.weights is not None
        assert model.weights.shape == (8, 8)
        # 中心区域权重应大于边缘
        center_val = model.weights[4, 4]
        edge_val = model.weights[0, 0]
        assert center_val > edge_val, "中心权重应大于边缘权重"


class TestCreateDefaultWeights:
    """测试权重创建函数"""

    def test_output_shape(self):
        """测试输出形状与网格大小一致"""
        weights = create_default_weights((6, 6))
        assert weights.shape == (6, 6)

    def test_center_weight_greater_than_edge(self):
        """测试中心区域权重大于边缘（高斯分布特性）"""
        weights = create_default_weights((8, 8))
        center_val = weights[4, 4]
        corner_val = weights[0, 0]
        assert center_val > corner_val

    def test_weights_positive(self):
        """测试所有权重为正值"""
        weights = create_default_weights((8, 8))
        assert np.all(weights > 0)


class TestWeightedLBPHTrainPredict:
    """测试训练和预测功能"""

    def test_train_basic(self):
        """测试基本训练：2个类别，每类3张模拟人脸"""
        model = WeightedLBPH(grid=(4, 4))
        # 创建模拟数据：2类，每类3张 100x100 的随机人脸
        images = []
        labels = []
        for label in range(2):
            for _ in range(3):
                # 生成不同分布的随机图像以模拟不同人
                img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
                # 给不同类别加入差异
                img = (img + label * 50) % 256
                images.append(img.astype(np.uint8))
                labels.append(label)

        label_names = ['Alice', 'Bob']
        model.train(images, labels, label_names)

        assert len(model.labels) == 6
        assert len(model.features) == 6
        assert model.label_names == label_names
        # 检查是否为每个标签设置了阈值
        assert 0 in model.label_thresholds
        assert 1 in model.label_thresholds

    def test_predict_same_person(self):
        """测试同一人图片识别结果正确"""
        model = WeightedLBPH(grid=(4, 4))

        # 用相似图像训练
        base_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        images = [base_img, np.clip(base_img + 5, 0, 255).astype(np.uint8)]
        labels = [0, 0]
        model.train(images, labels, ['PersonA'])

        # 用第一张图做预测
        pred_label, confidence = model.predict(base_img)
        assert pred_label == 0
        assert confidence >= 0  # 卡方距离非负

    def test_predict_different_person(self):
        """测试不同人图片识别为陌生人（-1）"""
        model = WeightedLBPH(grid=(4, 4))

        # 训练数据
        img_a = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        model.train([img_a], [0], ['PersonA'])

        # 用完全不同的图像预测
        img_b = np.ones((100, 100), dtype=np.uint8) * 200  # 全白图
        pred_label, confidence = model.predict(img_b)
        # 差别极大，应返回 -1（陌生人）
        assert pred_label == -1 or confidence > 10


class TestWeightedLBPHSaveLoad:
    """测试模型保存和加载"""

    def test_save_and_load(self, tmp_path):
        """测试保存后加载的模型能正常工作"""
        model = WeightedLBPH(grid=(4, 4))

        # 训练
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        model.train([img, img], [0, 0], ['Test'])

        # 保存
        save_path = os.path.join(tmp_path, "test_model.pkl")
        model.save(save_path)
        assert os.path.exists(save_path)

        # 加载
        loaded = WeightedLBPH()
        loaded.load(save_path)

        # 验证加载后的关键属性
        assert loaded.grid == model.grid
        assert loaded.label_names == model.label_names
        assert len(loaded.features) == len(model.features)

        # 加载后预测
        pred_label, _ = loaded.predict(img)
        assert pred_label == 0