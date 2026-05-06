"""
测试 recognition.py 中的图像预处理和人脸检测函数
"""
import pytest
import numpy as np
import cv2
import os
import sys

# 导入待测试函数
from recognition import (
    preprocess_face,
    white_balance,
    enhance_contour,
    detect_faces,
    skin_adaptive_preprocess,
    detect_skin_mask,
    assess_illumination,
)


class TestWhiteBalance:
    """测试白平衡函数"""

    def test_output_shape(self, test_image):
        """测试输出与输入形状一致"""
        result = white_balance(test_image)
        assert result.shape == test_image.shape

    def test_output_type(self, test_image):
        """测试输出类型为 uint8"""
        result = white_balance(test_image)
        assert result.dtype == np.uint8

    def test_no_crash_on_dark_image(self):
        """测试纯黑图片不崩溃"""
        dark_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = white_balance(dark_img)
        assert result is not None
        assert result.shape == dark_img.shape


class TestEnhanceContour:
    """测试轮廓增强函数"""

    def test_output_shape(self, test_image):
        """测试输出与灰度输入形状一致"""
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        result = enhance_contour(gray)
        assert result.shape == gray.shape

    def test_output_type(self):
        """测试输出为 uint8"""
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = enhance_contour(gray)
        assert result.dtype == np.uint8


class TestPreprocessFace:
    """测试人脸预处理管道"""

    def test_output_size(self, test_image):
        """测试输出尺寸为 100x100（LBPH 要求固定尺寸）"""
        # 裁剪人脸区域
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            pytest.skip("未检测到人脸")
        x, y, w, h = faces[0]
        face_bgr = test_image[y:y+h, x:x+w]

        result = preprocess_face(face_bgr)
        assert result.shape == (100, 100)

    def test_output_dtype(self, test_image):
        """测试输出数据类型为 uint8"""
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            pytest.skip("未检测到人脸")
        x, y, w, h = faces[0]
        face_bgr = test_image[y:y+h, x:x+w]

        result = preprocess_face(face_bgr)
        assert result.dtype == np.uint8

    def test_skin_adaptive_no_crash(self, test_image):
        """测试肤色自适应预处理不崩溃"""
        result = skin_adaptive_preprocess(test_image)
        assert result is not None
        assert len(result.shape) == 2  # 灰度图


class TestDetectFaces:
    """测试人脸检测函数"""

    def test_detect_on_test_image(self, test_image):
        """测试在测试图片上检测人脸"""
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, test_image)
        assert isinstance(faces, list) or isinstance(faces, tuple)
        # 如果检测到人脸，检查格式
        for face in faces:
            assert len(face) == 4

    def test_detect_on_blank_image(self):
        """测试空白图片不崩溃且检测结果为空"""
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, blank)
        assert len(faces) == 0


class TestSkinDetection:
    """测试肤色检测相关函数"""

    def test_skin_mask_output(self, test_image):
        """测试肤色掩码输出形状"""
        mask = detect_skin_mask(test_image)
        assert mask.shape == test_image.shape[:2]
        assert mask.dtype == np.uint8

    def test_assess_illumination(self, test_image):
        """测试光照评估函数"""
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        mask = detect_skin_mask(test_image)
        brightness, contrast = assess_illumination(gray, mask)
        # 返回值应在合理范围内
        assert 0.0 <= brightness <= 1.0
        assert 0.0 <= contrast <= 1.0