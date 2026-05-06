import sys
import os
import pytest
import numpy as np
import cv2

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 测试图片路径
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'test_face.jpg')

@pytest.fixture(scope="session")
def test_image():
    """加载测试用的真实人脸图片"""
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"测试图片不存在: {TEST_IMAGE_PATH}")
    img = cv2.imread(TEST_IMAGE_PATH)
    if img is None:
        pytest.skip("无法读取测试图片")
    return img

@pytest.fixture(scope="session")
def test_gray_face(test_image):
    """将测试图片转为灰度并裁剪人脸区域"""
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    # 使用 Haar 检测人脸
    cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        pytest.skip("测试图片中未检测到人脸")
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    # 统一缩放到 100x100（与 recognition.py 中一致）
    return cv2.resize(face_roi, (100, 100))