#!/usr/bin/env python3
import os
import urllib.request
from pathlib import Path

def download_file(url, destination):
    """从指定 URL 下载文件到目标路径"""
    print(f"正在下载 {os.path.basename(destination)}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✓ 成功下载 {os.path.basename(destination)}")
        return True
    except Exception as e:
        print(f"✗ 下载 {os.path.basename(destination)} 时出错: {e}")
        return False

def main():
    # 创建 models 目录
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # 模型文件 URL（FP16 版本，体积更小）
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel"

    # 目标路径（使用 _fp16 文件名）
    prototxt_path = models_dir / "deploy.prototxt"
    model_path = models_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

    print("=" * 60)
    print("OpenCV DNN 人脸检测模型下载器 (FP16)")
    print("=" * 60)

    if prototxt_path.exists():
        print(f"⚠ {prototxt_path.name} 已存在，跳过下载。")
    else:
        download_file(prototxt_url, prototxt_path)

    if model_path.exists():
        print(f"⚠ {model_path.name} 已存在，跳过下载。")
    else:
        download_file(model_url, model_path)

    if prototxt_path.exists() and model_path.exists():
        print("\n✓ 所有模型文件都已准备就绪！")
    else:
        print("\n⚠ 部分文件可能缺失，请检查上面的信息。")

if __name__ == "__main__":
    main()