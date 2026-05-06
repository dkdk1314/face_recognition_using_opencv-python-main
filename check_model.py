import os
import pickle

os.chdir("/home/shumeipai/face_recognition_using_opencv-python-main")  # 改为你的实际路径

print(f"当前目录: {os.getcwd()}")
print(f"model 目录存在: {os.path.exists('model')}")
print(f"模型文件存在: {os.path.exists('model/weighted_lbph.pkl')}")

if os.path.exists("model/label_names.pkl"):
    with open("model/label_names.pkl", "rb") as f:
        label_names = pickle.load(f)
    print(f"模型中的家长: {label_names}")
else:
    print("label_names.pkl 不存在")