import cv2
import os
import datetime
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from db_manager import DatabaseManager
from utils import play_alert_sound
from weighted_lbph import WeightedLBPH, create_default_weights
from picamera2 import Picamera2

# ========== 配置 ==========
database_folder = "database1"
csv_file = "store1.csv"
unknown_folder = "unknown_faces"
os.makedirs(unknown_folder, exist_ok=True)
os.makedirs("records", exist_ok=True)
os.makedirs("model", exist_ok=True)

# 肤色自适应预处理开关（True=启用高级预处理，False=使用原有简单预处理）
USE_SKIN_ADAPTIVE = True

# ========== 全局模型缓存 ==========
_CACHED_RECOGNIZER = None
_CACHED_NAME_TO_ID = None
_CACHED_LABEL_NAMES = None

def get_cached_model(db=None):
    """
    获取缓存的模型（避免重复加载）
    第一次调用时加载模型，之后直接返回缓存。
    """
    global _CACHED_RECOGNIZER, _CACHED_NAME_TO_ID, _CACHED_LABEL_NAMES
    if _CACHED_RECOGNIZER is None:
        _CACHED_RECOGNIZER, _CACHED_NAME_TO_ID, _CACHED_LABEL_NAMES = load_lbph_model(db)
    return _CACHED_RECOGNIZER, _CACHED_NAME_TO_ID, _CACHED_LABEL_NAMES

# ========== 全局变量 ==========
_font_cache = None
PEAK_MODE = False          # 高峰模式标志（可由自动切换修改）
auto_switch = True         # 是否启用自动切换（默认开启）
THRESHOLD_NORMAL = 80
THRESHOLD_PEAK = 100

# ========== 中文字体加载 ==========
def get_chinese_font(font_size=24):
    global _font_cache
    if _font_cache:
        return _font_cache
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "simhei.ttf",
        "msyh.ttc"
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                _font_cache = ImageFont.truetype(path, font_size)
                print(f"使用字体: {path}")
                return _font_cache
            except:
                continue
    print("警告: 未找到中文字体，中文将显示为方框")
    _font_cache = ImageFont.load_default()
    return _font_cache

def cv2_put_chinese_text(img, text, position, font_size=24, color=(0,255,0)):
    if not text:
        return img
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_chinese_font(font_size)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ========== 图像预处理增强 ==========
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:,:,1])
    avg_b = np.average(result[:,:,2])
    result[:,:,1] = result[:,:,1] - ((avg_a - 128) * (result[:,:,1] / 255.0))
    result[:,:,2] = result[:,:,2] - ((avg_b - 128) * (result[:,:,2] / 255.0))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def enhance_contour(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sobelx, sobely)
    mag = np.uint8(np.clip(mag, 0, 255))
    return cv2.addWeighted(gray, 0.7, mag, 0.3, 0)

# ---------- 肤色自适应预处理（选项B） ----------
def detect_skin_mask(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 180, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    return mask

def assess_illumination(gray, mask):
    if mask is None:
        mask = np.ones_like(gray, dtype=np.uint8) * 255
    skin_pixels = gray[mask > 0]
    if len(skin_pixels) == 0:
        return 0.5, 0.5
    mean_bright = np.mean(skin_pixels) / 255.0
    contrast = np.std(skin_pixels) / 255.0
    return mean_bright, min(1.0, contrast * 3)

def adaptive_white_balance(bgr, mask=None):
    if mask is None:
        mask = np.ones(bgr.shape[:2], dtype=np.uint8) * 255
    skin_bgr = bgr[mask > 0].reshape(-1, 3)
    if len(skin_bgr) == 0:
        return bgr
    avg_b, avg_g, avg_r = np.mean(skin_bgr, axis=0)
    gray_avg = (avg_b + avg_g + avg_r) / 3.0
    if gray_avg < 1:
        return bgr
    scale_b, scale_g, scale_r = gray_avg/avg_b, gray_avg/avg_g, gray_avg/avg_r
    result = bgr.astype(np.float32)
    result[:,:,0] = np.clip(result[:,:,0] * scale_b, 0, 255)
    result[:,:,1] = np.clip(result[:,:,1] * scale_g, 0, 255)
    result[:,:,2] = np.clip(result[:,:,2] * scale_r, 0, 255)
    return result.astype(np.uint8)

def adaptive_gamma_correction(gray, brightness_score):
    if brightness_score < 0.3:
        gamma = 0.5
    elif brightness_score < 0.5:
        gamma = 0.7
    elif brightness_score > 0.7:
        gamma = 1.5
    elif brightness_score > 0.85:
        gamma = 2.0
    else:
        gamma = 1.0
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(gray, table)

def adaptive_contrast_stretch(gray, mask, clip_limit=2.0):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    if mask is not None:
        result = np.where(mask > 0, enhanced, gray)
    else:
        result = enhanced
    return result

def skin_adaptive_preprocess(face_bgr):
    mask = detect_skin_mask(face_bgr)
    balanced = adaptive_white_balance(face_bgr, mask)
    gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)
    brightness, contrast = assess_illumination(gray, mask)
    gamma_corrected = adaptive_gamma_correction(gray, brightness)
    if contrast < 0.2:
        clip = 3.0
    elif contrast < 0.35:
        clip = 2.0
    else:
        clip = 1.5
    enhanced = adaptive_contrast_stretch(gamma_corrected, mask, clip)
    final = cv2.GaussianBlur(enhanced, (3,3), 0)
    return final

# ---------- 统一预处理入口（含开关和缩放到100x100） ----------
def preprocess_face(face_bgr):
    if USE_SKIN_ADAPTIVE:
        gray = skin_adaptive_preprocess(face_bgr)
    else:
        face_bgr = white_balance(face_bgr)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        gray = cv2.equalizeHist(gray)
    # 统一缩放到 100x100（加权 LBPH 要求固定尺寸）
    gray = cv2.resize(gray, (100, 100))
    return gray

# ========== DNN 人脸检测器初始化 ==========
def init_dnn_detector():
    model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config_path = "models/deploy.prototxt"
    if os.path.exists(model_path) and os.path.exists(config_path):
        net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        print("DNN 人脸检测器加载成功 (FP16)")
        return net
    else:
        print("警告: DNN 模型文件未找到，将仅使用 Haar 检测")
        return None

dnn_net = init_dnn_detector()
haar_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

def detect_faces(gray, bgr):
    gray_enhanced = enhance_contour(gray)
    faces_haar = haar_cascade.detectMultiScale(gray_enhanced, scaleFactor=1.1, minNeighbors=3, minSize=(40,40))
    valid_faces = []
    if dnn_net is not None:
        for (x,y,w,h) in faces_haar:
            face_roi = bgr[y:y+h, x:x+w]
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (300,300), [104,117,123], False, False)
            dnn_net.setInput(blob)
            detections = dnn_net.forward()
            if detections[0,0,0,2] > 0.5:
                valid_faces.append((x,y,w,h))
        if len(valid_faces) == 0:
            h,w = gray.shape
            blob = cv2.dnn.blobFromImage(bgr, 1.0, (300,300), [104,117,123], False, False)
            dnn_net.setInput(blob)
            detections = dnn_net.forward()
            for i in range(detections.shape[2]):
                conf = detections[0,0,i,2]
                if conf > 0.5:
                    x = int(detections[0,0,i,3] * w)
                    y = int(detections[0,0,i,4] * h)
                    right = int(detections[0,0,i,5] * w)
                    bottom = int(detections[0,0,i,6] * h)
                    valid_faces.append((x, y, right-x, bottom-y))
    else:
        valid_faces = faces_haar
    return valid_faces

# ========== 模型加载 ==========
def load_lbph_model(db=None):
    """加载加权 LBPH 模型，db 参数可选（用于数据库一致性检查）"""
    model_file = "model/weighted_lbph.pkl"
    if not os.path.exists(model_file):
        print("模型文件不存在，请先运行信息管理模块添加家长并训练模型")
        return None, None, None

    recognizer = WeightedLBPH()
    recognizer.load(model_file)

    # 加载标签映射
    with open("model/label_names.pkl", "rb") as f:
        label_names = pickle.load(f)
    with open("model/name_to_id.pkl", "rb") as f:
        name_to_id = pickle.load(f)

    # 如果提供了数据库连接，可以检查模型与数据库是否一致（可选）
    if db is not None:
        # 这里可以添加一致性检查，但不是必须的
        pass

    return recognizer, name_to_id, label_names

def load_parent_child_mapping(db):
    rows = db.fetch_all("SELECT p.name, p.id, c.id, c.name FROM parents p JOIN children c ON p.child_id = c.id")
    mapping = {}
    for pname, pid, cid, cname in rows:
        mapping[pname] = (pid, cid, cname)
    return mapping

# ========== 主识别函数 ==========
def run_recognition():
    """命令行版本的人脸识别（与 GUI 使用相同的核心逻辑）"""
    db = DatabaseManager(password='20040619')

    # 预先加载模型
    recognizer, name_to_id, label_names = get_cached_model(db)
    if recognizer is None:
        print("模型加载失败")
        db.close()
        return

    # 加载家长-幼儿映射
    parent_child_map = {}
    if db is not None:
        parent_child_map = load_parent_child_mapping(db)

    # 初始化摄像头
    from picamera2 import Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    print("摄像头已启动，按 'q' 退出")

    recorded_parents = set()
    stranger_saved = False

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 调用与 GUI 相同的单帧识别函数
            parent_name, child_name, confidence, processed_frame = run_recognition_once(frame, db)

            # 显示结果
            if parent_name == "陌生人":
                print(f"[安全预警] 陌生人 (置信度: {confidence:.2f})")
            elif parent_name == "无人脸":
                pass  # 不打印
            else:
                print(f"识别到: {parent_name} -> {child_name}")
                # 记录考勤（去重）
                if parent_name not in recorded_parents:
                    recorded_parents.add(parent_name)
                    # 可选：保存抓拍照片
                    # photo_path = f"records/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    # cv2.imwrite(photo_path, frame)
                    print(f"已记录: {parent_name} 接 {child_name}")

            cv2.imshow("Child Pickup System", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        db.close()
        print("识别已结束")

if __name__ == "__main__":
    run_recognition()


def run_recognition_once(frame, db=None):
    """
    单帧人脸识别函数，供 GUI 调用
    参数:
        frame: OpenCV 图像 (BGR)
        db: 数据库连接对象（可选，如果为 None 则跳过数据库操作）
    返回:
        (parent_name, child_name, confidence, processed_frame)
    """
    # 加载模型（这里每次调用都加载，实际可全局加载一次以提升性能）
    recognizer, name_to_id, label_names = get_cached_model(db)
    if recognizer is None:
        return "系统错误", "", 0.0, frame

    # 加载家长-幼儿映射（如果 db 不为 None）
    parent_child_map = {}
    if db is not None:
        parent_child_map = load_parent_child_mapping(db)

    # 人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, frame)

    if len(faces) == 0:
        return "无人脸", "", 0.0, frame

    # 只处理第一个检测到的人脸
    x, y, w, h = faces[0]
    face_bgr = frame[y:y+h, x:x+w]
    face_gray = preprocess_face(face_bgr)
    label, confidence = recognizer.predict(face_gray)

    # 绘制矩形和文字（直接在原帧上绘制，方便 GUI 显示）
    if label != -1:
        parent_name = label_names[label]
        if parent_name in parent_child_map:
            parent_id, child_id, child_name = parent_child_map[parent_name]
            color = (0, 255, 0)
        else:
            child_name = "未绑定幼儿"
            color = (0, 255, 255)  # 黄色警告
    else:
        parent_name = "陌生人"
        child_name = ""
        color = (0, 0, 255)
        # ========== 添加陌生人预警（蜂鸣器） ==========
        from utils import play_alert_sound
        play_alert_sound()
        # ===========================================

    # 绘制矩形和文字
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    text = f"{parent_name}" if parent_name != "陌生人" else "陌生人"
    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 返回结果
    child_name = child_name if 'child_name' in locals() else ""
    return parent_name, child_name, confidence, frame