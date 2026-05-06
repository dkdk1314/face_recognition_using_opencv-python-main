import sys
import cv2
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QTableWidget,
                             QTableWidgetItem, QGroupBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from db_manager import DatabaseManager
from recognition import run_recognition_once, get_cached_model
from picamera2 import Picamera2

class FaceRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("幼儿智能接送人脸核验系统")
        self.setGeometry(0, 0, 480, 320)

        self.init_ui()

        # 居中显示
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # 后台初始化数据库
        self.db = None
        self.db_ready = False
        self.init_db_thread = threading.Thread(target=self._init_db)
        self.init_db_thread.daemon = True
        self.init_db_thread.start()

        # 后台加载模型
        self.model_ready = False
        self.init_model_thread = threading.Thread(target=self._init_model)
        self.init_model_thread.daemon = True
        self.init_model_thread.start()

        # 摄像头相关（picamera2）
        self.picam2 = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.running = False

        # 记录已记录的家长
        self.recorded_parents = set()

    def _init_db(self):
        try:
            self.db = DatabaseManager(password='20040619', timeout=3)
            import time
            time.sleep(0.5)
            self.db_ready = self.db.is_ready()
            print(f"数据库连接结果: {self.db_ready}")
        except Exception as e:
            print(f"数据库初始化异常: {e}")
            self.db_ready = False

    def _init_model(self):
        try:
            get_cached_model(self.db if self.db_ready else None)
            self.model_ready = True
            print("人脸识别模型已加载")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model_ready = False

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧：摄像头显示
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.video_label = QLabel()
        self.video_label.setFixedSize(320, 240)
        self.video_label.setStyleSheet("border: 2px solid black; background-color: #333;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)
        left_layout.addWidget(self.video_label)

        self.result_label = QLabel("等待识别...")
        self.result_label.setFont(QFont("微软雅黑", 10))
        self.result_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.result_label)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("启动摄像头")
        self.stop_btn = QPushButton("停止摄像头")
        self.quit_btn = QPushButton("退出系统")
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.quit_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.quit_btn)
        left_layout.addLayout(btn_layout)

        main_layout.addWidget(left_widget)

        # 右侧：考勤记录表
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        info_group = QGroupBox("今日接送记录")
        info_layout = QVBoxLayout(info_group)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["家长姓名", "幼儿姓名", "核验时间"])
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        info_layout.addWidget(self.table)
        right_layout.addWidget(info_group)
        main_layout.addWidget(right_widget)

        main_layout.setStretch(0, 2)
        main_layout.setStretch(1, 1)

    def start_camera(self):
        if self.running:
            return
        # 使用 picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        self.picam2.configure(config)
        self.picam2.start()
        self.running = True
        self.timer.start(30)
        self.result_label.setText("识别中...")

    def stop_camera(self):
        self.timer.stop()
        if self.picam2:
            self.picam2.stop()
            self.picam2 = None
        self.running = False
        self.video_label.clear()
        self.result_label.setText("摄像头已停止")

    def update_frame(self):
        if not self.running or self.picam2 is None:
            return

        frame_rgb = self.picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        parent_name, child_name, confidence, processed_frame = run_recognition_once(frame,
                                                                                    self.db if self.db_ready else None)

        if parent_name == "陌生人":
            self.result_label.setText(f"⚠️ 陌生人 (置信度: {confidence:.2f})")
            self.result_label.setStyleSheet("color: red; font-size: 14pt;")
        elif parent_name == "无人脸":
            self.result_label.setText("未检测到人脸")
            self.result_label.setStyleSheet("color: gray; font-size: 14pt;")
        else:
            if parent_name not in self.recorded_parents:
                self.recorded_parents.add(parent_name)
                if self.db_ready and self.db:
                    self.add_record(parent_name, child_name)
            if child_name:
                self.result_label.setText(f"✅ {parent_name} → {child_name}")
            else:
                self.result_label.setText(f"✅ {parent_name}")
            self.result_label.setStyleSheet("color: green; font-size: 14pt;")

        self.display_frame(processed_frame)

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def add_record(self, parent_name, child_name):
        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S")
        self.table.insertRow(0)
        self.table.setItem(0, 0, QTableWidgetItem(parent_name))
        self.table.setItem(0, 1, QTableWidgetItem(child_name))
        self.table.setItem(0, 2, QTableWidgetItem(now))
        if self.table.rowCount() > 50:
            self.table.removeRow(self.table.rowCount() - 1)

    def closeEvent(self, event):
        self.stop_camera()
        if self.db:
            self.db.close()
        event.accept()


if __name__ == "__main__":
    from utils import init_gpio, cleanup_gpio
    init_gpio()
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    exit_code = app.exec_()
    cleanup_gpio()
    sys.exit(exit_code)