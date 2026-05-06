import cv2
import os
import csv
import numpy as np
import pickle
import shutil
import time
import datetime
from db_manager import DatabaseManager
from weighted_lbph import WeightedLBPH, create_default_weights

# 尝试导入 picamera2，若失败则标记为不可用
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("警告: picamera2 未安装，将使用 OpenCV 摄像头（仅适用于 USB 摄像头）")

class InfoManager:
    def __init__(self, db):
        self.db = db
        self.face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
        self.database_folder = "database1"
        self.csv_file = "store1.csv"
        self.ensure_directories()

    def ensure_directories(self):
        os.makedirs(self.database_folder, exist_ok=True)
        os.makedirs("faces", exist_ok=True)
        os.makedirs("records", exist_ok=True)
        os.makedirs("unknown_faces", exist_ok=True)
        os.makedirs("model", exist_ok=True)

    # ---------- 摄像头拍照（使用 picamera2）----------
    def _init_camera(self):
        """初始化 picamera2，返回相机对象"""
        if not PICAMERA_AVAILABLE:
            raise RuntimeError("picamera2 不可用")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        return picam2

    def capture_face_frame(self):
        """捕获一帧包含人脸的图像，返回 (success, frame)"""
        try:
            picam2 = self._init_camera()
        except Exception as e:
            print(f"无法打开摄像头: {e}")
            return False, None

        print("请将人脸对准摄像头，按空格键拍照，按 ESC 取消...")
        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Capture Face", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                if len(faces) == 0:
                    print("未检测到人脸，请重新拍照")
                    continue
                elif len(faces) > 1:
                    print("检测到多个人脸，请确保只有您一人")
                    continue
                picam2.stop()
                cv2.destroyAllWindows()
                return True, frame_bgr
            elif key == 27:
                print("取消拍照")
                break
        picam2.stop()
        cv2.destroyAllWindows()
        return False, None

    def capture_multiple_faces(self, count=5):
        """连续拍摄多张人脸照片，返回照片文件路径列表"""
        if count <= 0:
            return []
        try:
            picam2 = self._init_camera()
        except Exception as e:
            print(f"无法打开摄像头: {e}")
            return []

        saved_paths = []
        print(f"请准备拍摄 {count} 张人脸照片，按空格键拍摄，按 ESC 取消...")
        for i in range(count):
            print(f"\n拍摄第 {i+1}/{count} 张，请调整姿势/角度...")
            while True:
                frame_rgb = picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("Capture Face", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                    if len(faces) == 0:
                        print("未检测到人脸，请重新拍照")
                        continue
                    elif len(faces) > 1:
                        print("检测到多个人脸，请确保只有您一人")
                        continue
                    timestamp = int(time.time() * 1000)
                    photo_path = os.path.join("faces", f"temp_{timestamp}.jpg")
                    cv2.imwrite(photo_path, frame_bgr)
                    saved_paths.append(photo_path)
                    print(f"第 {i+1} 张照片已保存")
                    break
                elif key == 27:
                    print("取消拍摄")
                    picam2.stop()
                    cv2.destroyAllWindows()
                    return []
            cv2.waitKey(1000)  # 等待1秒调整姿势
        picam2.stop()
        cv2.destroyAllWindows()
        return saved_paths

    # ---------- 训练数据管理 ----------
    def update_lbph_training_data(self, parent_name, photo_path):
        timestamp = int(time.time())
        base_name = f"parent_{timestamp}.jpg"
        dest_path = os.path.join(self.database_folder, base_name)
        while os.path.exists(dest_path):
            timestamp += 1
            base_name = f"parent_{timestamp}.jpg"
            dest_path = os.path.join(self.database_folder, base_name)
        shutil.copyfile(photo_path, dest_path)
        print(f"已复制照片到: {dest_path}")

        file_exists = os.path.isfile(self.csv_file)
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['image_path', 'name'])
            writer.writerow([base_name, parent_name])
        print(f"已更新 {self.csv_file}，添加 {parent_name} -> {base_name}")

    def retrain_lbph_model(self):
        """使用加权 LBPH 训练模型并保存"""
        images = []
        labels = []
        label_names = []
        name_to_id = {}
        current_id = 0

        if not os.path.exists(self.csv_file):
            print("没有找到 store1.csv，无法训练")
            return False

        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(self.database_folder, row['image_path'])
                name = row['name']
                if not os.path.exists(img_path):
                    print(f"警告: 图片 {img_path} 不存在，跳过")
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                if len(faces) == 0:
                    print(f"警告: 在 {img_path} 中未检测到人脸，跳过")
                    continue
                (x, y, w, h) = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                images.append(face_roi)
                if name not in name_to_id:
                    name_to_id[name] = current_id
                    label_names.append(name)
                    current_id += 1
                labels.append(name_to_id[name])
                print(f"训练数据加载: {name}")

        if len(images) == 0:
            print("没有有效的人脸图片，无法训练模型")
            return False

        grid_size = (8, 8)
        weights = create_default_weights(grid_size)
        recognizer = WeightedLBPH(grid=grid_size, weights=weights)
        recognizer.train(images, labels, label_names)

        os.makedirs("model", exist_ok=True)
        recognizer.save("model/weighted_lbph.pkl")
        with open("model/name_to_id.pkl", "wb") as f:
            pickle.dump(name_to_id, f)
        with open("model/label_names.pkl", "wb") as f:
            pickle.dump(label_names, f)
        print(f"加权 LBPH 模型训练完成，共 {len(label_names)} 人")
        return True

    # ---------- 幼儿管理 ----------
    def add_child(self, name, class_name, enrollment_date):
        query = "INSERT INTO children (name, class_name, enrollment_date) VALUES (%s, %s, %s)"
        cursor = self.db.execute_query(query, (name, class_name, enrollment_date))
        if cursor:
            child_id = cursor.lastrowid
            cursor.close()
            print(f"幼儿 {name} 添加成功，ID: {child_id}")
            return child_id
        return None

    # ---------- 家长管理 ----------
    def add_parent(self, name, phone, relation, child_name):
        """添加家长，连续拍摄多张照片并自动训练"""
        try:
            child_row = self.db.fetch_one("SELECT id FROM children WHERE name = %s", (child_name,))
            if not child_row:
                print(f"错误：找不到名为 {child_name} 的幼儿")
                return None
            child_id = child_row[0]

            query = "INSERT INTO parents (name, phone, relation, child_id, face_photo_path) VALUES (%s, %s, %s, %s, %s)"
            cursor = self.db.execute_query(query, (name, phone, relation, child_id, None))
            if not cursor:
                print("插入家长基本信息失败")
                return None
            parent_id = cursor.lastrowid
            cursor.close()

            photo_paths = self.capture_multiple_faces(count=5)
            if not photo_paths:
                print("未拍到有效照片，添加失败")
                self.db.execute_query("DELETE FROM parents WHERE id = %s", (parent_id,))
                return None

            cover_photo = f"faces/parent_{parent_id}_cover.jpg"
            cv2.imwrite(cover_photo, cv2.imread(photo_paths[0]))
            self.db.execute_query("UPDATE parents SET face_photo_path = %s WHERE id = %s", (cover_photo, parent_id))

            for photo_path in photo_paths:
                self.update_lbph_training_data(name, photo_path)

            self.retrain_lbph_model()
            print(f"家长 {name} 添加完成，共采集 {len(photo_paths)} 张照片")
            return parent_id
        except Exception as e:
            print(f"添加家长过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ---------- 删除家长（带物理文件清理）----------
    def delete_parent(self):
        rows = self.db.fetch_all("SELECT id, name, face_photo_path FROM parents")
        if not rows:
            print("暂无家长数据")
            return
        print("现有家长：")
        for idx, (pid, name, photo_path) in enumerate(rows, 1):
            print(f"  {idx}. {name} (ID: {pid})")
        try:
            sel = int(input("请选择要删除的家长序号: "))
            if sel < 1 or sel > len(rows):
                print("无效选择")
                return
            parent_id, parent_name, photo_path = rows[sel-1]
        except ValueError:
            print("输入无效")
            return

        confirm = input(f"确定要删除家长 {parent_name} 及其所有照片、模型吗？(y/n): ")
        if confirm.lower() != 'y':
            print("取消删除")
            return

        import glob
        for f in glob.glob(f"faces/parent_{parent_id}*"):
            try:
                os.remove(f)
                print(f"已删除: {f}")
            except Exception as e:
                print(f"删除失败: {f} - {e}")

        if os.path.exists(self.csv_file):
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            new_lines = [lines[0]] if lines else []
            removed_files = []
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[1] == parent_name:
                    removed_files.append(parts[0])
                    continue
                new_lines.append(line)
            with open(self.csv_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            for filename in removed_files:
                train_path = os.path.join(self.database_folder, filename)
                if os.path.exists(train_path):
                    os.remove(train_path)
                    print(f"已删除训练照片: {train_path}")

        self.db.execute_query("DELETE FROM pickup_records WHERE parent_id = %s", (parent_id,))
        self.db.execute_query("DELETE FROM parents WHERE id = %s", (parent_id,))
        print(f"家长 {parent_name} 已从数据库删除")

        remaining = self.db.fetch_all("SELECT id FROM parents")
        if not remaining:
            print("目前没有任何家长，正在删除模型文件...")
            for f in glob.glob("model/*.pkl") + glob.glob("model/*.yml"):
                try:
                    os.remove(f)
                    print(f"已删除模型文件: {f}")
                except Exception as e:
                    print(f"删除失败: {f} - {e}")
            return

        answer = input("是否立即重新训练模型？(y/n): ").lower()
        if answer == 'y':
            self.retrain_lbph_model()
        else:
            print("请记得在信息管理菜单选择“5. 重新训练 LBPH 模型”以更新识别模型。")

    # ---------- 删除幼儿 ----------
    def delete_child(self):
        rows = self.db.fetch_all("SELECT id, name FROM children")
        if not rows:
            print("暂无幼儿数据")
            return
        print("现有幼儿：")
        for idx, (cid, cname) in enumerate(rows, 1):
            parent_count = self.db.fetch_one("SELECT COUNT(*) FROM parents WHERE child_id = %s", (cid,))[0]
            print(f"  {idx}. {cname} (ID: {cid}, 家长数: {parent_count})")
        try:
            sel = int(input("请选择要删除的幼儿序号: "))
            if sel < 1 or sel > len(rows):
                print("无效选择")
                return
            child_id, child_name = rows[sel-1]
        except ValueError:
            print("输入无效")
            return

        confirm = input(f"确定要删除幼儿 {child_name} 及其所有关联数据吗？(y/n): ")
        if confirm.lower() != 'y':
            print("取消删除")
            return

        parents = self.db.fetch_all("SELECT id, name, face_photo_path FROM parents WHERE child_id = %s", (child_id,))
        for pid, pname, photo_path in parents:
            if photo_path and os.path.exists(photo_path):
                os.remove(photo_path)
                print(f"已删除家长照片: {photo_path}")
            if os.path.exists(self.csv_file):
                with open(self.csv_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                new_lines = [lines[0]] if lines else []
                removed_file = None
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 2 and parts[1] == pname:
                        removed_file = parts[0]
                        continue
                    new_lines.append(line)
                with open(self.csv_file, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                if removed_file:
                    train_path = os.path.join(self.database_folder, removed_file)
                    if os.path.exists(train_path):
                        os.remove(train_path)
                        print(f"已删除训练照片: {train_path}")

        self.db.execute_query("DELETE FROM pickup_records WHERE child_id = %s", (child_id,))
        self.db.execute_query("DELETE FROM parents WHERE child_id = %s", (child_id,))
        self.db.execute_query("DELETE FROM children WHERE id = %s", (child_id,))
        print(f"幼儿 {child_name} 及其所有关联数据已删除")

        remaining = self.db.fetch_all("SELECT id FROM parents")
        if not remaining:
            print("目前没有任何家长，请添加家长后手动选择“重新训练 LBPH 模型”。")
            import glob
            for f in glob.glob("model/*.pkl") + glob.glob("model/*.yml"):
                try:
                    os.remove(f)
                    print(f"已删除无效模型文件: {f}")
                except Exception as e:
                    print(f"删除文件 {f} 失败: {e}")
        else:
            answer = input("是否立即重新训练模型？(y/n): ").lower()
            if answer == 'y':
                self.retrain_lbph_model()

    # ---------- 导出接送记录 ----------
    def export_records_to_csv(self):
        import csv
        from datetime import datetime
        print("\n===== 导出接送记录 =====")
        print("1. 导出全部记录")
        print("2. 按日期范围导出")
        print("3. 导出今天的记录")
        choice = input("请选择: ")
        query = """
            SELECT pr.id, p.name, c.name, pr.verify_time, pr.result, pr.photo_path
            FROM pickup_records pr
            JOIN parents p ON pr.parent_id = p.id
            JOIN children c ON pr.child_id = c.id
        """
        params = []
        if choice == '2':
            start = input("开始日期 (YYYY-MM-DD): ")
            end = input("结束日期 (YYYY-MM-DD): ")
            query += " WHERE DATE(pr.verify_time) BETWEEN %s AND %s"
            params = [start, end]
        elif choice == '3':
            today = datetime.now().strftime("%Y-%m-%d")
            query += " WHERE DATE(pr.verify_time) = %s"
            params = [today]
        query += " ORDER BY pr.verify_time DESC"
        rows = self.db.fetch_all(query, params)
        if not rows:
            print("暂无符合条件的记录")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if choice == '2':
            filename = f"pickup_records_{start}_to_{end}.csv"
        elif choice == '3':
            filename = f"pickup_records_{today}.csv"
        else:
            filename = f"pickup_records_{timestamp}.csv"
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['记录ID', '家长姓名', '幼儿姓名', '核验时间', '核验结果', '抓拍路径'])
            writer.writerows(rows)
        print(f"✅ 已导出 {len(rows)} 条记录到 {filename}")

    # ---------- 系统设置菜单 ----------
    def system_settings_menu(self):
        while True:
            camera_index = self.db.get_setting('camera_index', 0)
            rec_th = self.db.get_setting('recognition_threshold', 80)
            peak_th = self.db.get_setting('peak_mode_threshold', 100)
            print("\n===== 系统设置 =====")
            print(f"1. 摄像头索引 (当前: {camera_index})")
            print(f"2. 平峰模式识别阈值 (当前: {rec_th})")
            print(f"3. 高峰模式识别阈值 (当前: {peak_th})")
            print("4. 恢复默认设置")
            print("5. 返回上级菜单")
            ch = input("请选择: ")
            if ch == '1':
                try:
                    new = int(input("新摄像头索引: "))
                    self.db.execute_query("UPDATE system_settings SET value_int = %s WHERE key_name = 'camera_index'", (new,))
                    print("已更新，下次启动人脸识别时生效。")
                except: print("无效输入")
            elif ch == '2':
                try:
                    new = int(input("新平峰阈值(0-200): "))
                    self.db.execute_query("UPDATE system_settings SET value_int = %s WHERE key_name = 'recognition_threshold'", (new,))
                    print("已更新")
                except: print("无效输入")
            elif ch == '3':
                try:
                    new = int(input("新高阀阈值(0-200): "))
                    self.db.execute_query("UPDATE system_settings SET value_int = %s WHERE key_name = 'peak_mode_threshold'", (new,))
                    print("已更新")
                except: print("无效输入")
            elif ch == '4':
                confirm = input("恢复默认设置？(y/n): ")
                if confirm.lower() == 'y':
                    self.db.execute_query("UPDATE system_settings SET value_int = 0 WHERE key_name = 'camera_index'")
                    self.db.execute_query("UPDATE system_settings SET value_int = 80 WHERE key_name = 'recognition_threshold'")
                    self.db.execute_query("UPDATE system_settings SET value_int = 100 WHERE key_name = 'peak_mode_threshold'")
                    print("已恢复默认")
            elif ch == '5':
                break
            else:
                print("无效选择")

    # ---------- 主菜单 ----------
    def menu(self):
        while True:
            print("\n===== 信息管理菜单 =====")
            print("1. 添加幼儿")
            print("2. 添加家长并绑定幼儿")
            print("3. 查看所有幼儿")
            print("4. 查看所有家长")
            print("5. 重新训练 LBPH 模型")
            print("6. 删除家长")
            print("7. 删除幼儿")
            print("8. 导出接送记录")
            print("9. 系统设置")
            print("10. 返回主菜单")
            choice = input("请选择操作: ")

            if choice == '1':
                name = input("幼儿姓名: ")
                cls = input("班级: ")
                date = input("入学日期 (YYYY-MM-DD): ")
                self.add_child(name, cls, date)
            elif choice == '2':
                rows = self.db.fetch_all("SELECT id, name FROM children")
                if not rows:
                    print("请先添加幼儿")
                    continue
                unique = {}
                for cid, cname in rows:
                    if cname not in unique:
                        unique[cname] = cid
                print("可选幼儿（去重）:")
                for i, (cname, cid) in enumerate(unique.items(), 1):
                    print(f"  {i}. {cname} (ID: {cid})")
                child_name = input("请输入幼儿姓名: ")
                if child_name not in unique:
                    print("错误：幼儿不存在")
                    continue
                parent_name = input("家长姓名: ")
                phone = input("联系电话: ")
                relation = input("关系: ")
                self.add_parent(parent_name, phone, relation, child_name)
            elif choice == '3':
                rows = self.db.fetch_all("SELECT * FROM children")
                for row in rows:
                    print(row)
            elif choice == '4':
                rows = self.db.fetch_all("SELECT id, name, phone, relation, child_id, face_photo_path FROM parents")
                if not rows:
                    print("暂无家长数据")
                else:
                    for row in rows:
                        print(f"ID: {row[0]}, 姓名: {row[1]}, 电话: {row[2]}, 关系: {row[3]}, 幼儿ID: {row[4]}, 照片: {row[5]}")
            elif choice == '5':
                self.retrain_lbph_model()
            elif choice == '6':
                self.delete_parent()
            elif choice == '7':
                self.delete_child()
            elif choice == '8':
                self.export_records_to_csv()
            elif choice == '9':
                self.system_settings_menu()
            elif choice == '10':
                break
            else:
                print("无效输入")
