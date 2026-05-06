import pymysql
import threading

class DatabaseManager:
    def __init__(self, host='localhost', user='root', password='your_password', database='kindergarten_system', timeout=3):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.timeout = timeout
        self.connection = None
        self._connect_nonblocking()

    def _connect_nonblocking(self):
        def _connect():
            try:
                self.connection = pymysql.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    connect_timeout=self.timeout,
                    charset='utf8mb4'
                )
                print("数据库连接成功")
            except Exception as e:
                print(f"数据库连接失败: {e}")
                self.connection = None
        thread = threading.Thread(target=_connect)
        thread.daemon = True
        thread.start()
        # 不等待，立即返回

    def is_ready(self):
        return self.connection is not None

    def execute_query(self, query, params=None, commit=True):
        if self.connection is None:
            print("数据库未连接，无法执行查询")
            return None
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params or ())
            if commit and not query.strip().upper().startswith('SELECT'):
                self.connection.commit()
            return cursor
        except Exception as e:
            print(f"查询执行失败: {e}")
            if cursor:
                cursor.close()
            return None

    def fetch_all(self, query, params=None):
        cursor = self.execute_query(query, params, commit=False)
        if cursor:
            try:
                result = cursor.fetchall()
            except Exception:
                result = []
            cursor.close()
            return result
        return []

    def fetch_one(self, query, params=None):
        cursor = self.execute_query(query, params, commit=False)
        if cursor:
            try:
                result = cursor.fetchone()
            except Exception:
                result = None
            cursor.close()
            return result
        return None

    def insert_parent(self, name, phone, relation, child_id, face_photo_path):
        query = "INSERT INTO parents (name, phone, relation, child_id, face_photo_path) VALUES (%s, %s, %s, %s, %s)"
        cursor = self.execute_query(query, (name, phone, relation, child_id, face_photo_path))
        if cursor:
            parent_id = cursor.lastrowid
            cursor.close()
            return parent_id
        return None

    def update_parent_face_encoding(self, parent_id, face_encoding_blob):
        query = "UPDATE parents SET face_encoding = %s WHERE id = %s"
        self.execute_query(query, (face_encoding_blob, parent_id))

    def get_all_parents_with_encodings(self):
        query = "SELECT id, name, child_id, face_encoding FROM parents WHERE face_encoding IS NOT NULL"
        return self.fetch_all(query)

    def insert_pickup_record(self, parent_id, child_id, result, photo_path):
        query = "INSERT INTO pickup_records (parent_id, child_id, result, photo_path) VALUES (%s, %s, %s, %s)"
        self.execute_query(query, (parent_id, child_id, result, photo_path))

    def insert_stranger_log(self, photo_path):
        query = "INSERT INTO stranger_logs (photo_path) VALUES (%s)"
        self.execute_query(query, (photo_path,))

    def get_setting(self, key_name, default=None):
        row = self.fetch_one("SELECT value_int, value_str FROM system_settings WHERE key_name = %s", (key_name,))
        if row:
            return row[0] if row[0] is not None else row[1]
        return default

    def close(self):
        if self.connection:
            self.connection.close()