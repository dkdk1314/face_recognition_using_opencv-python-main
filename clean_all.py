import os
import shutil
from db_manager import DatabaseManager


def clean():
    # 清理文件夹
    folders = ['faces', 'database1', 'records', 'unknown_faces', 'model']
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)
            print(f"已清空: {folder}")

    # 删除 store1.csv
    if os.path.exists('store1.csv'):
        os.remove('store1.csv')
        print("已删除 store1.csv")

    # 清空数据库表
    db = DatabaseManager(password='20040619')
    db.execute_query("DELETE FROM pickup_records")
    db.execute_query("DELETE FROM stranger_logs")
    db.execute_query("DELETE FROM parents")
    # 如果也想清空幼儿表，取消下一行注释
    # db.execute_query("DELETE FROM children")
    db.close()
    print("数据库相关表已清空")


if __name__ == "__main__":
    clean()