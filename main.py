from db_manager import DatabaseManager
from info_manager import InfoManager
from recognition import run_recognition

def main():
    db = DatabaseManager(password='20040619')  # 请修改为实际密码

    while True:
        print("\n========== 幼儿智能接送人脸核验系统 ==========")
        print("1. 信息管理（注册幼儿/家长）")
        print("2. 启动人脸识别（接送核验）")
        print("3. 退出系统")
        choice = input("请选择操作: ")

        if choice == '1':
            manager = InfoManager(db)
            manager.menu()
        elif choice == '2':
            run_recognition()
        elif choice == '3':
            db.close()
            print("系统已退出")
            break
        else:
            print("无效输入，请重新选择")

if __name__ == "__main__":
    main()