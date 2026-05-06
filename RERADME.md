幼儿智能接送人脸核验系统
 
基于 OpenCV + PyQt5 的人脸识别接送系统，适用于树莓派等嵌入式平台。
 
本项目配套完整的自动化测试套件，用于测试开发岗位能力展示。
 
 
 
功能简介
 
- 幼儿信息管理（添加、删除、查询）
- 家长人脸信息绑定与管理（多角度采集）
- 接送时段实时人脸核验
- 陌生人预警（蜂鸣器 + 界面提示）
- 接送记录存储与导出
- 系统支持树莓派平台部署（含 GPIO 硬件控制）
 
 
 
技术栈
 
层级 技术 
编程语言 Python 3.8+ 
图形界面 PyQt5 
人脸检测 OpenCV Haar Cascade + DNN (SSD) 
人脸识别 自研加权 LBPH 算法（WeightedLBPH） 
图像预处理 白平衡、肤色自适应、Gamma校正、CLAHE 
数据库 MySQL (PyMySQL) 
硬件平台 树莓派 4B, USB摄像头, GPIO蜂鸣器 
自动化测试 Pytest, Allure 
 
 
 
自动化测试
 
本项目使用 Pytest 框架编写了完整的单元测试和集成测试，用于展示测试开发能力。
 
测试覆盖范围
 
测试文件 覆盖模块 测试内容 
 test_weighted_lbph.py  加权 LBPH 识别算法 模型初始化、高斯权重生成、训练与预测、保存与加载 
 test_recognition.py  图像预处理与人脸检测 白平衡、轮廓增强、肤色自适应、人脸检测、预处理管道 
 test_utils.py  GPIO 与蜂鸣器控制 环境检测、预警声音播放、GPIO 初始化与清理、陌生人预警流程 
 
运行测试
 
bash
  
# 安装依赖
pip install -r requirements.txt

# 运行全部测试
pytest tests/ -v

# 生成 Allure 可视化测试报告（需安装 allure 命令行工具）
pytest tests/ --alluredir=allure-results
allure generate allure-results -o allure-report --clean
allure open allure-report
 
 
测试报告样例
 
文件名：test_report.png（测试结果截图，存放于项目根目录）
 
 
 
部署说明
 
硬件要求
 
- 树莓派 4B 或同等性能开发板
- USB 摄像头
- 蜂鸣器模块（GPIO23，高电平触发）
- 7寸 LCD 触摸屏（可选）
 
软件环境
 
- Python 3.8+
- MySQL 数据库
- OpenCV 4.5+
 
安装步骤
 
1. 克隆仓库
bash
  
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
 
2. 安装 Python 依赖
bash
  
pip install -r requirements.txt
 
3. 下载 DNN 人脸检测模型（可选，用于提升检测精度）
bash
  
python download_models.py
 
4. 配置数据库
- 创建 MySQL 数据库 kindergarten_system
- 修改 db_manager.py 中的数据库连接参数（host, user, password）
5. 准备测试数据（用于自动化测试）
- 在项目根目录放入一张人脸照片，命名为 test_face.jpg
6. 运行主程序
bash
  
python main.py
 
 
 
 
项目结构
 
plaintext
  
.
├── main.py                 # 命令行入口
├── gui_main.py             # PyQt5 图形界面入口
├── recognition.py          # 人脸识别核心流程
├── weighted_lbph.py        # 加权 LBPH 算法实现
├── info_manager.py         # 数据管理（注册/训练）
├── db_manager.py           # 数据库操作封装
├── utils.py                # GPIO 与蜂鸣器控制
├── download_models.py      # DNN 模型下载脚本
├── clean_all.py            # 数据清理工具
├── check_model.py          # 模型信息查看工具
├── tests/                  # 自动化测试脚本
│   ├── conftest.py
│   ├── test_weighted_lbph.py
│   ├── test_recognition.py
│   └── test_utils.py
├── models/                 # DNN 人脸检测模型文件
├── database1/              # 训练数据存储
├── model/                  # 训练好的识别模型
├── requirements.txt        # Python 依赖列表
├── test_face.jpg           # 测试用图片
├── test_report.png         # 测试报告截图
└── README.md
 
 
 
 
作者
 
- 长沙师范学院 电子信息工程 2026届
- 求职意向：测试开发工程师 / 自动化测试工程师