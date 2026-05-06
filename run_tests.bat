@echo off
echo ===== 安装依赖 =====
pip install -r requirements.txt

echo.
echo ===== 清理旧报告 =====
if exist allure-results rmdir /s /q allure-results
if exist allure-report rmdir /s /q allure-report

echo.
echo ===== 运行测试 =====
pytest tests/ -v --alluredir=allure-results

echo.
echo ===== 生成 Allure 报告 =====
allure generate allure-results -o allure-report --clean

echo.
echo ===== 启动报告服务器 =====
allure open allure-report