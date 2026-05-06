import sys
import time

# 尝试导入 RPi.GPIO，如果失败则标记为非树莓派环境
try:
    import RPi.GPIO as GPIO
    IS_RASPBERRY_PI = True
except (ImportError, RuntimeError):
    IS_RASPBERRY_PI = False
    GPIO = None

# GPIO 引脚定义
BUZZER_PIN = 23   # 蜂鸣器信号引脚（高电平触发）

# 内部标志，记录是否已初始化 GPIO
_gpio_initialized = False

def _ensure_gpio_initialized():
    """确保 GPIO 模式已设置（自动调用，无需显式调用）"""
    global _gpio_initialized
    if not IS_RASPBERRY_PI:
        return
    if not _gpio_initialized:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
            _gpio_initialized = True
            print("GPIO 自动初始化成功")
        except Exception as e:
            print(f"GPIO 自动初始化失败: {e}")

def init_gpio():
    """手动初始化 GPIO（兼容旧调用）"""
    _ensure_gpio_initialized()

def cleanup_gpio():
    """清理 GPIO 资源"""
    global _gpio_initialized
    if not IS_RASPBERRY_PI:
        return
    if _gpio_initialized:
        try:
            GPIO.cleanup()
            _gpio_initialized = False
            print("GPIO 清理完成")
        except:
            pass

def play_alert_sound():
    """播放预警声音（树莓派上通过 GPIO 控制蜂鸣器，PC 上通过系统蜂鸣声）"""
    if IS_RASPBERRY_PI:
        # 确保 GPIO 已初始化
        _ensure_gpio_initialized()
        # 高电平触发蜂鸣器，持续 0.5 秒
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        print("蜂鸣器已触发")  # 调试输出，可删除
    else:
        # PC 环境：使用系统蜂鸣声或打印提示
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 500)
        else:
            try:
                import subprocess
                subprocess.run(['beep', '-f', '1000', '-l', '500'], check=False)
            except:
                print('\a')   # 终端响铃字符
