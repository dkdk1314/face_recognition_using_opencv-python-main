#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

# 引脚配置（与您代码中的 GPIO23 一致）
BUZZER_PIN = 23


def setup():
    """初始化 GPIO"""
    GPIO.setmode(GPIO.BCM)  # 使用 BCM 编号
    GPIO.setup(BUZZER_PIN, GPIO.OUT)  # 设置为输出模式
    GPIO.output(BUZZER_PIN, GPIO.LOW)  # 初始低电平（不响）


def buzzer_on():
    """蜂鸣器鸣响（高电平触发）"""
    GPIO.output(BUZZER_PIN, GPIO.HIGH)


def buzzer_off():
    """蜂鸣器停止"""
    GPIO.output(BUZZER_PIN, GPIO.LOW)


def test():
    """测试蜂鸣器"""
    print("蜂鸣器测试开始...")
    print(f"使用 GPIO{BUZZER_PIN} (BCM 编号)，对应物理引脚 16")
    print("注意：您的蜂鸣器模块应为高电平触发")
    print("-" * 30)

    # 测试1：鸣响 0.5 秒
    print("测试1: 鸣响 0.5 秒")
    buzzer_on()
    time.sleep(0.5)
    buzzer_off()
    time.sleep(0.5)

    # 测试2：鸣响 1 秒
    print("测试2: 鸣响 1 秒")
    buzzer_on()
    time.sleep(1.0)
    buzzer_off()
    time.sleep(0.5)

    # 测试3：连续鸣响3次（每次0.3秒）
    print("测试3: 连续鸣响3次")
    for i in range(3):
        print(f"  第 {i + 1} 次")
        buzzer_on()
        time.sleep(0.3)
        buzzer_off()
        time.sleep(0.3)

    print("-" * 30)
    print("测试完成！")
    print("如果听到了声音，说明蜂鸣器硬件和 GPIO 控制正常。")
    print("如果没有声音，请检查：")
    print("  1. 蜂鸣器模块是否连接到正确的引脚（VCC→5V, GND→GND, I/O→GPIO23）")
    print("  2. 是否为高电平触发（某些模块可能是低电平触发，需要修改代码）")
    print("  3. 蜂鸣器模块是否损坏")


def cleanup():
    """清理 GPIO"""
    GPIO.cleanup()
    print("GPIO 已清理")


if __name__ == "__main__":
    try:
        setup()
        test()
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        cleanup()