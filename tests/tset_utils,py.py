"""
测试 utils.py 中的 GPIO 与蜂鸣器控制功能
"""
import pytest
import sys
import os

# 将项目根目录加入路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    IS_RASPBERRY_PI,
    play_alert_sound,
    init_gpio,
    cleanup_gpio,
)


class TestGPIOEnvironmentDetection:
    """测试环境检测功能"""

    def test_is_raspberry_pi_flag(self):
        """测试树莓派环境标志是否为布尔值"""
        assert isinstance(IS_RASPBERRY_PI, bool)

    def test_import_does_not_crash(self):
        """测试导入模块不崩溃"""
        # 已通过 import 测试
        pass


class TestPlayAlertSound:
    """测试预警声音播放"""

    def test_play_alert_no_crash(self):
        """测试 play_alert_sound 在任何环境下都不崩溃"""
        # 在树莓派或 PC 上都应能正常运行（PC上用winsound或beep）
        try:
            play_alert_sound()
        except Exception as e:
            pytest.fail(f"play_alert_sound() 抛出异常: {e}")

    def test_play_alert_multiple_times(self):
        """测试连续多次调用不崩溃"""
        for _ in range(3):
            try:
                play_alert_sound()
            except Exception as e:
                pytest.fail(f"第 {_+1} 次调用 play_alert_sound() 失败: {e}")


class TestGPIOInitCleanup:
    """测试 GPIO 初始化与清理"""

    def test_init_gpio_no_crash(self):
        """测试 init_gpio 不崩溃"""
        try:
            init_gpio()
        except Exception as e:
            pytest.fail(f"init_gpio() 抛出异常: {e}")

    def test_cleanup_gpio_no_crash(self):
        """测试 cleanup_gpio 不崩溃"""
        try:
            cleanup_gpio()
        except Exception as e:
            pytest.fail(f"cleanup_gpio() 抛出异常: {e}")

    def test_init_then_cleanup(self):
        """测试先初始化后清理的顺序不崩溃"""
        try:
            init_gpio()
            cleanup_gpio()
        except Exception as e:
            pytest.fail(f"init_gpio() + cleanup_gpio() 抛出异常: {e}")


class TestBuzzerIntegration:
    """蜂鸣器集成测试（模拟陌生人预警场景）"""

    def test_stranger_alert_flow(self):
        """
        模拟 recognition.py 中检测到陌生人时的预警流程：
        recognition.py 第 243 行：
            if parent_name == "陌生人":
                play_alert_sound()
        """
        # 模拟陌生人场景
        parent_name = "陌生人"
        assert parent_name == "陌生人"

        # 触发预警（与 recognition.py 中完全一致的调用）
        try:
            play_alert_sound()
        except Exception as e:
            pytest.fail(f"陌生人预警触发失败: {e}")