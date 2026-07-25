"""
音频提示工具模块
支持在训练过程中播放语音和提示音
"""
import subprocess
import threading
import os


class AudioNotifier:
    """音频通知器，用于播放训练状态的语音提示"""

    def __init__(self, device="plughw:3,0", enabled=True):
        """
        初始化音频通知器

        Args:
            device: ALSA设备路径，默认 plughw:3,0 (USB Camera)
            enabled: 是否启用音频，False则所有播放函数不执行
        """
        self.device = device
        self.enabled = enabled

        # 检查 espeak 是否可用
        try:
            subprocess.run(['which', 'espeak'], check=True, capture_output=True)
            self.has_espeak = True
        except:
            self.has_espeak = False
            print(f"[AudioNotifier] espeak not found, will use beep sounds instead")

    def _play_async(self, func):
        """异步播放，避免阻塞主线程"""
        if not self.enabled:
            return
        thread = threading.Thread(target=func, daemon=True)
        thread.start()

    def _play_beep(self, freq=1000, duration=0.2):
        """播放简单的提示音"""
        try:
            # 使用 speaker-test 播放指定频率的正弦波
            cmd = f'timeout {duration} speaker-test -t sine -f {freq} -D {self.device} >/dev/null 2>&1'
            subprocess.run(cmd, shell=True, timeout=duration+1)
        except Exception as e:
            pass  # 静默失败，不影响训练

    def _play_text(self, text):
        """播放文字转语音"""
        if not self.has_espeak:
            return

        try:
            # espeak 生成语音并通过 aplay 播放
            cmd = f'espeak "{text}" --stdout 2>/dev/null | aplay -D {self.device} 2>/dev/null'
            subprocess.run(cmd, shell=True, timeout=3)
        except:
            pass  # 静默失败

    def play_success(self, step=None):
        """播放成功提示音"""
        def _play():
            if self.has_espeak and step is not None:
                self._play_text(f"Success, step {step}")
            else:
                # 播放双音：高-更高
                self._play_beep(1500, 0.2)
                self._play_beep(2000, 0.3)

        self._play_async(_play)

    def play_failure(self, reason="timeout"):
        """播放失败提示音"""
        def _play():
            if self.has_espeak:
                self._play_text(f"Failed, {reason}")
            else:
                # 播放低音长鸣
                self._play_beep(400, 0.5)

        self._play_async(_play)

    def play_reset(self):
        """播放重置提示音"""
        def _play():
            # 播放短促中音
            self._play_beep(1000, 0.1)

        self._play_async(_play)

    def play_intervention(self):
        """播放人工干预提示音"""
        def _play():
            # 播放两声短促高音
            self._play_beep(1800, 0.1)
            self._play_beep(1800, 0.1)

        self._play_async(_play)


# 全局单例
_global_notifier = None

def get_audio_notifier(device="plughw:3,0", enabled=True):
    """获取全局音频通知器实例"""
    global _global_notifier
    if _global_notifier is None:
        _global_notifier = AudioNotifier(device=device, enabled=enabled)
    return _global_notifier
