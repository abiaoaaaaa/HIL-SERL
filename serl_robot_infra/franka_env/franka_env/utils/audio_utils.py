"""
音频通知工具模块

用于在训练过程中提供声音反馈：
- 任务成功时播放成功提示音
- 任务失败时播放失败提示音
- 支持USB摄像头内置扬声器
"""
import subprocess
import threading
import os
import numpy as np
import wave
import tempfile


class AudioNotifier:
    """音频通知器"""

    def __init__(self, device="plughw:3,0", enabled=True):
        """
        初始化音频通知器

        Args:
            device: ALSA音频设备 (默认: plughw:3,0 for USB Camera)
            enabled: 是否启用音频 (可用于调试时关闭)
        """
        self.device = device
        self.enabled = enabled
        self.temp_dir = tempfile.gettempdir()
        self._ensure_volume()
        self._generate_audio_files()

    def _ensure_volume(self):
        """确保音量设置正确"""
        if not self.enabled:
            return

        try:
            # 设置音量到100%
            subprocess.run(
                ["amixer", "-c", "3", "sset", "PCM", "100%"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1
            )
        except Exception:
            pass  # 静默失败

    def _generate_tone(self, frequency, duration_ms, sample_rate=16000):
        """生成正弦波音调"""
        duration_sec = duration_ms / 1000.0
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
        tone = np.sin(2 * np.pi * frequency * t)

        # 添加淡入淡出，避免爆音
        fade_samples = int(0.01 * sample_rate)  # 10ms淡入淡出
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out

        # 转换为16位整数
        audio = (tone * 32767).astype(np.int16)
        return audio

    def _save_wav(self, audio, filename):
        """保存为WAV文件"""
        filepath = os.path.join(self.temp_dir, filename)
        with wave.open(filepath, 'w') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16位
            wf.setframerate(16000)  # 16kHz
            wf.writeframes(audio.tobytes())
        return filepath

    def _generate_audio_files(self):
        """预生成音频文件"""
        if not self.enabled:
            return

        try:
            # 成功音：C5 -> E5 -> G5
            success_1 = self._generate_tone(523, 150)
            success_2 = self._generate_tone(659, 150)
            success_3 = self._generate_tone(784, 300)
            success_audio = np.concatenate([success_1, success_2, success_3])
            self.success_file = self._save_wav(success_audio, "train_success.wav")

            # 失败音：G4 -> E4 -> C4
            failure_1 = self._generate_tone(392, 200)
            failure_2 = self._generate_tone(330, 200)
            failure_3 = self._generate_tone(262, 400)
            failure_audio = np.concatenate([failure_1, failure_2, failure_3])
            self.failure_file = self._save_wav(failure_audio, "train_failure.wav")

        except Exception as e:
            print(f"[AudioNotifier] 音频文件生成失败: {e}")
            self.enabled = False

    def _play_wav(self, filepath, label=""):
        """播放WAV文件（在后台线程中）"""
        if not self.enabled or not os.path.exists(filepath):
            return

        def _play():
            try:
                # 使用 aplay 播放
                subprocess.run(
                    ["aplay", "-D", self.device, filepath],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2
                )
            except Exception:
                pass  # 静默失败，避免干扰训练

        # 在后台线程中播放，避免阻塞主线程
        thread = threading.Thread(target=_play, daemon=True)
        thread.start()

    def _say(self, text):
        """使用 espeak-ng 语音合成（如果可用）"""
        if not self.enabled:
            return

        def _speak():
            try:
                # 检查 espeak-ng 是否安装
                result = subprocess.run(
                    ["which", "espeak-ng"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if result.returncode == 0:
                    subprocess.run(
                        ["espeak-ng", "-s", "150", text],  # -s 150: 语速150
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env={**os.environ, "AUDIODEV": self.device},
                        timeout=3
                    )
            except Exception:
                pass  # 静默失败

        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()

    def play_success(self, step=None):
        """播放成功提示音"""
        if not self.enabled:
            return

        print(f"🎉 [AudioNotifier] 任务成功! (step={step})")
        self._play_wav(self.success_file, "success")

    def play_failure(self, reason="unknown"):
        """播放失败提示音"""
        if not self.enabled:
            return

        print(f"❌ [AudioNotifier] 任务失败: {reason}")
        self._play_wav(self.failure_file, "failure")

    def play_warning(self, message="warning"):
        """播放警告提示音"""
        if not self.enabled:
            return

        print(f"⚠️  [AudioNotifier] 警告: {message}")
        # 警告音使用失败音的前半部分（快速警告）
        self._play_wav(self.failure_file, "warning")


class DummyAudioNotifier:
    """空音频通知器（用于禁用音频时）"""

    def __init__(self, *args, **kwargs):
        pass

    def play_success(self, **kwargs):
        pass

    def play_failure(self, **kwargs):
        pass

    def play_warning(self, **kwargs):
        pass


def get_audio_notifier(device="plughw:3,0", enabled=True):
    """
    获取音频通知器实例

    Args:
        device: ALSA音频设备
        enabled: 是否启用音频

    Returns:
        AudioNotifier 或 DummyAudioNotifier
    """
    if enabled:
        return AudioNotifier(device=device, enabled=True)
    else:
        return DummyAudioNotifier()


# 示例用法
if __name__ == "__main__":
    import time

    notifier = get_audio_notifier(device="plughw:3,0", enabled=True)

    print("测试成功提示音...")
    notifier.play_success(step=42)
    time.sleep(2)

    print("测试失败提示音...")
    notifier.play_failure(reason="timeout")
    time.sleep(2)

    print("测试警告提示音...")
    notifier.play_warning(message="collision detected")
    time.sleep(2)

    print("测试完成!")
