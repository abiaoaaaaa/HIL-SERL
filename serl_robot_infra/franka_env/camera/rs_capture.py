import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API


class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False, exposure=40000):
        self.name = name
        # 注释掉序列号检查：当多个相机已被占用时，枚举设备会触发 "failed to set power state" 错误
        # pyrealsense2 会在 enable_device() 和 start() 时自动验证序列号
        # assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number
        self.depth = depth
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        self.profile = self.pipe.start(self.cfg)

        # D405 的彩色流来自 Stereo Module，而 D415 使用独立的 RGB Camera。
        # 不能依赖 query_sensors()[0]，必须根据传感器提供的流类型查找。
        device = self.profile.get_device()
        self.s = next(
            (
                sensor
                for sensor in device.query_sensors()
                if any(
                    stream_profile.stream_type() == rs.stream.color
                    for stream_profile in sensor.get_stream_profiles()
                )
            ),
            None,
        )
        if self.s is None:
            self.pipe.stop()
            raise RuntimeError(
                f"{self.name}: no RealSense sensor provides the color stream"
            )

        sensor_name = self.s.get_info(rs.camera_info.name)
        if self.s.supports(rs.option.enable_auto_exposure):
            self.s.set_option(rs.option.enable_auto_exposure, 0)
        if not self.s.supports(rs.option.exposure):
            self.pipe.stop()
            raise RuntimeError(
                f"{self.name}: color sensor {sensor_name!r} does not support exposure"
            )

        exposure_range = self.s.get_option_range(rs.option.exposure)
        if not exposure_range.min <= exposure <= exposure_range.max:
            self.pipe.stop()
            raise ValueError(
                f"{self.name}: exposure={exposure} is outside the supported range "
                f"[{exposure_range.min}, {exposure_range.max}]"
            )

        self.s.set_option(rs.option.exposure, float(exposure))
        actual_exposure = self.s.get_option(rs.option.exposure)
        print(
            f"[RealSense] {self.name}: color_sensor={sensor_name}, "
            f"exposure={actual_exposure:g}, auto_exposure=off"
        )

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            image = np.asarray(color_frame.get_data())
            if self.depth and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)
            else:
                return True, image
        else:
            return False, None

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()
