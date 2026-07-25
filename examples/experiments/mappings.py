import sys
import os

# 确保项目根和 serl_robot_infra 在 Python path 中
# - 项目根: 使 from serl_robot_infra.xxx 可用 (USB 等)
# - serl_robot_infra/: 使 from franka_env.xxx / from marvin_env.xxx 可用
_examples_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_examples_dir, '../..'))
_serl_infra = os.path.join(_project_root, 'serl_robot_infra')
for _p in (_serl_infra, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
from experiments.marvin_usb_insertion.config import TrainConfig as MarvinUSBInsertionTrainConfig


CONFIG_MAPPING = {
                "usb_pickup_insertion": USBPickupInsertionTrainConfig,
                "marvin_usb_insertion": MarvinUSBInsertionTrainConfig,
               }