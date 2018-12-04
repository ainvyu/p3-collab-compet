from typing import NamedTuple
import torch

class Config(NamedTuple):
    num_workers: int = 1
    episode_count: int = 4000
    buffer_size = int(1e5)  # replay buffer size
    #mini_batch_size: int = 128
    mini_batch_size: int = 1024
        
class DeviceConfig:
    DEVICE = torch.device('cpu')