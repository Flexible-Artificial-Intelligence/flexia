import torch
from typing import Optional, Union
import os

from .import_utils import is_torch_xla_available, is_torch_backend_mps_available


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


def initialize_device(device: Optional[Union[torch.device, str]] = None):
    if device is None:
        if is_cuda_available():
            device = torch.device("cuda:0")
        elif is_tpu_available():
            device = xm.xla_device(n=0)
        elif is_mps_available():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu:0")

    device = torch.device(device)

    return device

def is_cuda_available() -> bool:
    return torch.cuda.is_available()

def is_cpu_available() -> bool:
    return os.cpu_count() > 0

def is_tpu_available() -> bool:
    if is_torch_xla_available():
        devices = xm.get_xla_supported_devices()
        return len(devices) > 0
    else:
        return False
        
def is_mps_available() -> bool:
    if is_torch_backend_mps_available():
        return torch.backends.mps.is_available()
    
    return False