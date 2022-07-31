import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from .accelerator import Accelerator
from ..enums import DeviceType


nvmlInit()

class GPUAccelerator(Accelerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nvml_info = self.__get_nvml_info(index=self.device_index)
        self.device_properties = torch.cuda.get_device_properties(device=self.device)
        
        if self.device_type != DeviceType.GPU:
            raise TypeError(f"Given device's type is not GPU.")
    
    def __get_nvml_info(self, index):
        handle = nvmlDeviceGetHandleByIndex(index)
        info = nvmlDeviceGetMemoryInfo(handle)
        
        return info
    
    @property
    def memory(self):
        return self.nvml_info.total
        
    @property
    def memory_usage(self):
        return self.nvml_info.used

    @property
    def name(self):
        return self.device_properties.name

    @staticmethod
    def is_available(self):
        return torch.cuda.is_available()