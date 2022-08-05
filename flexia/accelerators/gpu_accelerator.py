import torch

from .accelerator import Accelerator
from ..enums import DeviceType
from .utils import convert_bytes
from ..import_utils import is_pynvml_available


if is_pynvml_available():
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
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
        memory = convert_bytes(bytes=self.nvml_info.total, from_unit="B", to_unit=self.unit)
        return memory
        
    @property
    def memory_usage(self):
        memory_usage = convert_bytes(bytes=self.nvml_info.used, from_unit="B", to_unit=self.unit)
        return memory_usage

    @property
    def name(self):
        return self.device_properties.name

    @staticmethod
    def is_available():
        return torch.cuda.is_available()