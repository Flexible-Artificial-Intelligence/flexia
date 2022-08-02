from . import *
from ..utils import initialize_device
from ..enums import DeviceType


class AutoAccelerator:
    def __init__(self, device=None, *args, **kwargs):
        self.device = initialize_device(device)
        self.device_type = DeviceType(self.device.type)
        self.device_index = self.device.index

        self.__accelerator_name = f"{self.device_type.name}Accelerator"
        self.__accelerator = globals()[self.__accelerator_name](device=self.device, *args, **kwargs) 
        
    @property
    def memory(self):
        return self.__accelerator.memory
    
    @property
    def memory_usage(self):
        return self.__accelerator.memory_usage
    
    @property
    def memory_available(self):
        return self.__accelerator.memory_available
    
    @property
    def name(self):
        return self.__accelerator.name
    
    @property
    def stats(self):
        return self.__accelerator.stats
    
    @property
    def is_available(self):
        return self.__accelerator.is_available