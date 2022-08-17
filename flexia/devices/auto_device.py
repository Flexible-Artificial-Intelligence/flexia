import torch
from typing import Any, Union, Optional, Dict

from . import *
from ..device_utils import initialize_device
from ..enums import DeviceType
from .enums import MemoryUnit


class AutoDevice:
    def __init__(self,
                 device: Optional[Union[str, torch.device]] = None, 
                 unit: Union[MemoryUnit, str] = "MB", 
                 *args, 
                 **kwargs
                 ) -> None:
        self.device = initialize_device(device)
        self.device_type = DeviceType(self.device.type)
        self.device_index = self.device.index
        self.unit = MemoryUnit(unit)
        
        self.__device_name = f"{self.device_type.name}Accelerator"
        self.__device = globals()[self.__device_name](device=self.device, *args, **kwargs) 
        
    @property
    def memory(self) -> float:
        return self.__device.memory
    
    @property
    def memory_usage(self) -> float:
        return self.__device.memory_usage
    
    @property
    def memory_available(self) -> float:
        return self.__device.memory_available
    
    @property
    def name(self) -> str:
        return self.__device.name
    
    @property
    def stats(self) -> Dict[str, Any]:
        return self.__device.stats
    
    @property
    def is_available(self) -> bool:
        return self.__device.is_available