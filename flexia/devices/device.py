import torch
from typing import Any, Optional, Union, Dict as TypingDict
from abc import ABC, abstractmethod

from ..device_utils import initialize_device
from ..enums import DeviceType
from .enums import MemoryUnit
from ..third_party.addict import Dict


class Device(ABC):
    def __init__(self, 
                device: Optional[Union[str, torch.device]] = None, 
                unit: Union[MemoryUnit, str] = "MB"
                ) -> None:
        self.device = initialize_device(device)
        self.device_type = DeviceType(self.device.type)
        self.device_index = self.device.index if self.device.index is not None else 0
        self.unit = MemoryUnit(unit)

    @property
    def memory_usage(self) -> float:
        pass
    
    @property
    @abstractmethod
    def memory(self) -> float:
        pass
    
    @property
    def memory_available(self) -> float:
        return self.memory - self.memory_usage
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        pass
    
    @property
    def stats(self) -> TypingDict[str, Any]:
        stats = Dict({
            "name": self.name, 
            "memory": self.memory, 
            "memory_usage": self.memory_usage, 
            "memory_available": self.memory_available
        })
         
        return stats
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', memory={self.memory}, memory_usage={self.memory_usage}, memory_available={self.memory_available})"
    
    __repr__ = __str__