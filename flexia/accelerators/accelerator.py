import torch
from abc import ABC, abstractmethod

from ..enums import DeviceType
from .enums import MemoryUnit
from ..third_party.addict import Dict


class Accelerator(ABC):
    def __init__(self, device, unit="MB"):
        self.device = torch.device(device)
        self.device_type = DeviceType(self.device.type)
        self.device_index = self.device.index
        self.unit = MemoryUnit(unit)

    @property
    def memory_usage(self):
        pass
    
    @property
    @abstractmethod
    def memory(self):
        pass
    
    @property
    def memory_available(self):
        return self.memory - self.memory_usage
    
    @property
    @abstractmethod
    def name(self):
        pass
    
    @property
    @abstractmethod
    def stats(self):
        pass

    @staticmethod
    @abstractmethod
    def is_available():
        pass
    
    @property
    def stats(self):
        stats = Dict({
            "name": self.name, 
            "memory": self.memory, 
            "memory_usage": self.memory_usage, 
            "memory_available": self.memory_available
        })
         
        return stats
    
    def __str__(self):
        return f"{self.__class__.__name__}(name='{self.name}', memory={self.memory}, memory_usage={self.memory_usage}, memory_available={self.memory_available})"
    
    __repr__ = __str__