import torch
from abc import ABC, abstractmethod

from ..enums import DeviceType


class Accelerator(ABC):
    def __init__(self, device):
        self.device = torch.device(device)
        self.device_type = DeviceType(self.device.type)
        self.device_index = self.device.index

    @property
    @abstractmethod
    def memory_usage(self):
        pass
    
    @property
    @abstractmethod
    def memory(self):
        pass
    
    @property
    def memory_free(self):
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
        stats = {
            "name": self.name, 
            "memory": self.memory, 
            "memory_usage": self.memory_usage, 
            "memory_free": self.memory_free
        }
         
        return stats
    
    def __str__(self):
        return f"{self.__class__.__name__}(name='{self.name}', memory={self.memory}, memory_usage={self.memory_usage}, memory_free={self.memory_free})"
    
    __repr__ = __str__