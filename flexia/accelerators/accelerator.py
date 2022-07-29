from abc import ABC, abstractmethod


class Accelerator(ABC):
    def __init__(self, device):
        self.device = device

    @abstractmethod
    def get_memory_usage(self):
        pass
    
    @abstractmethod
    def get_memory(self):
        pass

    @abstractmethod
    def get_device_stats(self):
        pass

    @staticmethod
    @abstractmethod
    def is_available(self):
        pass
