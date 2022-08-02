from .accelerator import Accelerator
from . import *


class AutoAccelerator(Accelerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.__accelerator_name = f"{self.device_type.name}Accelerator"
        self.__accelerator = globals()[self.__accelerator_name](*args, **kwargs) # change
        
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