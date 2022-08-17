import psutil
import os
import platform
import subprocess
import re

from .device import Device
from .utils import convert_bytes
from ..device_utils import is_cpu_available


class CPUDevice(Device):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.virtual_memory = psutil.virtual_memory() # bytes
        self.__name = self.__get_processor_name()
    
    def __get_processor_name(self) -> str:
        """
        https://stackoverflow.com/a/13078519
        """

        if platform.system() == "Windows":
            return platform.processor().strip()
        elif platform.system() == "Darwin":
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            command ="sysctl -n machdep.cpu.brand_string"
            return subprocess.check_output(command).strip()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).decode().strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return re.sub( ".*model name.*:", "", line,1).strip()
        return ""
    
    @property
    def memory(self) -> float:
        memory = convert_bytes(bytes=self.virtual_memory.total, from_unit="B", to_unit=self.unit)
        return memory
        
    @property
    def memory_usage(self):
        memory_usage = convert_bytes(bytes=self.virtual_memory.used, from_unit="B", to_unit=self.unit)
        return memory_usage

    @property
    def name(self):
        return self.__name

    @staticmethod
    def is_available():
        return is_cpu_available()