import psutil
import os
import platform
import subprocess
import re

from .accelerator import Accelerator
from .utils import convert_bytes


class CPUAccelerator(Accelerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.virtual_memory = psutil.virtual_memory()
        self.__name = self.__get_processor_name()
    
    def __get_processor_name(self):
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
    def memory(self):
        memory = convert_bytes(bytes=self.virtual_memory.total, from_unit="KB", to_unit=self.unit)
        return memory
        
    @property
    def memory_usage(self):
        memory_usage = convert_bytes(bytes=self.virtual_memory.used, from_unit="KB", to_unit=self.unit)
        return memory_usage

    @property
    def name(self):
        return self.__name

    @staticmethod
    def is_available():
        return os.cpu_count() > 0