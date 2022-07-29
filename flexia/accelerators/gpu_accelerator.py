from .accelerator import Accelerator


class GPUAccelerator(Accelerator):
    def get_memory_usage(self):
        pass
    
    def get_memory(self):
        pass


    def get_device_stats(self):
        pass

    @staticmethod
    def is_available(self):
        pass