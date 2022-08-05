import sys
sys.path.append("./")

from flexia.accelerators.cpu_accelerator import CPUAccelerator
from flexia.loggers.utils import format_accelerator_stats


accelerator = CPUAccelerator(device="cpu:0", unit="MB")
print(format_accelerator_stats(accelerator=accelerator))