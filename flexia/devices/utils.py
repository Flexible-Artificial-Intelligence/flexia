from typing import Union

from .enums import MemoryUnit


# bytes converters
convert_bytes_to_kilobytes = lambda bytes: bytes / 1024
convert_bytes_to_megabytes = lambda bytes: bytes / 1024 / 1024
convert_bytes_to_gigabytes = lambda bytes: bytes / 1024 / 1024 / 1024
convert_bytes_to_terabytes = lambda bytes: bytes / 1024 / 1024 / 1024 / 1024


# kilobytes converters
convert_kilobytes_to_bytes = lambda bytes: bytes * 1024
convert_kilobytes_to_megabytes = lambda bytes: bytes / 1024
convert_kilobytes_to_gigabytes = lambda bytes: bytes / 1024 / 1024
convert_kilobytes_to_terabytes = lambda bytes: bytes / 1024 / 1024 / 1024 

# megabytes converters
convert_megabytes_to_bytes = lambda bytes: bytes * 1024 * 1024
convert_megabytes_to_kilobytes = lambda bytes: bytes * 1024
convert_megabytes_to_gigabytes = lambda bytes: bytes / 1024
convert_megabytes_to_terabytes = lambda bytes: bytes / 1024 / 1024

# gigabytes converters
convert_gigabytes_to_bytes = lambda bytes: bytes * 1024 * 1024 * 1024
convert_gigabytes_to_kilobytes = lambda bytes: bytes * 1024 * 1024
convert_gigabytes_to_megabytes = lambda bytes: bytes * 1024
convert_gigabytes_to_terabytes = lambda bytes: bytes / 1024

# Aliases

convert_b_to_kb = convert_bytes_to_kilobytes
convert_b_to_mb = convert_bytes_to_megabytes
convert_b_to_gb = convert_bytes_to_gigabytes
convert_b_to_tb = convert_bytes_to_terabytes

convert_kb_to_b = convert_kilobytes_to_bytes
convert_kb_to_mb = convert_kilobytes_to_megabytes
convert_kb_to_gb = convert_kilobytes_to_gigabytes
convert_kb_to_tb = convert_kilobytes_to_terabytes

convert_mb_to_b = convert_megabytes_to_bytes
convert_mb_to_kb = convert_megabytes_to_kilobytes
convert_mb_to_gb = convert_megabytes_to_gigabytes
convert_mb_to_tb = convert_megabytes_to_terabytes

convert_gb_to_b = convert_gigabytes_to_bytes
convert_gb_to_kb = convert_gigabytes_to_kilobytes
convert_gb_to_mb = convert_gigabytes_to_megabytes
convert_gb_to_tb = convert_gigabytes_to_terabytes


# general converter
def convert_bytes(bytes: float, 
                  from_unit: Union[MemoryUnit, str] = "KB", 
                  to_unit: Union[MemoryUnit, str] = "KB"
                  ) -> float:
    from_unit = MemoryUnit(from_unit)
    to_unit = MemoryUnit(to_unit)
    
    if from_unit == to_unit:
        return bytes
    
    function_name = f"convert_{from_unit.value.lower()}_to_{to_unit.value.lower()}"
    converted_bytes = globals()[function_name](bytes)
    return converted_bytes
