# from .enums import MemoryUnit


# def convert_to_kilobytes(bytes, unit="KB"):
#     unit = MemoryUnit(unit)
#     multiplier = 1
        
#     if unit == MemoryUnit.MB:
#         bytes = 1024
#     elif unit == MemoryUnit.GB:
#         bytes = 1024 * 1024
    
#     bytes *= multiplier

#     return multiplier


# def convert_memory_units(bytes, from_unit="KB", to_unit="MB"):
#     from_unit = MemoryUnit(from_unit)
#     to_unit = MemoryUnit(to_unit)

#     kilobytes = convert_to_kilobytes(bytes=bytes, unit=from_unit)
