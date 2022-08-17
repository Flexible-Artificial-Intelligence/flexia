from ..enums import EqualEnum, ExplicitEnum


class MemoryUnit(EqualEnum, ExplicitEnum):
    B = "B" # bytes
    KB = "KB" # kilobytes
    MB = "MB" # megabytes
    GB = "GB" # gigabytes
    TB = "TB" # terabytes