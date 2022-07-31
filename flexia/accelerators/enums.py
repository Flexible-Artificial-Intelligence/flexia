from ..enums import EqualEnum, ExplicitEnum


class MemoryUnit(EqualEnum, ExplicitEnum):
    KB = "KB" # kilobytes
    MB = "MB" # megabytes
    GB = "GB" # gigabytes
    TB = "TB" # terabytes