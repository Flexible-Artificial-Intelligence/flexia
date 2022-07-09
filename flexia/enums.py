from enum import Enum


class ExplicitEnum(Enum):
    @classmethod
    def _missing_(cls, value):
        keys = list(cls._value2member_map_.keys())
        raise ValueError(f"`{value}` is not a valid `{cls.__name__}`, select one of `{keys}`.")


class SchedulerLibraries(ExplicitEnum):
    TRANSFORMERS = "transformers"
    TORCH = "torch"


class OptimizerLibraries(ExplicitEnum):
    TRANSFORMERS = "transformers"
    TORCH = "torch"
    BITSANDBYTES = "bitsandbytes"