from .enums import Modes


def get_delta_value(value, delta=0.0, mode="min"):
    mode = Modes(mode)
    delta_value = (value - delta) if mode == Modes.MIN else (value + delta)
    
    return delta_value


def compare(value, other, mode="min"):
    mode = Modes(mode)
    condition = (value < other) if mode == Modes.MIN else (value > other)
    
    return condition