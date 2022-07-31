from .logger import Logger

from ..import_utils import is_pandas_available


if is_pandas_available():
    import pandas as pd


class DataFrameLogger(Logger):
    def __init__(self):
        pass

    def on_training_start(self, trainer) -> None:
        pass

    def on_training_step_end(self, trainer) -> None:
        pass

    def on_validation_end(self, trainer) -> None:
        pass

    def on_training_end(self, trainer) -> None:
        pass