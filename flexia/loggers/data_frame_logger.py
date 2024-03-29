from .logger import Logger
from ..import_utils import is_pandas_available


if is_pandas_available():
    import pandas as pd


class DataFrameLogger(Logger):
    pass