from abc import ABC, abstractmethod
from typing import Any


class Dataset:
    def __init__(self) -> None:
        self.__n = 0

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> Any:
        pass

    def __iter__(self) -> "Dataset":
        return self

    def __next__(self) -> Any:
        if self.__n > len(self):
            raise StopIteration

        sample = self[self.__n]
        self.__n += 1

        return sample