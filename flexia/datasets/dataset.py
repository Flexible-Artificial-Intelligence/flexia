from abc import ABC, abstractmethod


class Dataset:
    def __init__(self):
        self.__n = 0

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self.__n > len(self):
            raise StopIteration

        sample = self[self.__n]
        self.__n += 1

        return sample