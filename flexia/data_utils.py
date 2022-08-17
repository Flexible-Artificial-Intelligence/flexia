from torch.utils.data import Dataset, DataLoader
from typing import Any

from .python_utils import get_random_number


def get_random_sample(dataset:Dataset) -> Any:
    num_samples = len(dataset)
    index = get_random_number(min_value=0, max_value=num_samples - 1)
    sample = dataset[index]
    return sample


def get_batch(loader:DataLoader) -> Any:
    batch = next(iter(loader))
    return batch