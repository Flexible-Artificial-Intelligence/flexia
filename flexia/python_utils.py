import os
import shutil
import random
from typing import Any


def read_file(path: str, mode: str = "r") -> Any:
    with open(path, mode) as file:
        data = file.read()

    return data


def make_directory(directory: str, overwriting: bool = False) -> str:
    """
    Makes directory
    """
    
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        if overwriting:
            remove_files_from_directory(directory=directory)
    
    return directory


def remove_files_from_directory(directory: str, verbose: bool = False) -> None:
    """
    Removes all files and folders from directory.
    """
        
    filenames = os.listdir(directory)
    pathes = [os.path.join(directory, filename) for filename in filenames]
        
    for path in pathes:
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

        if verbose:
            print(f"Removed '{path}' from '{directory}'.")


def get_random_number(min_value: int = 0, max_value: int = 50) -> int:
    """
    Returns random value from [`min_value`, `max_value`] range.
    """
    
    return random.randint(min_value, max_value)