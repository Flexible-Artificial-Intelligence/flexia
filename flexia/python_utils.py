import os
import shutil


def make_directory(directory, overwriting=False):
    """
    Makes directory
    """
    
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        if overwriting:
            remove_files_from_directory(directory=directory)
    
    return directory


def remove_files_from_directory(directory:str, verbose=False) -> None:
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