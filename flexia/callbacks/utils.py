# Copyright 2022 The Flexia Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .enums import Modes
import shutil
import os


def get_delta_value(value, delta=0.0, mode="min"):
    mode = Modes(mode)
    delta_value = (value - delta) if mode == Modes.MIN else (value + delta)
    
    return delta_value


def compare(value, other, mode="min"):
    mode = Modes(mode)
    condition = (value < other) if mode == Modes.MIN else (value > other)
    
    return condition


def remove_files_from_directory(self, directory:str) -> None:
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