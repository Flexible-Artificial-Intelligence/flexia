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


def is_transformers_available() -> bool:
    """
    Checks the availablity of `transformers` library.
    """

    try:
        import transformers
        return True
    except ModuleNotFoundError:
        return False


def is_wandb_available() -> bool:
    """
    Checks the availablity of `wandb` library.
    """

    try:
        import wandb
        return True
    except ModuleNotFoundError:
        return False
        

def is_bitsandbytes_available() -> bool:
    """
    Checks the availablity of `bitsandbytes` library.    
    """
    
    try:
        import bitsandbytes
        return True
    except:
        return False


def is_pandas_available() -> bool:
    """
    Checks the availablity of `pandas` library.    
    """

    try:
        import pandas
        return True
    except ModuleNotFoundError:
        return False


def is_torch_xla_available() -> bool:
    """
    Checks the availablity of `torch_xla` library.    
    """

    try:
        import torch_xla
        return True
    except ModuleNotFoundError:
        return False


def is_pynvml_available() -> bool:
    """
    Checks the availablity of `pynvml` library.    
    """

    try:
        import pynvml
        return True
    except ModuleNotFoundError:
        return False