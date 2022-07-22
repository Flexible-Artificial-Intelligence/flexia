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


import torch
import numpy as np
from typing import Any
import logging
from .import_utils import is_torch_xla_available


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


logger = logging.getLogger(__name__)


def unsqueeze(inputs, dim=0):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.unsqueeze(dim=dim)
    elif isinstance(inputs, np.ndarray):
        inputs = np.expand_dims(inputs, axis=dim)
    else:
        raise TypeError(f"Unsupported type `{type(inputs)}` of inputs.")
    
    return inputs


def to_list(input:Any) -> list:
    if not isinstance(input, list):
        if isinstance(input, torch.Tensor):
            input = input.detach().to("cpu").tolist()
        elif isinstance(input, np.ndarray):
            input = input.tolist()
        else:
            try:
                input = list(input)
            except TypeError:
                input = list([input])
            
    return input


def to_lists(*inputs:Any) -> tuple:
    """
    Converts all inputs to torch.Tensor.
    """
    return tuple(map(lambda input: to_list(input), inputs))


def to_tensor(input:Any) -> torch.Tensor:
    """
    Converts input to torch.Tensor.
    """

    if not isinstance(input, torch.Tensor):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        else:
            input = torch.tensor(input)
        
    return input


def to_tensors(*inputs:Any) -> tuple:
    """
    Converts all inputs to torch.Tensor.
    """
    return tuple(map(lambda input: to_tensor(input), inputs))


def is_cuda_available():
    return torch.cuda.is_available()


def is_tpu_available():
    if is_torch_xla_available():
        devices = xm.get_xla_supported_devices()
        return len(devices) > 0
    else:
        return False
