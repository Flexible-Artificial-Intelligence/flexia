import torch
import numpy as np
from typing import Any
import logging


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