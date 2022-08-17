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
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import functional as F
import numpy as np
from typing import Union, Optional, List, Union, Tuple, Dict
import random


from .import_utils import is_torch_xla_available
from .python_utils import get_random_number
from .enums import DeviceType


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


precision_dtypes = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

mixed_precision_dtypes = ("fp16", "bf16")
default_checkpoint_custom_keys = {
    "model": "model_state",
    "optimizer": "optimizer_state",
    "scheduler": "scheduler_state",
    "scaler": "scaler_state",
}


def seed_everything(seed:Optional[int]=None, deterministic:int=True, benchmark:int=True) -> int:
    """
    Sets seed for `torch`, `numpy` and `random` libraries to have opportunity to reproduce results.
    """
    if seed is None:
        seed = get_random_number()
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if is_torch_xla_available():
        xm.set_rng_seed(seed)
    
    return seed


def load_checkpoint(path:str, 
                    model:nn.Module=None, 
                    optimizer:Optional[Optimizer]=None, 
                    scheduler:Optional[_LRScheduler]=None, 
                    scaler=None,
                    strict:bool=True,  
                    custom_keys:Optional[Dict[str, str]]=default_checkpoint_custom_keys, 
                    eval_mode=False, 
                    load_states=True) -> dict:

        """
        Loads checkpoint and then load state for model, optimizer or scheduler, if they are set. 

        Inputs:
            path: str - checkpoint's path.
            model: nn.Module - PyTorch's module.
            optimizer: Optional[Optimizer] - PyTorch's or HuggingFace Transformers's optimizer. Default: None.
            scheduler: Optional[_LRScheduler] - PyTorch's or HuggingFace Transformers's scheduler. Default: None.
            strict: bool - whether to strictly enforce that the keys in state_dict match the keys returned by this module’s state_dict() function. Default: True
            ignore_warnings: bool - if True the further warnings will be ignored. Default: False.
            custom_keys: Optional["dict[str, str]"] - sets keys for the checkpoint.

        Outputs:
            checkpoint: dict - loaded checkpoint.
            
        """

        checkpoint = torch.load(path)
        
        if model is not None:
            model_key = custom_keys.get("model", "model_state")
            model_state = checkpoint.get(model_key)
            
            if model_state is not None and load_states:
                model.load_state_dict(model_state, strict=strict)

        if optimizer is not None:
            optimizer_key = custom_keys.get("optimizer", "optimizer_state")
            optimizer_state = checkpoint.get(optimizer_key)

            if optimizer_state is not None and load_states:
                optimizer.load_state_dict(optimizer_state, strict=strict)

        if scheduler is not None:
            scheduler_key = custom_keys.get("scheduler", "scheduler_state")
            scheduler_state = checkpoint.get(scheduler_key)
            
            if scheduler_state is not None and load_states:
                scheduler.load_state_dict(scheduler_state, strict=strict)

        if scaler is not None:
            scaler_key = custom_keys.get("scaler", "scaler_state")
            scaler_state = checkpoint.get(scaler_key)

            if scaler_state is not None and load_states:
                scaler.load_state_dict(scaler_state, strict=strict)

        if eval_mode:
            model.eval()

        return checkpoint


def save_checkpoint(path:str, 
                    model:nn.Module=None, 
                    optimizer:Optional[Optimizer]=None, 
                    scheduler:Optional[_LRScheduler]=None, 
                    scaler=None,
                    custom_keys:Optional[Dict[str, str]]=default_checkpoint_custom_keys,
                    device_type:Union[DeviceType, str]="cpu",
                    **kwargs
                    ) -> str:
        
    device_type = DeviceType(device_type)
    
    checkpoint = {}

    if model is not None:
        model_key = custom_keys.get("model", "model_state")
        checkpoint[model_key] = model.state_dict()

    if optimizer is not None:
        optimizer_key = custom_keys.get("optimizer", "optimizer_state")
        checkpoint[optimizer_key] = optimizer.state_dict()

    if scheduler is not None:
        scheduler_key = custom_keys.get("scheduler", "scheduler_state")
        checkpoint[scheduler_key] = scheduler.state_dict()

    if scaler is not None:
        scaler_key = custom_keys.get("scaler", "scaler_state")
        checkpoint[scaler_key] = scaler.state_dict()

    checkpoint.update(kwargs)

    save_function = torch.save if device_type != DeviceType.TPU else xm.save
    save_function(checkpoint, path)

    return path


def concat_tensors_with_padding(tensors:List[torch.Tensor], 
                                padding:Union[int, float]=0,
                                dim=1) -> torch.Tensor:
    """
    Concatenate the list of tensors to be a single tensor with paddings.
    
    Args:
        tensors: The list of tensors which have different lengths. They should have
            the shape of `(batch_size, seq_len, dim)` or `(batch_size, seq_len)`.
        padding: The padding value for the tensors. If the tensor is shorter than other
            tensors, than it will be padded with this value. Default is `0`.
    Returns:
        A concatenated single tnesor.

    References:
        https://github.com/affjljoo3581/Feedback-Prize-Competition/blob/034427117cc8a3e1dd63401b3519fc28e3f18830/src/utils/model_utils.py#L65
    """
    
    max_length = max(tensor.shape[dim] for tensor in tensors)

    padded_tensors = []
    for tensor in tensors:
        length_diff = max_length - tensor.shape[dim]

        # This function only supports two and three dimensional tensors.
        if tensor.ndim == 2:
            padding_size = (0, length_diff)
        elif tensor.ndim == 3:
            padding_size = (0, 0, 0, length_diff)

        padded_tensor = F.pad(input=tensor, 
                              pad=padding_size, 
                              value=padding, 
                              mode="constant")

        padded_tensors.append(padded_tensor)

    padded_tensors = torch.cat(padded_tensors, dim=0)

    return padded_tensors