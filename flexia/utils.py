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
from torch import nn, optim
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Any, Union, Optional, List, Union, Tuple, Dict
import random
import os


from .import_utils import is_transformers_available, is_bitsandbytes_available, is_torch_xla_available, is_torch_backend_mps_available
from .bnb_utils import set_layer_optim_bits
from .enums import SchedulerLibrary, OptimizerLibrary, DeviceType


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_transformers_available():
    import transformers

if is_bitsandbytes_available():
    import bitsandbytes as bnb


precision_dtypes = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

mixed_precision_dtypes = ("fp16", "bf16")


def initialize_device(device=None):
    if device is None:
        if is_cuda_available():
            device = torch.device("cuda:0")
        elif is_tpu_available():
            device = xm.xla_device(n=0)
        elif is_mps_available():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu:0")

    device = torch.device(device)

    return device
    

def get_random_number(min_value:int=0, max_value:int=50) -> int:
    """
    Returns random value from [`min_value`, `max_value`] range.
    """
    
    return random.randint(min_value, max_value)

def seed_everything(seed:Optional[int]=None) -> int:
    """
    Sets seed for `torch`, `numpy` and `random` libraries to have opportunity to reproduce results.
    """
    if seed is None:
        seed = get_random_number()
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if is_torch_xla_available():
        xm.set_rng_seed(seed)
    
    return seed
    

def get_lr(optimizer:Optimizer, only_last_group:bool=False, key:str="lr") -> Union[List[float], float]:
    """
    Returns optimizer's learning rates for each or last group.
    """

    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"The given `optimizer` type is not supported, it must be instance of Optimizer.")
    
    param_groups = optimizer.param_groups
    lrs = [param_group[key] for param_group in param_groups]        
    return lrs[-1] if only_last_group else lrs


def get_stepped_lrs(optimizer:Optimizer, 
                    scheduler:Optional[_LRScheduler]=None, 
                    steps:int=10, 
                    steps_start:int=1,
                    return_as_dict:bool=False,
                    return_steps_list:bool=False,
                    only_last_group:bool=False, 
                    key:str="lr"
                    ) -> Union[List[float], Dict[int, List[float]], Tuple[List[int]], Union[List[List[float]], Dict[int, List[float]]]]:
    
    steps = range(0+steps_start, steps+steps_start)
    
    param_groups = optimizer.param_groups
    num_param_groups = len(param_groups)
    
    groups_lrs = [[]]*num_param_groups
    for step in steps:
        groups_lr = get_lr(optimizer)
        
        for group_index, group_lr in enumerate(groups_lr):
            groups_lrs[group_index].append(group_lr)
            
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
    
    if return_as_dict:
        groups_lrs = {group_index: group_lrs for group_index, group_lrs in enumerate(groups_lrs)}
        
    if only_last_group:
        groups_lrs = groups_lrs[-1]
        
    if return_steps_list:
        return steps, groups_lrs
    
    return groups_lrs


def load_checkpoint(path:str, 
                    model:nn.Module, 
                    optimizer:Optional[Optimizer]=None, 
                    scheduler:Optional[_LRScheduler]=None, 
                    strict:bool=True,  
                    custom_keys:Optional["dict[str, str]"]=dict(model="model_state", 
                                                                optimizer="optimizer_state",
                                                                scheduler="scheduler_state"), 
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

        checkpoint = torch.load(path, map_location=initialize_device())
        
        model_key = custom_keys.get("model", "model_state")
        model_state = checkpoint[model_key]
        
        if load_states:
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
    
        if eval_mode:
            model.eval()

        return checkpoint


def save_checkpoint(path:str, 
                    model:nn.Module, 
                    optimizer:Optional[Optimizer]=None, 
                    scheduler:Optional[_LRScheduler]=None, 
                    custom_keys:Optional["dict[str, str]"]=dict(model="model_state", 
                                                                optimizer="optimizer_state",
                                                                scheduler="scheduler_state"),
                    device_type="cpu",
                    **kwargs) -> dict:
        
    device_type = DeviceType(device_type)
    
    checkpoint = {}
    model_key = custom_keys.get("model", "model_state")
    checkpoint[model_key] = model.state_dict()

    if optimizer is not None:
        optimizer_key = custom_keys.get("optimizer", "optimizer_state")
        checkpoint[optimizer_key] = optimizer.state_dict()

    if scheduler is not None:
        scheduler_key = custom_keys.get("scheduler", "scheduler_state")
        checkpoint[scheduler_key] = scheduler.state_dict()

    checkpoint.update(kwargs)

    save_function = torch.save if device_type != DeviceType.TPU else xm.save
    save_function(checkpoint, path)

    return path


def get_random_sample(dataset:Dataset) -> Any:
    """
    Returns random sample from dataset.
    """

    index = random.randint(0, len(dataset)-1)
    sample = dataset[index]
    return sample


def get_batch(loader:DataLoader) -> Any:
    """
    Returns batch from loader.
    """

    batch = next(iter(loader))
    return batch


def __get_from_library(library, name, parameters, **kwargs):
    instance = getattr(library, name)
    instance = instance(**kwargs, **parameters)

    return instance


def get_scheduler(optimizer:Optimizer, name:str, parameters:dict={}, library="torch", *args, **kwargs) -> _LRScheduler:
    """
    Returns instance of scheduler.

    Inputs:
        name:str - name of scheduler, e.g ReduceLROnPlateau, CosineAnnealingWarmRestarts, get_cosine_schedule_with_warmup.
        parameters:dict - parameters of scheduler, e.g num_training_steps, T_mult, last_epoch. Default: {}.
        optimizer:Any - instance of optimizer to schedule the learning rate.
        library:str - library from which the scheduler will be used. Possible values: ["torch", "transformers"]. Default: "torch".
    
    Returns:
        scheduler:_LRScheduler - instance of scheduler.

    """


    library = SchedulerLibrary(library)

    if library == SchedulerLibrary.TORCH:
        module = lr_scheduler

    elif library == SchedulerLibrary.TRANSFORMERS:
        if is_transformers_available():
            module = transformers
        else:
            raise ValueError(f"Library `{library}` is not found or not provided.")

    scheduler = __get_from_library(library=module, 
                                   name=name, 
                                   parameters=parameters, 
                                   optimizer=optimizer)

    return scheduler


def get_transformers_scheduler(optimizer:Optimizer, 
                               name:str,
                               num_training_steps:int,  
                               parameters:dict={}, 
                               warmup:Union[float, int]=0.0, 
                               gradient_accumulation_steps:int=1):

    if is_transformers_available():
        # number of training steps relatively on gradient accumulation steps
        num_training_steps = num_training_steps // gradient_accumulation_steps

        # ratio of warmup steps
        if 0 <= warmup <= 1:
            num_warmup_steps = int(num_training_steps * warmup)
        else:
            num_warmup_steps = warmup
        
        # updating parameters dictionary with new defined parameters
        parameters.update({
            "num_training_steps": num_training_steps, 
            "num_warmup_steps": num_warmup_steps
        })

        # getting scheduler from `transformers` library
        module = transformers
        scheduler = __get_from_library(library=module, name=name, parameters=parameters, optimizer=optimizer)

        return scheduler
    else:
        raise ValueError(f"Library `transformers` is not found.")


def get_optimizer(model_parameters:Any, name:str, parameters:dict={}, library:str="torch") -> Optimizer:
    """
    Returns instance of optimizer.

    Inputs:
        name:str - name of optimizer, e.g AdamW, SGD, RMSprop.
        parameters:dict - parameters of optimizer, e.g lr, weight_decay. Default: {}.
        model_parameters:Any - model's parameters to optimize.
        library:str - library from which the optimizer will be used. Possible values: ["torch", "transformers", "bitsandbytes"]. Default: "torch".
    
    Returns:
        optimizer:Optimizer - instance of optimizer.

    """


    library = OptimizerLibrary(library)

    if library == OptimizerLibrary.TORCH:
        module = optim

    elif library == OptimizerLibrary.TRANSFORMERS:
        if is_transformers_available():
            module = transformers
        else:
            raise ValueError(f"Library `{library}` is not found or not provided.")

    elif library == OptimizerLibrary.BITSANDBYTES:
        if is_bitsandbytes_available():
            module = bnb.optim
        else:
            raise ValueError(f"Library `{library}` is not found or not provided.")

    optimizer = __get_from_library(library=module, 
                                   name=name, 
                                   parameters=parameters, 
                                   params=model_parameters)

    return optimizer


def get_bitsandbytes_optimizer(model:nn.Module, 
                               name:str,
                               model_parameters=None,  
                               parameters:dict={}, 
                               layers_optim_bits=[32], 
                               layers=[nn.Embedding], 
                               verbose=False):

    if model_parameters is None:
        model_parameters = model.parameters()

    optimizer = get_optimizer(model_parameters=model_parameters, 
                              name=name, 
                              parameters=parameters, 
                              library="bitsandbytes")

    for layer, layer_optim_bits in zip(layers, layers_optim_bits):
        set_layer_optim_bits(model=model, optim_bits=layer_optim_bits, layer=layer)

        if verbose:
            print(f"Changed precision of {layer} to {layer_optim_bits}.")

    return optimizer


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


def is_cuda_available():
    return torch.cuda.is_available()

def is_cpu_available():
    return os.cpu_count() > 0

def is_tpu_available():
    if is_torch_xla_available():
        devices = xm.get_xla_supported_devices()
        return len(devices) > 0
    else:
        return False
        
def is_mps_available():
    if is_torch_backend_mps_available():
        return torch.backends.mps.is_available()
    
    return False