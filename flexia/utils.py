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
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Any, Union, Optional
import random
import os
import logging

from flexia.torch_utils import is_cuda_available

from .import_utils import is_transformers_available, is_bitsandbytes_available, is_torch_xla_available
from .exceptions import LibraryException
from .enums import SchedulerLibrary, OptimizerLibrary
from .torch_utils import is_cuda_available, is_tpu_available


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


def initialize_device(device=None):
    if device is None:
        if is_cuda_available():
            device = torch.device("cuda:0")
        elif is_tpu_available():
            device = xm.xla_device(n=1)
        else:
            device = torch.device("cpu")

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
    os.environ['PC_SEED'] = str(seed)
    
    return seed
    

def get_lr(optimizer:Optimizer, only_last:bool=False, key="lr") -> Union[int, list]:
    """
    Returns optimizer's leearning rates for each group.
    """

    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"The given `optimizer` type is not supported, it must be instance of Optimizer.")
    
    lrs = []
    for param_group in optimizer.param_groups:
        if key not in param_group:
            key = "lr"
        
        lr = param_group[key]
        lrs.append(lr)
        
    return lrs[-1] if only_last else lrs


def load_checkpoint(path:str, 
                    model:nn.Module, 
                    optimizer:Optional[Optimizer]=None, 
                    scheduler:Optional[_LRScheduler]=None, 
                    strict:bool=True, 
                    ignore_warnings:bool=False, 
                    custom_keys:Optional["dict[str, str]"]=dict(model="model_state", 
                                                                optimizer="optimizer_state",
                                                                scheduler="scheduler_state")) -> dict:

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

        checkpoint = torch.load(path) if torch.cuda.is_available() else torch.load(path, map_location=torch.device("cpu"))
        
        model_key = custom_keys.get("model", "model_state")
        model_state = checkpoint[model_key]
        model.load_state_dict(model_state, strict=strict)

        if optimizer is not None:
            optimizer_key = custom_keys.get("optimizer", "optimizer_state")
            optimizer_state = checkpoint.get(optimizer_key)

            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state, strict=strict)

        if scheduler is not None:
            scheduler_key = custom_keys.get("scheduler", "scheduler_state")
            scheduler_state = checkpoint.get(scheduler_key)
            
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state, strict=strict)
    

        return checkpoint


def save_checkpoint(path:str, 
                    model:nn.Module, 
                    optimizer:Optional[Optimizer]=None, 
                    scheduler:Optional[_LRScheduler]=None, 
                    custom_keys:Optional["dict[str, str]"]=dict(model="model_state", 
                                                                optimizer="optimizer_state",
                                                                scheduler="scheduler_state"),
                    **kwargs) -> dict:
        
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

    torch.save(checkpoint, path)

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


def get_scheduler(optimizer:Optimizer, name:str="LinearLR", parameters:dict={}, library="torch") -> _LRScheduler:
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
        scheduler = __get_from_library(library=lr_scheduler, 
                                       name=name, 
                                       parameters=parameters, 
                                       optimizer=optimizer)

    elif library == SchedulerLibrary.TRANSFORMERS:
        if is_transformers_available():
            scheduler = __get_from_library(library=transformers, 
                                           name=name, 
                                           parameters=parameters, 
                                           optimizer=optimizer)
        else:
            raise LibraryException("transformers")

    return scheduler


def get_optimizer(model_parameters:Any, name:str="AdamW", parameters:dict={}, library:str="torch") -> Optimizer:
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
        optimizer = __get_from_library(library=optim, 
                                       name=name, 
                                       parameters=parameters, 
                                       params=model_parameters)

    elif library == OptimizerLibrary.TRANSFORMERS:
        if is_transformers_available():
            optimizer = __get_from_library(library=transformers, 
                                           name=name, 
                                           parameters=parameters, 
                                           params=model_parameters)
        else:
            raise LibraryException("transformers")

    elif library == OptimizerLibrary.BITSANDBYTES:
        if is_bitsandbytes_available():
            optimizer = __get_from_library(library=bnb.optim, 
                                           name=name, 
                                           parameters=parameters, 
                                           params=model_parameters)
        else:
            raise LibraryException("bitsandbytes")

    return optimizer



def freeze_module(module:nn.Module) -> None:
    """
    Freezes module's parameters.
    """
    
    for parameter in module.parameters():
        parameter.requires_grad = False
        
        
def get_freezed_module_parameters(module:nn.Module) -> list:
    """
    Returns names of freezed parameters of the given module.
    """
    
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
            
    return freezed_parameters