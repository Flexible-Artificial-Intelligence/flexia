from tokenize import group
from torch import nn, optim
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any, Iterator, Union, List, Tuple, Dict, Optional, Iterator
import numpy as np

from .import_utils import is_transformers_available, is_bitsandbytes_available
from .enums import OptimizerLibrary, SchedulerLibrary


if is_transformers_available():
    import transformers

if is_bitsandbytes_available():
    import bitsandbytes as bnb
    from .bitsandbytes_utils import set_layers_precisions


no_decay_parameters = ("bias", "LayerNorm.bias", "LayerNorm.weight")


def __get_from_library(library:Any, 
                       name:str, 
                       parameters: Dict[str, Any], 
                       **kwargs
                       ) -> Any:
    instance = getattr(library, name)
    instance = instance(**kwargs, **parameters)

    return instance


def get_scheduler(optimizer: Optimizer, 
                  name: str, 
                  parameters: Dict[str, Any] = {}, 
                  library: Union[str, SchedulerLibrary] = "torch", 
                  ) -> _LRScheduler:
    library = SchedulerLibrary(library)

    if library == SchedulerLibrary.TORCH:
        library = lr_scheduler

    elif library == SchedulerLibrary.TRANSFORMERS:
        if is_transformers_available():
            library = transformers
        else:
            raise ValueError(f"Library `{library}` is not found or not provided.")

    scheduler = __get_from_library(library=library, 
                                   name=name, 
                                   parameters=parameters, 
                                   optimizer=optimizer)

    return scheduler


def get_transformers_scheduler(optimizer: Optimizer, 
                               name: str,
                               num_training_steps: int,  
                               parameters: Dict[str, Any] = {}, 
                               warmup: Union[float, int] = 0.0, 
                               gradient_accumulation_steps: int = 1
                               ) -> _LRScheduler:

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
        library = transformers
        scheduler = __get_from_library(library=library, 
                                       name=name, 
                                       parameters=parameters, 
                                       optimizer=optimizer)

        return scheduler
    else:
        raise ValueError(f"Library `transformers` is not found.")


def get_optimizer(module_parameters: Any, 
                  name: str, 
                  parameters: Dict[str, Any] = {}, 
                  library: Union[str, OptimizerLibrary] = "torch"
                  ) -> Optimizer:
    library = OptimizerLibrary(library)

    if library == OptimizerLibrary.TORCH:
        library = optim

    elif library == OptimizerLibrary.TRANSFORMERS:
        if is_transformers_available():
            library = transformers
        else:
            raise ValueError(f"Library `{library}` is not found or not provided.")

    elif library == OptimizerLibrary.BITSANDBYTES:
        if is_bitsandbytes_available():
            library = bnb.optim
        else:
            raise ValueError(f"Library `{library}` is not found or not provided.")

    optimizer = __get_from_library(library=library, 
                                   name=name, 
                                   parameters=parameters, 
                                   params=module_parameters)

    return optimizer


def get_bitsandbytes_optimizer(module: nn.Module, 
                               name: str,
                               module_parameters: Optional[Any] = None,  
                               parameters: Dict[str, Any] = {}, 
                               precisions: List[int] = [32], 
                               layers: List[nn.Module] = [nn.Embedding]
                               ) -> Optimizer:

    if module_parameters is None:
        module_parameters = module.parameters()

    optimizer = get_optimizer(module_parameters=module_parameters, 
                              name=name, 
                              parameters=parameters, 
                              library="bitsandbytes")

    set_layers_precisions(module=module, layers=layers, precisions=precisions)

    return optimizer


def get_optimizer_and_scheduler(module_parameters: Any,
                                optimizer_name: str, 
                                optimizer_parameters: Dict[str, Any] = {}, 
                                optimizer_library: Union[str, OptimizerLibrary] = "torch", 
                                scheduler_name: Optional[str] = None, 
                                scheduler_parameters: Dict[str, Any] = {}, 
                                scheduler_library: Union[str, SchedulerLibrary] = "torch",
                                optimizer_kwargs: Dict[str, Any] = {},
                                scheduler_kwargs: Dict[str, Any] = {},
                                ) -> Tuple[Optimizer, Optional[_LRScheduler]]:

    
    optimizer = get_optimizer(module_parameters=module_parameters, 
                              name=optimizer_name, 
                              parameters=optimizer_parameters, 
                              library=optimizer_library, 
                              **optimizer_kwargs)

    scheduler = None
    if scheduler_name is not None:
        scheduler = get_scheduler(optimizer=optimizer, 
                                  name=scheduler_name, 
                                  parameters=scheduler_parameters, 
                                  library=scheduler_library, 
                                  **scheduler_kwargs)

    return optimizer, scheduler



def get_lr(optimizer: Optimizer, 
           groups: Optional[Union[int, List[int]]] = None, 
           key: str = "lr",
           ) -> Union[List[float], float]:

    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"The given `optimizer` type is not supported, it must be instance of Optimizer.")
        
    if isinstance(groups, int):
        groups = [groups]
    
    param_groups = optimizer.param_groups
    lrs = {param_group_index: param_group[key] for param_group_index, param_group in enumerate(param_groups)}
    
    if groups is not None:
        lrs = {group_index: lrs[group_index] for group_index in groups}
    
    return lrs

def get_stepped_lrs(optimizer: Optimizer, 
                    scheduler: Optional[_LRScheduler] = None, 
                    steps: int = 10, 
                    steps_start: int = 1,
                    return_steps: bool = False,
                    groups: Optional[Union[int, List[int]]] = None, 
                    gradient_accumulation_steps: int = 1,
                    key: str = "lr"
                    ) -> Union[List[float], Dict[int, List[float]], Tuple[List[int]], Union[List[List[float]], Dict[int, List[float]]]]:
    
    steps = range(0+steps_start, steps+steps_start)
    
    groups_lrs = {}
    for step in steps:
        groups_lr = get_lr(optimizer=optimizer, groups=groups, key=key)
        
        for group_index, group_lr in groups_lr.items():
            if step == steps_start:
                groups_lrs[group_index] = []
            
            groups_lrs[group_index].append(group_lr)
            
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
    
    if return_steps:
        return groups_lrs, steps
    
    return groups_lrs


"""
References:
    https://github.com/affjljoo3581/CommonLit-Readability-Prize/blob/master/src/optimization/param_groups.py
"""


def get_decay_module_parameters(module: nn.Module, 
                                no_decay_parameters: Iterator[str] = no_decay_parameters,
                                recurse: bool = True,
                                ) -> Iterator[nn.Parameter]:
                                
    for name, parameter in list(module.named_parameters(recurse=recurse)):
        if name not in no_decay_parameters:
            yield parameter

            
def get_no_decay_module_parameters(module: nn.Module, 
                                   no_decay_parameters: Iterator[str] = no_decay_parameters,
                                   recurse: bool = True,
                                   ) -> Iterator[nn.Parameter]:

    for name, parameter in list(module.named_parameters(recurse=recurse)):
        if name in no_decay_parameters:
            yield parameter


def layerwise_learning_rate_decay(module: Union[Iterator[nn.Module], nn.Module], 
                                  lr: float = 1e-3, 
                                  layerwise_lr_decay: float = 0.1, 
                                  weight_decay: float = 0.01, 
                                  **kwargs,
                                  ) -> Iterator[Dict[str, Any]]:
    
    if isinstance(module, nn.Module):
        modules = reversed(list(module.modules()))
    
    for index, submodule in enumerate(modules):
        submodule_layerwise_lr_decay = layerwise_lr_decay ** index
        submodule_lr = lr * submodule_layerwise_lr_decay
        
        submodule_decay_parameters = get_decay_module_parameters(module=submodule, **kwargs)
        num_submodule_decay_parameters = len(list(submodule_decay_parameters))
        
        if num_submodule_decay_parameters > 0:
            yield {
                "params": submodule_decay_parameters,
                "lr": submodule_lr,
                "weight_decay": weight_decay,
            }
            
        submodule_no_decay_parameters = get_no_decay_module_parameters(module=submodule, **kwargs)
        num_submodule_no_decay_parameters = len(list(submodule_no_decay_parameters))
        
        if num_submodule_no_decay_parameters > 0:
            yield {
                "params": submodule_no_decay_parameters,
                "lr": submodule_lr,
                "weight_decay": 0.0,
            }


def get_parameter_groups(module: nn.Module, 
                         weight_decay: float = 0.01,
                         **kwargs
                         ) -> List[Dict[str, Any]]:
    decay_module_parameters = get_decay_module_parameters(module=module, **kwargs)
    no_decay_parameters = get_no_decay_module_parameters(module=module, **kwargs)

    groups = [
        {"params": decay_module_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ] 

    return group 

# Aliases
get_decay_module_params = get_decay_module_parameters
get_no_decay_module_params = get_decay_module_parameters
llrd = layerwise_learning_rate_decay
get_learning_rate = get_lr
get_stepped_learning_rates = get_stepped_lrs