from torch import nn, optim
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any, Union, List, Tuple, Dict, Optional

from .import_utils import is_transformers_available, is_bitsandbytes_available
from .enums import OptimizerLibrary, SchedulerLibrary


if is_transformers_available():
    import transformers

if is_bitsandbytes_available():
    import bitsandbytes as bnb
    from .bitsandbytes_utils import set_layers_precisions


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


def get_lr(optimizer: Optimizer, 
           only_last_group: bool = False, 
           key: str = "lr"
           ) -> Union[List[float], float]:
    """
    Returns optimizer's learning rates for each or last group.
    """

    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"The given `optimizer` type is not supported, it must be instance of Optimizer.")
    
    param_groups = optimizer.param_groups
    lrs = [param_group[key] for param_group in param_groups]        
    return lrs[-1] if only_last_group else lrs


def get_stepped_lrs(optimizer: Optimizer, 
                    scheduler: Optional[_LRScheduler] = None, 
                    steps: int = 10, 
                    steps_start: int = 1,
                    return_as_dict: bool = False,
                    return_steps_list: bool = False,
                    only_last_group: bool = False, 
                    key: str = "lr"
                    ) -> Union[List[float], Dict[int, List[float]], Tuple[List[int]], Union[List[List[float]], Dict[int, List[float]]]]:
    
    steps = range(0+steps_start, steps+steps_start)
    
    param_groups = optimizer.param_groups
    num_param_groups = len(param_groups)
    
    groups_lrs = [[]]*num_param_groups
    for step in steps:
        groups_lr = get_lr(optimizer, only_last_group=False, key=key)
        
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