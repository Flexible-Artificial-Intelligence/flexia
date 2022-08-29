import torch
from torch import nn
from typing import Any, Dict, List, Callable


def freeze_module(module: nn.Module, verbose: bool = False) -> None:
    for name, parameter in module.named_parameters():
        parameter.requires_grad = False

        if verbose:
            print(f"Parameter `{name}` was freezed.")
        
        
def get_module_freezed_parameters(module: nn.Module) -> List[str]:
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
            
    return freezed_parameters


def move_module_to_eval_mode(module: nn.Module, half_mode: bool = False) -> None:
    module.eval()

    if half_mode:
        module.half()


def save_module(module: nn.Module, path: str = "module.pt") -> None:
    module_state = module.state_dict()
    saved_state = torch.save(module_state, path)


def replace_layer_with_other_layer(module: nn.Module, 
                                   layer_instance: nn.Module,
                                   other_layer_instance: nn.Module, 
                                   get_parameters: Callable[[nn.Module], Dict[str, Any]] = lambda module: {}, 
                                   load_state_dict: bool = False):
    """
    Recursively iterrates over all modules and submodules in the given module and replaces `layer_instance` module with `other_layer_instance` module.
    The `other_layer_instance` parameters are given from `get_parameters` function.
    Additionally, `replace_layer_with_other_layer` can load `state_dict` from `layer_instance` to newly initialized `other_layer_instance`.
    """
    
    for submodule in module.modules():
        for name, layer in submodule.named_children():
            layer_children = layer.children()
            num_layer_children = len(list(layer_children))
            
            # Replaces the layer's childrens and replaces `layer_instance` module with `other_layer_instance` module.
            if num_layer_children > 0:
                replace_layer_with_other_layer(module=layer, 
                                               layer_instance=layer_instance,
                                               other_layer_instance=other_layer_instance, 
                                               get_parameters=get_parameters, 
                                               load_state_dict=load_state_dict)
                
            # Replaces `layer_instance` module with `other_layer_instance` module.
            if isinstance(layer, layer_instance):
                new_layer_parameters = get_parameters(layer)
                new_layer = other_layer_instance(**new_layer_parameters)
                
                # Loads state dictionary from `layer` (i.e `layer_instance`) for newly initialized `other_layer_instance`.
                if load_state_dict:
                    layer_state_dict = layer.state_dict()
                    new_layer.load_state_dict(layer_state_dict)
                    
                # Replaces `name` in `submodule` with `new_layer` (i.e newly initialized `other_layer_instnce`).
                setattr(submodule, name, new_layer)


def average_parameters(modules: List[nn.Module], weights: List[float]):
    num_average_modules = len(modules)
    
    averaged_state_dict = None
    for index, (module, weight) in enumerate(zip(modules, weights)):
        module_state_dict = module.state_dict()

        if averaged_state_dict is None and index == 0:
            averaged_state_dict = module_state_dict
        else:
            for key, value in module_state_dict.items():
                averaged_state_dict[key] += (value * weight) / num_average_modules

    return averaged_state_dict
    