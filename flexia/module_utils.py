import torch
from torch import nn
from typing import Any, Dict, List


def freeze_module(module:nn.Module, verbose:bool=False) -> None:
    for name, parameter in module.named_parameters():
        parameter.requires_grad = False

        if verbose:
            print(f"Parameter `{name}` was freezed.")
        
        
def get_module_freezed_parameters(module:nn.Module) -> List[str]:
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
            
    return freezed_parameters


def move_module_to_eval_mode(module:nn.Module, half_mode:bool=False) -> None:
    module.eval()

    if half_mode:
        module.half()


def get_grouped_module_parameters(module:nn.Module, weight_decay:float=0.01) -> List[Dict[str, Any]]:
    model_parameters = list(module.named_parameters())
    no_decay_layers = ("bias", "LayerNorm.bias", "LayerNorm.weight")
    
    decay_parameters = [p for n, p in model_parameters if not any(nd in n for nd in no_decay_layers)]
    no_decay_parameters = [p for n, p in model_parameters if any(nd in n for nd in no_decay_layers)]

    grouped_module_parameters = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]

    return grouped_module_parameters


def save_module(module:nn.Module, path:str="module.pt") -> None:
    module_state = module.state_dict()
    saved_state = torch.save(module_state, path)