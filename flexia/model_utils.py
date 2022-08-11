from typing import Any
from torch import nn
import warnings

from .import_utils import is_apex_available


if is_apex_available():
    from apex.normalization import FusedLayerNorm
else:
    from torch.nn import LayerNorm as FusedLayerNorm


def freeze_module(module:nn.Module, verbose:bool=False) -> None:
    """
    Freezes module's parameters.
    """
    
    for name, parameter in module.named_parameters():
        parameter.requires_grad = False

        if verbose:
            print(f"Module `{name}` was freezed.")
        
        
def get_freezed_module_parameters(module:nn.Module) -> list:
    """
    Returns names of freezed parameters of the given module.
    """
    
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
            
    return freezed_parameters


def replace_layernorm_with_fused_layernorm(module:nn.Module):
    """
    Replace the normal (PyTorch-vanilla) LayerNorm to Apex Fused LayerNorm.
    
    Args:
        module: The target module to be replaced.

    References:
        https://github.com/affjljoo3581/Feedback-Prize-Competition/blob/034427117cc8a3e1dd63401b3519fc28e3f18830/src/utils/model_utils.py#L33
    
    Gists:
        https://github.com/NVIDIA/apex/issues/449

    """

    if not is_apex_available():
        warnings.warn("The `apex` is not installed, hence the LayerNorm will be replaced with itself.")

    for submodule in module.modules():
        for name, layer in submodule.named_children():
            if not isinstance(layer, nn.LayerNorm):
                continue

            # Create new fused layer-norm and copy the original parameters.
            new_layer = FusedLayerNorm(layer.normalized_shape, layer.eps)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias

            # Replace the layer-norm to the new one.
            setattr(submodule, name, new_layer)


def move_module_to_eval_mode(module:nn.Module, half_mode:bool=False) -> None:
    module.eval()

    if half_mode:
        module.half()


def get_grouped_module_parameters(module:nn.Module, weight_decay:float=0.01) -> Any:
    model_parameters = list(module.named_parameters())
    no_decay_layers = ("bias", "LayerNorm.bias", "LayerNorm.weight")
    
    decay_parameters = [p for n, p in model_parameters if not any(nd in n for nd in no_decay_layers)]
    no_decay_parameters = [p for n, p in model_parameters if any(nd in n for nd in no_decay_layers)]

    grouped_module_parameters = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]

    return grouped_module_parameters