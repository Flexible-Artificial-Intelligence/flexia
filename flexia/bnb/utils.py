from torch import nn
from ..import_utils import is_bitsandbytes_available


if is_bitsandbytes_available():
    import bitsandbytes as bnb


def set_layer_optim_bits(model:nn.Module, optim_bits:int=32, layer:nn.Module=nn.Embedding) -> None:
    """
    Overrides keeping bits for given layer.

    Inputs:
        model:nn.Module - model with certain layer to override keeping bits.
        optim_bits:int - optimizer's bits for layer. Default: 32.
        layer:nn.Module - layer to change optimizer's bits. Default: nn.Embedding

    
    """
    
    for module in model.modules():
        if isinstance(module, layer):
            module_instance = bnb.optim.GlobalOptimManager.get_instance()
            module_instance.register_module_override(module, "weight", {"optim_bits": optim_bits})