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


from torch import nn
from typing import List

from .import_utils import is_bitsandbytes_available


if is_bitsandbytes_available():
    import bitsandbytes as bnb


def set_layers_precisions(module: nn.Module, 
                          layers: List[nn.Module] = [nn.Embedding], 
                          precisions: List[int] = [32],
                          ) -> None:
                          
    assert len(layers) == len(precisions)

    for layer, precision in zip(layers, precisions):
        set_layer_precision(module=module, layer=layer, precision=precision)



def set_layer_precision(module: nn.Module, 
                        layer: nn.Module = nn.Embedding, 
                        precision: int = 32,
                        ) -> None:
    """
    Overrides keeping bits for given layer.

    Inputs:
        model:nn.Module - model with certain layer to override keeping bits.
        optim_bits:int - optimizer's bits for layer. Default: 32.
        layer:nn.Module - layer to change optimizer's bits. Default: nn.Embedding
        
    """
    
    for submodule in module.modules():
        if isinstance(submodule, layer):
            submodule_instance = bnb.optim.GlobalOptimManager.get_instance()
            submodule_instance.register_module_override(submodule, "weight", {"optim_bits": precision})