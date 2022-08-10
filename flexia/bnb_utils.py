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
from .import_utils import is_bitsandbytes_available


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