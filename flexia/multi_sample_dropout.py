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
from torch import nn
from typing import Any, Optional, Union, List, Callable, Tuple
import logging


logger = logging.getLogger(__name__)

class MultiSampleDropout(nn.Module):
    def __init__(self, 
                 layer:nn.Module,
                 criterion:Callable, 
                 p:Union[float, List[float]]=0.1):
        
        """
        Implementation of Multi-Sample Dropout: https://arxiv.org/abs/1905.09788
        """
        
        super(MultiSampleDropout, self).__init__()
        
        if isinstance(p, float):
            self.dropouts = nn.ModuleList(modules=[nn.Dropout(p=p)])
        elif isinstance(p, list):
            self.dropouts = nn.ModuleList(modules=[nn.Dropout(p=_) for _ in p])
        else:
            raise "Given type of `p` is not supported"
            
        self.layer = layer
        self.criterion = criterion
        self.p = p
        self.n = len(self.dropouts)
        

    def forward(self, inputs:Any, targets:Optional[Any]=None) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        losses, outputs = [], []
        for dropout in self.dropouts:
            output = self.layer(dropout(inputs))
            outputs.append(output)
            
            if targets is not None:
                dropout_loss = self.criterion(output, targets)
                losses.append(dropout_loss)

        outputs = torch.stack(outputs, dim=0).mean(dim=0)
        losses = torch.stack(losses, dim=0).mean(dim=0)
              
        return (outputs, losses) if targets is not None else outputs