import torch
from torch import nn
from typing import Any, Optional, Union, List, Callable, Tuple


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