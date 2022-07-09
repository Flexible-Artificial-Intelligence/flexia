import torch
from torch import nn
from typing import Optional, Union, Any

from .base import Base


class FGM(Base):
    def __init__(self,
                model:nn.Module,
                parameter:str="word_embeddings",
                eps:float=0.2,
                start_epoch:int=0,
                start_step:int=0,
                steps:int=1,
                device:Optional[Union[str, torch.device]]="cpu"):

            """
            Implementation of Fast Gradient sign Method (FGM) method of Adversarial Training - https://arxiv.org/abs/1412.6572
            
            Inputs:
                model:nn.Module - model for adversarial attack.
                parameter:str - parameter for adversarial  attack. Default: "word_embeddings".
                eps:float - Default: 0.2.
                start_epoch:int - Default: 0.
                start_step:int - Default: 0.
                steps:int - number of adversarial attacks to apply in each adversarial step. 
                device: Optional[Union[str, torch.device]] - device for model and batch's data. Default: "cpu".
            
            """

            self.model = model
            self.parameter = parameter
            self.eps = eps
            self.start_epoch = start_epoch
            self.start_step = start_step
            self.steps = steps
            self.device = device
            self.is_cuda = torch.cuda.is_available()

            if not isinstance(self.model, nn.Module):
                raise TypeError(f"`model` must be subinstance of `torch.nn.Module`, but given `{type(self.model)}`")

            if self.eps < 0:
                raise ValueError(f"`eps` must be greater than 0, but given `{self.eps}`")

            if self.start_epoch < 0:
                raise ValueError(f"`start_epoch` must be greater than 0, but given `{self.start_epoch}`")

            if self.start_step < 0:
                raise ValueError(f"`start_epoch` must be greater than 0, but given `{self.start_step}`")

            if self.steps < 0:
                raise ValueError(f"`steps` must be greater than 0, but given `{self.steps}`")

            if self.device is None:
                if self.is_cuda:
                    self.device = "cuda"
                else:
                    self.device = "cpu"


            self.reset()

    def attack(self, batch:Any, epoch:int=0, step:int=0) -> Union[None, torch.Tensor, float]:
        if epoch < self.start_epoch or step < self.start_step:
            return 
        
        loss = 0
        self.save() 
        for step in range(self.steps):
            self.adversarial_step() 
            loss += self.compute_loss(batch=batch)
            self.model.zero_grad()
            
        self.restore()
        
        return loss

    def adversarial_step(self) -> None:
        eps = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.parameter in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    noise = self.eps * param.grad / (norm + eps)
                    param.data.add_(noise)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.parameter in name:
                assert name in self.backup
                param.data = self.backup[name]
            
            self.reset()


    def reset(self) -> None:
        self.backup = {}