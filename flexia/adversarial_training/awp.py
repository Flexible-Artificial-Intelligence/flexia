import torch
from torch import nn
from typing import Optional, Union, Any


from .base import Base


class AWP(Base):
    def __init__(self,
                 model:nn.Module,
                 parameter:str="weight",
                 lr:float=1.0,
                 eps:float=0.2,
                 start_epoch:int=0,
                 start_step:int=0,
                 steps:int=1,
                 device:Optional[Union[str, torch.device]]="cpu"):

        """
        Implementation of Adversarial Weight Perturbation (AWP) method of Adversarial Training - https://arxiv.org/abs/2004.05884
        
        Inputs:
            model:nn.Module - model for adversarial attack.
            parameter:str - parameter for adversarial  attack. Default: "weight".
            lr:float - Default: 1.0.
            eps:float - Default: 0.2.
            start_epoch:int - Default: 0.
            start_step:int - Default: 0.
            steps:int - number of adversarial attacks to apply in each adversarial step. 
            device: Optional[Union[str, torch.device]] - device for model and batch's data. Default: "cpu".
        
        """

        self.model = model
        self.parameter = parameter
        self.lr = lr
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

        if self.lr < 0:
            raise ValueError(f"`lr` must be greater than 0, but given `{self.lr}`")

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
        if epoch <= self.start_epoch or step <= self.start_step:
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
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    norm1, norm2 = norm1 + eps, norm2 + eps
                    noise = self.lr * param.grad / (norm1 * norm2)
                    param.data.add_(noise)
                    backup_eps_nev, backup_eps_pos = self.backup_eps[name]
                    param.data = torch.min(torch.max(param.data, backup_eps_nev), backup_eps_pos)
                # param.data.clamp_(*self.backup_eps[name])

    def save(self) -> None: 
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.parameter in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def reset(self) -> None:
        self.backup = {}
        self.backup_eps = {}

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        self.reset()