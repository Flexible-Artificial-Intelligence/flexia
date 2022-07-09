import torch
from torch import nn
from typing import  Union, Any


class Base:
    def compute_loss(self, batch:Any):

        raise NotImplementedError(f"`compute_loss` function is not implemented.")


    def adversarial_step(self) -> None:
        pass

    def attack(self, batch:Any, epoch:int=0, step:int=0) -> Union[None, torch.Tensor, float]:       
        pass

    def attack(self) -> Union[None, torch.Tensor, float]:
        pass

    def save(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def restore(self) -> None:
        pass