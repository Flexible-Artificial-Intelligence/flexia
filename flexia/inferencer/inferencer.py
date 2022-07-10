import torch
from torch import nn
from torch.cuda.amp import autocast
from typing import Optional, Union, Any
from torch.utils.data import DataLoader
import numpy as np
from abc import ABC, abstractmethod
import logging


from .. import Timer
from .enums import InferencerStates
from ..third_party.addict import Dict
from ..utils import initialize_device


logger = logging.getLogger(__name__)


class Inferencer(ABC):
    def __init__(self, 
                 model:nn.Module, 
                 device:Optional[Union[str, torch.device]]="cpu", 
                 amp:bool=False, 
                 loggers:Union[str, list]=[],
                 callbacks=[], 
                 time_format:str="{hours}:{minutes}:{seconds}"):

        self.model = model
        self.device = initialize_device(self.device)
        self.amp = amp
        self.time_format = time_format
        self.callbacks = callbacks
        self.loggers = loggers

        self.loader = None
        self._state = InferencerStates.INIT
        self.state = self._state
        self.history = Dict()

    def __runner(self, instances=None):
        def run(instance):
            method = getattr(instance, self.state.value)
            method(self)

        if instances is not None:
            if isinstance(instances, list):
                for instance in instances:
                    run(instance)
            else:
                run(instances)


    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

        self.__runner(instances=self.callbacks)
        self.__runner(instances=self.loggers)

    @abstractmethod
    def prediction_step(self, batch:Any):
        pass
        
    def __call__(self, loader:DataLoader):
        self.loader = loader      
        steps = len(self.loader)
        self.history["steps"] = steps
        timer = Timer(self.time_format)
        outputs = []

        self.state = InferencerStates.PREDICTION_START
        
        self.model.to(self.device)
        self.model.eval()   
        for step, batch in enumerate(self.loader, 1):
            self.history["step"] = step
            
            with torch.no_grad():
                with autocast(enabled=self.amp):
                    self.state = InferencerStates.PREDICTION_STEP_START

                    batch_outputs = self.prediction_step(batch=batch)

                    elapsed, remain = timer(self.history["step"]/self.history["steps"])
                    self.history.update({
                        "elapsed": elapsed,
                        "remain": remain,
                    })
                    
                    self.state = InferencerStates.PREDICTION_STEP_END

                    batch_outputs = batch_outputs.to("cpu").numpy()
                    outputs.extend(batch_outputs)
                    
        self.state = InferencerStates.PREDICTION_END

        outputs = np.array(outputs)
    
        return outputs