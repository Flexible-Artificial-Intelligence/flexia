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


from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import  Union, Any
from torch.utils.data import DataLoader

from ..third_party.addict import Dict
from ..timer import Timer
from .enums import InferencerState
from ..utils import initialize_device, precision_dtypes
from ..enums import Precision
from ..hook.utils import run_hooks, exception_handler


class Inferencer(ABC):
    def __init__(self, 
                 model:nn.Module, 
                 device="cpu", 
                 precision="fp32",
                 amp:bool=False, 
                 loggers:Union[str, list]=[],
                 callbacks=[]):


        self.model = model
        self.device = device
        self.precision = Precision(precision)
        self.amp = amp
        self.loggers = loggers
        self.callbacks = callbacks
        self.device = initialize_device(self.device)
        self.device_type = self.device.type

        self._state = InferencerState.INIT_START
        self.state = self._state


        self.precision_dtype = precision_dtypes[self.precision.value]
        self.loader = None
        self.history = Dict()

        self.state = InferencerState.INIT_END

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if self.state != InferencerState.EXCEPTION:
            self._state = value

        run_hooks(hooks=self.loggers, trainer=self)
        run_hooks(hooks=self.callbacks, trainer=self)

    @abstractmethod
    def prediction_step(self, batch:Any):
        pass

    @exception_handler
    def predict(self, loader:DataLoader):
        self.loader = loader      
        steps = len(self.loader)
        self.history["steps"] = steps
        timer = Timer()
        outputs = []

        self.state = InferencerState.PREDICTION_START
        
        self.model.to(self.device)
        self.model.eval()   
        for step, batch in enumerate(self.loader, 1):
            self.history["step"] = step
            
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=self.precision_dtype, enabled=self.amp):
                    self.state = InferencerState.PREDICTION_STEP_START

                    batch_outputs = self.prediction_step(batch=batch)

                    elapsed, remain = timer(self.history["step"]/self.history["steps"])
                    self.history.update({
                        "elapsed": elapsed,
                        "remain": remain,
                    })
                    
                    self.state = InferencerState.PREDICTION_STEP_END

                    batch_outputs = batch_outputs.to("cpu")
                    outputs.extend(batch_outputs)
                    
        self.state = InferencerState.PREDICTION_END

        outputs = torch.stack(outputs, dim=0)
    
        return outputs