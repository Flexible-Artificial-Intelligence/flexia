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
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from typing import Optional, Union, Any, Tuple, List
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import os

from .enums import TrainerState
from ..timer import Timer
from ..averager import Averager
from ..loggers import Logger
from ..callbacks import Callback
from ..utils import get_lr, initialize_device, precision_dtypes
from ..third_party.addict import Dict
from ..enums import Precision, IntervalStrategy, DeviceType
from ..hook.utils import run_hooks, exception_handler
from ..import_utils import is_torch_xla_available


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm




class Trainer(ABC):
    def __init__(self, 
                 model:nn.Module, 
                 optimizer:optim.Optimizer,
                 scheduler:Optional[lr_scheduler._LRScheduler]=None, 
                 scheduling_strategy:str="step", 
                 gradient_accumulation_steps:int=1, 
                 gradient_scaling:bool=False, 
                 scaler:Optional["GradScaler"]=None,
                 precision="fp32",
                 amp=False,
                 gradient_norm:float=None, 
                 device:Optional[Union[str, torch.device]]="cpu", 
                 validation_strategy:str="epoch",
                 validation_steps:int=1, 
                 loggers:Optional[List["Logger"]]=None, 
                 epochs:int=1, 
                 callbacks=Optional[List["Callback"]]) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduling_strategy = IntervalStrategy(scheduling_strategy)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_scaling = gradient_scaling
        self.precision = Precision(precision)
        self.gradient_norm = gradient_norm
        self.amp = amp
        self.device = device
        self.validation_strategy = IntervalStrategy(validation_strategy)
        self.validation_steps = validation_steps
        self.scaler = scaler
        self.loggers = loggers
        self.epochs = epochs
        self.callbacks = callbacks

        self._state = TrainerState.INIT_START
        self.state = self._state
        

        if not isinstance(self.model, nn.Module):
            raise TypeError("model")

        if not isinstance(self.optimizer, optim.Optimizer):
            raise TypeError("optimizer")


        assert 0 < self.epochs, f"`epochs` must be greater than 0, but given {self.epochs}."
        assert isinstance(self.gradient_accumulation_steps, int), f"`gradient_accumulation_steps` must be integer type, but given `{type(self.gradient_accumulation_steps)}`"

        self.precision_dtype = precision_dtypes[self.precision.value]
        self.device = initialize_device(self.device)
        self.device_type = DeviceType(self.device.type)

        if self.gradient_scaling and self.scaler is None and self.amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.history = Dict({
            "step": 0,
            "epoch": 0,
            "best_validation_loss": None,
            "best_validation_metrics": None,
            "best_validation_outputs": None,
        })
        
        self.train_loader, self.validation_loader = None, None
        
        self.state = TrainerState.INIT_END


    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if self.state != TrainerState.TRAINING_STOP:
            self._state = value

        run_hooks(hooks=self.loggers, trainer=self)
        run_hooks(hooks=self.callbacks, trainer=self)
    
    @exception_handler
    def train(self, 
              train_loader:DataLoader, 
              validation_loader:Optional[DataLoader]=None, 
              return_validation_outputs:bool=True) -> tuple:
        
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.return_validation_outputs = return_validation_outputs

        self.model.to(self.device)
        
        if self.validation_strategy == IntervalStrategy.EPOCH:
            self.validation_steps = len(self.train_loader) * self.validation_steps
        else:
            # validation model after N training steps!
            self.validation_steps = int(self.validation_steps * self.gradient_accumulation_steps)

        steps = len(self.train_loader)    
        
        self.history.update({
            "epochs": self.epochs, 
            "steps": int(self.epochs*steps),
            "steps_epoch": steps,
            "step_epoch": 0,
        })

        train_loss, train_metrics = Averager(), Averager()

        self.state = TrainerState.TRAINING_START

        timer = Timer()
        for epoch in range(1, self.epochs+1):
            self.history["epoch"] = epoch

            epoch_train_loss, epoch_train_metrics = Averager(), Averager()
            epoch_timer = Timer()
            
            self.state = TrainerState.EPOCH_START

            self.model.zero_grad(set_to_none=True)
            for step, batch in enumerate(self.train_loader, 1):
                self.history["epoch"] = epoch
                self.history["step"] += 1
                self.history["step_epoch"] = step
                
                batch_size = len(batch)
                
                self.state = TrainerState.TRAINING_STEP_START

                batch_loss, batch_metrics = self.training_step(batch=batch)

                lr = self.get_lr()

                if (step % self.gradient_accumulation_steps == 0) or (step == steps):
                    self.optimization_step()

                    if self.scheduling_strategy == IntervalStrategy.STEP:
                        self.scheduling_step(loop="training")

                if self.gradient_accumulation_steps > 1:
                    batch_loss = batch_loss * self.gradient_accumulation_steps

                train_loss.update(batch_loss.item(), n=batch_size)
                epoch_train_loss.update(batch_loss.item(), n=batch_size)
                train_metrics.update(batch_metrics, n=batch_size)
                epoch_train_metrics.update(batch_metrics, n=batch_size)

                elapsed_epoch, remain_epoch = epoch_timer(step/steps)
                elapsed, remain = timer(self.history["step"]/self.history["steps"])

                self.history.update({
                    "train_loss": train_loss.average,
                    "train_loss_batch": batch_loss.item(),
                    "train_loss_epoch": epoch_train_loss.average,
                    "lr": lr,
                    "elapsed": elapsed,
                    "remain": remain,
                    "elapsed_epoch": elapsed_epoch,
                    "remain_epoch": remain_epoch,
                    "train_metrics_list": list(train_metrics.average.keys()),
                    "train_metrics": train_metrics.average,
                    "train_metrics_batch": batch_metrics,
                    "train_metrics_epoch": epoch_train_metrics.average,
                })

                self.__update_history_data(data=train_metrics.average, key_format="train_{key}")
                self.__update_history_data(data=batch_metrics, key_format="train_{key}_batch")
                self.__update_history_data(data=epoch_train_metrics.average, key_format="train_{key}_epoch")
                

                self.state = TrainerState.TRAINING_STEP_END

                if self.validation_loader is not None:
                    if (self.history["step"] % self.validation_steps) == 0:

                        validation_loss, validation_metrics, validation_outputs = self.validation_loop(loader=self.validation_loader)

                        self.scheduling_step(loss=validation_loss, loop="validation")

                        if self.state == TrainerState.CHECKPOINT_SAVE:
                            self.history.update({
                                "best_validation_loss": self.history["validation_loss"],
                                "best_validation_metrics": self.history["validation_metrics"],
                                "best_validation_outputs": validation_outputs,
                            })

                            self.__update_history_data(data=self.history["validation_metrics"], key_format="best_validation_{key}")

                        del validation_outputs

                if self.state == TrainerState.TRAINING_STOP:
                    return self.history

            if self.scheduling_strategy == IntervalStrategy.EPOCH:
                self.scheduling_step(loop="training")

            self.state = TrainerState.EPOCH_END

        self.state = TrainerState.TRAINING_END

        return self.history


    def __update_history_data(self, data, key_format="train_{key}"):
        new_data_dict = {key_format.format(key=key): value for key, value in data.items()}
        self.history.update(new_data_dict)

    def get_lr(self):
        return get_lr(optimizer=self.optimizer, only_last=True, key="lr")

    def backward_step(self, loss:torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss
    

    def optimization_step(self) -> None:     
        self.clip_gradients()

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.device_type == DeviceType.TPU:
            xm.optimizer_step(self.optimizer, barrier=True)
        else:
            self.optimizer.step()

        self.model.zero_grad(set_to_none=True)
        

    def scheduling_step(self, loss:Optional[torch.Tensor]=None, loop:str="training") -> None:
        if self.scheduler is not None:
            if loop == "validation":
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss)
            else:
                if not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()

                    
    def training_step(self, batch:Any) -> Tuple[torch.Tensor, dict]:
        self.model.train()
        with torch.autocast(device_type=self.device.type, dtype=self.precision_dtype, enabled=self.amp):
            loss, outputs = self.compute_loss(batch=batch, return_outputs=True)
            outputs = outputs.detach()
            metrics = self.compute_metrics(batch=batch, outputs=outputs)

            if self.gradient_accumulation_steps > 1:
                loss /= self.gradient_accumulation_steps
            
            loss = self.backward_step(loss=loss)

        return loss.detach(), metrics
                
    def clip_gradients(self) -> None:
        if self.gradient_norm is not None:
            if self.device_type == DeviceType.TPU:
                xm.reduce_gradients(self.optimizer)

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_norm)
    

    @exception_handler
    def validation_loop(self, loader:DataLoader) -> Tuple[Any, dict]:
        self.validation_loader = loader

        self.model.to(self.device)
        self.model.eval()
        loss, metrics = Averager(), Averager()
        timer = Timer()
        outputs, targets = [], []
        steps = len(self.validation_loader)
        self.history["steps_validation"] = steps
        
        self.state = TrainerState.VALIDATION_START

        for step, batch in enumerate(self.validation_loader, 1):
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=self.precision_dtype, enabled=self.amp):
                    batch_size = len(batch)

                    self.state = TrainerState.VALIDATION_STEP_START

                    batch_loss, batch_outputs = self.compute_loss(batch=batch, return_outputs=True)
                    batch_outputs = batch_outputs.detach()
                    batch_metrics = self.compute_metrics(batch=batch, outputs=batch_outputs)

                    loss.update(batch_loss.item(), n=batch_size)
                    metrics.update(batch_metrics, n=batch_size)

                    elapsed, remain = timer(step/steps)

                    self.history.update({
                        "validation_step": step,
                        "validation_elapsed": elapsed,
                        "validation_remain": remain,
                        "validation_loss": loss.average,
                        "validation_loss_batch": batch_loss.item(),
                        "validation_metrics_list": list(metrics.average.keys()),
                        "validation_metrics": metrics.average,
                        "validation_metrics_batch": batch_metrics,
                    })

                    
                    self.__update_history_data(data=metrics.average, key_format="validation_{key}")
                    self.__update_history_data(data=batch_metrics, key_format="validation_{key}_batch")

                    self.state = TrainerState.VALIDATION_STEP_END

                    if self.return_validation_outputs:
                        outputs.extend(batch_outputs.to("cpu"))

        if self.return_validation_outputs:
            outputs = torch.stack(outputs, dim=0)
        else:
            outputs = None

        self.on_validation_end(outputs=outputs)
        self.state = TrainerState.VALIDATION_END

        return (loss.average, metrics.average, outputs)

    @abstractmethod
    def compute_loss(self, 
                      batch:Any, 
                      return_outputs:bool=True) -> torch.Tensor:
        pass

    def compute_metrics(self, batch:Any, outputs):
        return {}

    def on_validation_end(self, outputs) -> None:
        """
        Called when the validation loop ends.
        """

        pass