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
from typing import Optional, Union, Any, Tuple, List, Dict as TypingDict
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import contextlib
import warnings
import math
import gc

from .enums import TrainerState
from ..timer import Timer
from ..averager import Averager
from ..loggers import Logger
from ..callbacks import Callback
from ..optimization_utils import get_lr
from ..utils import precision_dtypes, mixed_precision_dtypes, seed_everything
from ..third_party.addict import Dict
from ..enums import Precision, IntervalStrategy, DeviceType, GradientClippingStrategy
from ..hooks.utils import run_hook, exception_handler
from ..import_utils import is_torch_xla_available
from ..callbacks import Callbacks
from ..loggers import Loggers
from ..devices import AutoDevice


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


# enabling garbage collection
gc.enable()


class Trainer(ABC):
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: Optional[optim.Optimizer]=None,
                 scheduler: Optional[lr_scheduler._LRScheduler]=None, 
                 scheduling_strategy: Union[IntervalStrategy, str] = "step", 
                 gradient_accumulation_steps: int = 1, 
                 scale_loss_after_gradient_accumulation: bool = False,
                 gradient_scaling: bool = False, 
                 scaler: Optional[GradScaler] = None,
                 precision: Union[Precision, str] = "fp32",
                 gradient_clipping_strategy: Union[GradientClippingStrategy, str] = "off",
                 gradient_clipping_value: Optional[float] = None, 
                 device:Optional[Union[str, torch.device]]="cpu:0", 
                 validation_strategy: Union[IntervalStrategy, str] = "epoch",
                 validation_steps: Union[float, int] = 1, 
                 epochs: int = 1, 
                 loggers: Optional[List["Logger"]] = None, 
                 callbacks: Optional[List["Callback"]] = None, 
                 seed: Optional[int] = None,
                 ) -> None:
        
        self.model = model
        self.model_wrapped = self.model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduling_strategy = IntervalStrategy(scheduling_strategy)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scale_loss_after_gradient_accumulation = scale_loss_after_gradient_accumulation
        self.gradient_scaling = gradient_scaling
        self.precision = Precision(precision)
        self.gradient_clipping_strategy = GradientClippingStrategy(gradient_clipping_strategy)
        self.gradient_clipping_value = gradient_clipping_value
        self.device = AutoDevice(device)
        self.validation_strategy = IntervalStrategy(validation_strategy)
        self.validation_steps = validation_steps
        self.scaler = scaler
        self.epochs = epochs
        self.loggers = loggers
        self.callbacks = callbacks
        self.seed = seed

        self.loggers =  self.loggers if self.loggers is not None else []
        self.loggers = Loggers(self.loggers)
        
        self.callbacks =  self.callbacks if self.callbacks is not None else []
        self.callbacks = Callbacks(self.callbacks)

        self._state = TrainerState.INIT_START
        self.state = self._state

        if not isinstance(self.model, nn.Module):
            raise TypeError(f"`model` must be instance of `nn.Module`, but given `{type(self.model)}`")

        if self.optimizer is not None:
            if not isinstance(self.optimizer, optim.Optimizer):
                raise TypeError(f"`optimizer` must be instance of `torch.optim.Optimizer`, but given `{type(self.optimizer)}`")
        

        self.precision_dtype = precision_dtypes[self.precision.value]
        self.use_amp = self.precision.value in mixed_precision_dtypes


        assert 0 < self.epochs, f"`epochs` must be greater than 0, but given {self.epochs}."
        assert isinstance(self.gradient_accumulation_steps, int), f"`gradient_accumulation_steps` must be integer type, but given `{type(self.gradient_accumulation_steps)}`"


        if self.gradient_scaling and self.use_amp:
            if self.scaler is None:
                self.scaler = GradScaler()
        else:
            self.scaler = None

        self.__apply_gradient_scaling = self.use_amp and self.gradient_scaling and self.scaler is not None

        if self.seed is not None:
            self.seed = seed_everything(seed=self.seed)


        self.history = Dict({
            "step": 0,
            "epoch": 0,
            "best_validation_loss": None,
            "best_validation_metrics": None,
            "best_validation_outputs": None,
        })
        
        self.train_loader, self.validation_loader = None, None
        
        self.state = TrainerState.INIT_END


    def context_manager(self):
        if self.device.device_type != DeviceType.TPU:
            manager = torch.autocast(device_type=self.device.device_type.value, 
                                     dtype=self.precision_dtype, 
                                     enabled=self.use_amp)
        else:
            manager = contextlib.nullcontext()

        return manager

    @property
    def state(self) -> TrainerState:
        return self._state

    @state.setter
    def state(self, value: TrainerState) -> None:
        if self.state != TrainerState.TRAINING_STOP:
            self._state = value

        run_hook(hook=self.loggers, trainer=self)
        run_hook(hook=self.callbacks, trainer=self)
    
    @exception_handler
    def train(self, 
              train_loader: DataLoader, 
              validation_loader: Optional[DataLoader] = None, 
              return_validation_outputs: bool = True
              ) -> TypingDict[str, Any]:
        
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        model = self.model
        model.to(self.device.device)
        
        if self.validation_strategy == IntervalStrategy.EPOCH:
            self.validation_steps = math.ceil(len(self.train_loader) * self.validation_steps)
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

            model.zero_grad(set_to_none=True)
            for step, batch in enumerate(self.train_loader, 1):
                self.history["step"] += 1
                self.history["step_epoch"] = step

                lr = self.get_lr()
                
                batch_size = len(batch)
                
                self.state = TrainerState.TRAINING_STEP_START

                batch_loss, batch_metrics = self.train_one_step(model=model, 
                                                                batch=batch, 
                                                                step=step, 
                                                                steps=steps)

                if self.gradient_accumulation_steps > 1 and self.scale_loss_after_gradient_accumulation:
                    batch_loss = batch_loss * self.gradient_accumulation_steps

                train_loss.update(batch_loss.item(), n=batch_size)
                epoch_train_loss.update(batch_loss.item(), n=batch_size)
                train_metrics.update(batch_metrics, n=batch_size)
                epoch_train_metrics.update(batch_metrics, n=batch_size)

                epoch_fraction = step / steps
                elapsed_epoch, remain_epoch = epoch_timer(epoch_fraction)

                total_fraction = self.history["step"] / self.history["steps"]
                elapsed, remain = timer(total_fraction)

                self.history.update({
                    "train_loss": train_loss.average,
                    "train_loss_batch": batch_loss.item(),
                    "train_loss_epoch": epoch_train_loss.average,
                    "lr": lr,
                    "elapsed": elapsed,
                    "remain": remain,
                    "elapsed_epoch": elapsed_epoch,
                    "remain_epoch": remain_epoch,
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

                        validation_loss, validation_metrics, validation_outputs = self.validate(loader=self.validation_loader, return_validation_outputs=return_validation_outputs)

                        self.scheduling_step(loss=validation_loss, loop="validation")

                        if self.state == TrainerState.CHECKPOINT_SAVE:
                            self.history.update({
                                "best_validation_loss": self.history["validation_loss"],
                                "best_validation_metrics": self.history["validation_metrics"],
                                "best_validation_outputs": validation_outputs,
                            })

                            self.__update_history_data(data=self.history["validation_metrics"], key_format="best_validation_{key}")

                        del validation_outputs
                        gc.collect()

                if self.state == TrainerState.TRAINING_STOP:
                    return self.history

            if self.scheduling_strategy == IntervalStrategy.EPOCH:
                self.scheduling_step(loop="training")

            self.state = TrainerState.EPOCH_END

        self.state = TrainerState.TRAINING_END

        gc.collect()

        return self.history


    def __update_history_data(self, data: TypingDict[str, Any], key_format: str = "train_{key}") -> None:
        new_data_dict = {key_format.format(key=key): value for key, value in data.items()}
        self.history.update(new_data_dict)

    def get_lr(self) -> float:
        num_param_groups = len(self.optimizer.param_groups)
        lr = get_lr(optimizer=self.optimizer, key="lr", groups=num_param_groups - 1)[0]
        return lr

    def backward_step(self, loss: torch.Tensor) -> torch.Tensor:
        if self.__apply_gradient_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss
    

    def optimization_step(self, model: nn.Module) -> None:     
        self.clip_gradients(model)

        if self.__apply_gradient_scaling:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.device.device_type == DeviceType.TPU:
            xm.optimizer_step(self.optimizer, barrier=True)
        else:
            self.optimizer.step()

        model.zero_grad(set_to_none=True)
        

    def scheduling_step(self, loss: Optional[torch.Tensor] = None, loop: str = "training") -> None:
        if self.scheduler is not None:
            if loop == "validation":
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss)
            else:
                if not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()

                    
    def train_one_step(self, 
                       model:nn.Module, 
                       batch:Any, 
                       step: int, 
                       steps: int,
                       ) -> Tuple[torch.Tensor, TypingDict[str, Any]]:
        model.train()
        with self.context_manager():
            batch_loss, batch_metrics, *batch_outputs = self.training_step(model=model, batch=batch)
            self.adversarial_step(model=model, batch=batch)


            if self.gradient_accumulation_steps > 1:
                batch_loss /= self.gradient_accumulation_steps
            
            batch_loss = self.backward_step(loss=batch_loss)

        if (step % self.gradient_accumulation_steps == 0) or (step == steps):
            self.optimization_step(model)

            if self.scheduling_strategy == IntervalStrategy.STEP:
                self.scheduling_step(loop="training")

        return batch_loss.detach(), batch_metrics
                
    def clip_gradients(self, model:nn.Module) -> None:
        if self.gradient_clipping_value is not None and self.gradient_clipping_strategy != GradientClippingStrategy.OFF:
            if self.device.device_type == DeviceType.TPU:
                xm.reduce_gradients(self.optimizer)

            # unscaling gradients before gradient clipping
            if self.__apply_gradient_scaling: 
                self.scaler.unscale_(self.optimizer)
            
            # gradient clipping by norm
            if self.gradient_clipping_strategy == GradientClippingStrategy.NORM:
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.gradient_clipping_value)

            # gradient clipping by value
            elif self.gradient_clipping_strategy == GradientClippingStrategy.VALUE:
                nn.utils.clip_grad_norm_(parameters=model.parameters(), clip_value=self.gradient_clipping_value)


    @exception_handler
    @torch.no_grad()
    def validate(self, 
                 loader: DataLoader, 
                 return_validation_outputs: bool = True, 
                 torchscript:bool = False,
                 ) -> Tuple[Any, TypingDict[str, Any], Any]:
        self.validation_loader = loader

        model = self.model
        model.to(self.device.device)
        model.eval()

        loss, metrics = Averager(), Averager()
        timer = Timer()
        outputs = []
        steps = len(self.validation_loader)
        self.history["steps_validation"] = steps
        
        self.state = TrainerState.VALIDATION_START

        for step, batch in enumerate(self.validation_loader, 1):
            with self.context_manager():
                batch_size = len(batch)

                self.state = TrainerState.VALIDATION_STEP_START

                batch_loss, batch_metrics, batch_outputs = self.validation_step(model=model, batch=batch)

                loss.update(batch_loss.item(), n=batch_size)
                metrics.update(batch_metrics, n=batch_size)

                elapsed, remain = timer(step/steps)

                self.history.update({
                    "validation_step": step,
                    "validation_elapsed": elapsed,
                    "validation_remain": remain,
                    "validation_loss": loss.average,
                    "validation_loss_batch": batch_loss.item(),
                    "validation_metrics": metrics.average,
                    "validation_metrics_batch": batch_metrics,
                })

                    
                self.__update_history_data(data=metrics.average, key_format="validation_{key}")
                self.__update_history_data(data=batch_metrics, key_format="validation_{key}_batch")

                self.state = TrainerState.VALIDATION_STEP_END

                if return_validation_outputs:
                    outputs.append(batch_outputs)

        gc.collect()

        self.on_validation_end(outputs=outputs)
        self.state = TrainerState.VALIDATION_END

        return (loss.average, metrics.average, outputs)

    @abstractmethod
    def training_step(self, model: nn.Module, batch: Any) -> Tuple[torch.Tensor, TypingDict[str, Any], Any]:
        pass

    def validation_step(self, model: nn.Module, batch: Any) -> Tuple[torch.Tensor, TypingDict[str, Any], Any]:
        return self.training_step(model, batch)

    def adversarial_step(self, model: nn.Module, batch: Any) -> None:
        pass

    def on_validation_end(self, outputs: Any) -> None:
        """
        Called when the validation loop ends.
        """

        pass

    # Aliases
    fit = train
    evaluate = validate