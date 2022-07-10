from typing import Callable, Union, Optional
import numpy as np
import logging

from ..trainer.trainer_enums import TrainingStates
from . import Callback
from .utils import compare, get_delta_value
from .enums import Modes


logger = logging.Logger(__name__)


class EarlyStopping(Callback):   
    def __init__(self, 
                 monitor_value="validation_loss",
                 mode:str="min", 
                 delta:Union[float, int]=0.0, 
                 patience:Union[float, int]=5, 
                 stopping_threshold:Optional[float]=None, 
                 check_finite:bool=False):
        
        self.monitor_value = monitor_value
        self.mode = Modes(mode)
        self.delta = delta
        self.patience = patience
        self.stopping_threshold = stopping_threshold
        self.check_finite = check_finite
        
        self.stop = False
        self.case = None
        self.fails = 0
        self.best_value = np.inf if self.mode == Modes.MIN else -np.inf
        
        if self.patience < 0:
            raise ValueError(f"`patience` must be in range (0, +inf), but given {self.patience}.")
        else:
            if not isinstance(self.patience, int):
                self.patience = round(self.patience)

        if not self.monitor_value.startswith("validation"):
            raise ValueError(f"Early Stopping works only on validation loss and metrics.")


    def on_validation_end(self, trainer):
        value = trainer.history[self.monitor_value]
        self.check(value=value)

        if self.stop:
            trainer.state = TrainingStates.TRAINING_STOP
            logger.info(self.case)

    
    def check(self, value) -> bool:               
        delta_value = get_delta_value(value=value, delta=self.delta, mode=self.mode)
        
        if not self.stop:
            if self.check_finite:
                if not np.isfinite(value):
                    self.stop = True
                    self.case = f"The value is not finite, maybe problem of Gradient Exploding."

            if self.stopping_threshold is not None:
                if compare(value=self.stopping_threshold, other=delta_value, mode=self.mode):
                    self.stop = True
                    self.case = f"Monitored value reached `stopping_threshold`. Value: {self.value}. Stopping threshold: {self.stopping_threshold}."
            
            if compare(value=delta_value, other=self.best_value, mode=self.mode):
                improvement_delta = abs(value - self.best_value)
                self.case = f"Moniroted value is improved by {improvement_delta}! New `best_value`: {value}."
                self.best_value = value
                self.fails = 0
            else:
                self.fails += 1
            
            if self.fails >= self.patience:
                self.stop = True
                self.case = f"Number of attempts has been expired. The best monitored value wasn't beaten." 
        
        return self.stop