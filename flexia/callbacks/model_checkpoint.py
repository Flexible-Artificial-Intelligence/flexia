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


import os
import gc
import re
import numpy as np
import torch
from typing import Union, Dict, Optional

from .callback import Callback
from ..utils import save_checkpoint, default_checkpoint_custom_keys
from ..trainer.enums import TrainerState
from .utils import get_delta_value, compare
from ..python_utils import remove_files_from_directory
from ..enums import IntervalStrategy
from ..module_utils import save_module
from .enums import Mode


class ModelCheckpoint(Callback):  
    def __init__(self, 
                 monitor_value:str="validation_loss",
                 mode:Union[Mode, str]="min", 
                 delta:Union[float, int]=0.0, 
                 directory:str="./", 
                 overwriting:bool=False, 
                 filename_format:str="checkpoint.pt", 
                 num_candidates:Union[str, float, int]=1, 
                 save_model_on_end:bool=False,
                 model_path:bool=None,
                 save_optimizer_state:bool=True, 
                 save_scheduler_state:bool=True, 
                 save_scaler_state:bool=False,
                 custom_keys:Dict[str, str]=default_checkpoint_custom_keys, 
                 save_checkpoint_on_exception:bool=False, 
                 on_exception_filename_format:Optional[str]=None,
                 save_interval:Optional[int]=None, 
                 save_interval_strategy:Union[IntervalStrategy, str]="off", 
                 save_interval_directory:Optional[str]=None, 
                 save_interval_filename_format:Optional[str]=None):
        
        super().__init__()

        self.monitor_value = monitor_value
        self.mode = Mode(mode)
        self.delta = delta
        self.directory = directory
        self.overwriting = overwriting
        self.filename_format = filename_format
        self.num_candidates = num_candidates
        self.save_model_on_end = save_model_on_end
        self.model_path = model_path
        self.save_optimizer_state = save_optimizer_state
        self.save_scheduler_state = save_scheduler_state
        self.save_scaler_state = save_scaler_state
        self.custom_keys = custom_keys
        self.save_checkpoint_on_exception = save_checkpoint_on_exception
        self.save_interval = save_interval
        self.save_interval_strategy = IntervalStrategy(save_interval_strategy)
        self.save_interval_directory = save_interval_directory
        self.save_interval_filename_format = save_interval_filename_format
        
        self.best_value = np.inf if self.mode == Mode.MIN else -np.inf
        
        if isinstance(self.num_candidates, str):
            if self.num_candidates != "all":
                raise ValueError(f"`num_candidates` can be a string, but only with 1 value: 'all', but given '{self.num_candidates}'")
        
        if not os.path.exists(self.directory):
            if self.overwriting:
                os.mkdir(self.directory)
            else:
                raise FileNotFoundError(f"Directory '{self.directory}' does not exist.")
        else:
            if os.path.isdir(self.directory):
                if self.overwriting:
                    remove_files_from_directory(directory=self.directory)
            else:
                raise NotADirectoryError(f"'{self.directory}' is not directory.")

        if self.model_path is None:
            self.model_path = os.path.join(self.directory, "model.pt")

        if self.save_interval_filename_format is None:
            if self.save_interval_strategy == IntervalStrategy.EPOCH:
                self.save_interval_filename_format = "checkpoint_{epoch}.pt"
            else:
                self.save_interval_filename_format = "checkpoint_{step}.pt"
    
        self.all_candidates = []
        self.all_interval_candidates = []
    
    @classmethod
    def check_filename_format_uniqueness(filename_format):
        regex = '{([^"]*)}'
        results = re.findall(regex, filename_format)

        return len(results) > 0
    
    def append_candidate(self, candidates_list, path) -> None:   
        """
        Appends new candidate.
        """
        
        if not os.path.exists(path):
            raise FileNotFoundError("`path` does not exist.")
        
        candidates_list.append(path)
        
    
    def __select_candidates(self, candidates_list) -> None:
        """
        Deleted not selected candidates.
        """
        if len(candidates_list) > self.num_candidates:
            selected_candidates = candidates_list[-self.num_candidates:]
            deleted_candidates = 0
            for candidate_path in candidates_list:
                if candidate_path not in selected_candidates:                        
                    if os.path.exists(candidate_path):
                        os.remove(candidate_path)

                    deleted_candidates += 1
                
            candidates_list = candidates_list[-self.num_candidates:]
                
            
    def format_filename(self, filename_format="checkpoint.pt", data={}) -> str:
        filename = filename_format.format(**data)            
        return filename
            
    def check(self, trainer) -> bool:
        value = trainer.history[self.monitor_value]
        delta_value = get_delta_value(value=value, delta=self.delta, mode=self.mode)

        is_saved = False
        if compare(value=delta_value, other=self.best_value, mode=self.mode) and self.num_candidates != 0:
            checkpoint_path, checkpoint = self.save_checkpoint(trainer=trainer, 
                                                               directory=self.directory, 
                                                               filename_format=self.filename_format)
            
            improvement_delta = abs(value - self.best_value)
            message = f"'best_value' is improved by {improvement_delta}! New 'best_value': {value}. Checkpoint path: '{checkpoint_path}'."
            print(message)

            self.append_candidate(candidates_list=self.all_candidates, path=checkpoint_path)
            
            self.best_value = value
            trainer.history["best_checkpoint_path"] = checkpoint_path
            is_saved = True

            self.__select_candidates(candidates_list=self.all_candidates)

            # removing checkpoint from memory
            del checkpoint
            gc.collect()
        
        return is_saved


    def on_validation_end(self, trainer):
        is_saved = self.check(trainer=trainer)
        if is_saved:
            trainer.state = TrainerState.CHECKPOINT_SAVE


    def __check_interval(self, trainer, interval_strategy=IntervalStrategy.OFF):
        if self.save_interval_strategy == interval_strategy and self.save_interval is not None:
            interval = trainer.history[interval_strategy.value]

            if interval % self.save_interval == 0:
                self.save_interval_checkpoint(trainer=trainer)


    def on_training_step_end(self, trainer) -> None:
        self.__check_interval(trainer=trainer, interval_strategy=IntervalStrategy.STEP)

    
    def on_epoch_end(self, trainer) -> None:
        self.__check_interval(trainer=trainer, interval_strategy=IntervalStrategy.EPOCH)


    def save_interval_checkpoint(self, trainer):
        checkpoint_path, checkpoint = self.save_checkpoint(trainer=trainer, 
                                                           directory=self.save_interval_directory, 
                                                           filename_format=self.save_interval_filename_format)
                
        self.append_candidate(candidates_list=self.all_interval_candidates, path=checkpoint_path)
        self.__select_candidates(candidates_list=self.all_interval_candidates)


    def save_checkpoint(self, trainer, directory=None, filename_format="checkpoint.pt", **kwargs):
        if directory is None:
            directory = self.directory

        checkpoint_filename = self.format_filename(filename_format=filename_format, data=trainer.history)
        checkpoint_path = os.path.join(directory, checkpoint_filename)

        checkpoint = save_checkpoint(
            model=trainer.model, 
            optimizer=trainer.optimizer if self.save_optimizer_state else None, 
            scheduler=trainer.scheduler if self.save_scheduler_state else None, 
            scaler=trainer.scaler if trainer.scaler is not None and self.save_scaler_state else None,
            custom_keys=self.custom_keys, 
            path=checkpoint_path, 
            step=trainer.history["step"], 
            epoch=trainer.history["epoch"],
            monitor_value=trainer.history[self.monitor_value], 
            device_type=trainer.accelerator.device_type,
            **kwargs
        )

        return checkpoint_path, checkpoint

    def on_exception(self, trainer):
        if self.save_checkpoint_on_exception:
            filename_format = "last_checkpoint_step_{step}_epoch_{epoch}.pt"
            checkpoint_path, checkpoint = self.save_checkpoint(trainer=trainer, filename_format=filename_format)

    
    def on_training_end(self, trainer) -> None:
        if self.save_model_on_end:
            save_module(module=trainer.model, path=self.model_path)