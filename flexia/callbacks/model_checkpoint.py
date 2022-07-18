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
import shutil
import os
import gc
import numpy as np
from typing import  Union
import logging


from .callback import Callback
from ..utils import save_checkpoint
from ..trainer.enums import TrainerStates
from .utils import get_delta_value, compare
from .enums import Modes


logger = logging.getLogger(__name__)


class ModelCheckpoint(Callback):  
    def __init__(self, 
                 monitor_value="validation_loss",
                 mode:str="min", 
                 delta:Union[float, int]=0.0, 
                 directory:str="./", 
                 overwriting:bool=False, 
                 filename_format:str="checkpoint_{step}_{value}.pth", 
                 num_candidates:Union[str, float, int]=1, 
                 save_optimizer_state=True, 
                 save_scheduler_state=True, 
                 custom_keys={"model": "model_state",  
                              "optimizer": "optimizer_state", 
                              "scheduler": "scheduler_state"}, 
                save_checkpoint_on_exception=True):
        
        self.monitor_value = monitor_value
        self.mode = Modes(mode)
        self.delta = delta
        self.directory = directory
        self.overwriting = overwriting
        self.filename_format = filename_format
        self.num_candidates = num_candidates
        self.save_optimizer_state = save_optimizer_state
        self.save_scheduler_state = save_scheduler_state
        self.custom_keys = custom_keys
        self.save_checkpoint_on_exception = save_checkpoint_on_exception
        
        self.best_value = np.inf if self.mode == Modes.MIN else -np.inf
        
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
                    self.__remove_files_from_directory(self.directory)
            else:
                raise NotADirectoryError(f"'{self.directory}' is not directory.")
        
        self.all_candidates = []
    
    
    def __remove_files_from_directory(self, directory:str) -> None:
        """
        Removes all files and folders from directory.
        """
        
        filenames = os.listdir(directory)
        pathes = [os.path.join(directory, filename) for filename in filenames]
        
        for path in pathes:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    
    
    def append_candidate(self, path:str) -> None:   
        """
        Appends new candidate.
        """
        
        if not os.path.exists(path):
            raise FileNotFoundError("`path` does not exist.")
        
        self.all_candidates.append(path)
        
    
    def __select_candidates(self) -> None:
        """
        Deleted not selected candidates.
        """
        if len(self.all_candidates) > self.num_candidates:
            selected_candidates = self.all_candidates[-self.num_candidates:]
            deleted_candidates = 0
            for candidate_path in self.all_candidates:
                if candidate_path not in selected_candidates:                        
                    if os.path.exists(candidate_path):
                        os.remove(candidate_path)

                    deleted_candidates += 1
                
            self.all_candidates = self.all_candidates[-self.num_candidates:]
                
            
    def format_filename(self, filename_format="checkpoint.pth", data={}) -> str:
        filename = filename_format.format(**data)            
        return filename
            
    def check(self, trainer) -> bool:
        value = trainer.history[self.monitor_value]
        delta_value = get_delta_value(value=value, delta=self.delta, mode=self.mode)

        is_saved = False
        if compare(value=delta_value, other=self.best_value, mode=self.mode) and self.num_candidates != 0:
            checkpoint_filename = self.format_filename(filename_format=self.filename_format, data=trainer.history)
            checkpoint_path = os.path.join(self.directory, checkpoint_filename)
            
            checkpoint = save_checkpoint(model=trainer.model, 
                                         optimizer=trainer.optimizer if self.save_optimizer_state else None, 
                                         scheduler=trainer.scheduler if self.save_scheduler_state else None, 
                                         custom_keys=self.custom_keys, 
                                         path=checkpoint_path, 
                                         step=trainer.history["step"], 
                                         epoch=trainer.history["epoch"])
            
            improvement_delta = abs(value - self.best_value)
            message = f"'best_value' is improved by {improvement_delta}! New 'best_value': {value}. Checkpoint path: '{checkpoint_path}'."
            print(message)

            self.append_candidate(checkpoint_path)
            
            self.best_value = value
            trainer.history["best_checkpoint_path"] = checkpoint_path
            is_saved = True

            self.__select_candidates()

            # removing checkpoint from memory
            del checkpoint
            gc.collect()
        
        return is_saved


    def on_validation_end(self, trainer):
        is_saved = self.check(trainer=trainer)
        if is_saved:
            trainer.state = TrainerStates.CHECKPOINT_SAVE


    def on_exception(self, trainer):
        if self.save_checkpoint_on_exception:
            filename_format = "checkpoint_step_{step}_epoch_{epoch}.pth"
            checkpoint_filename = self.format_filename(filename_format=filename_format, data=trainer.history)
            checkpoint_path = os.path.join(self.directory, checkpoint_filename)

            checkpoint = save_checkpoint(model=trainer.model, 
                                        optimizer=trainer.optimizer if self.save_optimizer_state else None, 
                                        scheduler=trainer.scheduler if self.save_scheduler_state else None, 
                                        custom_keys=self.custom_keys, 
                                        path=checkpoint_path, 
                                        step=trainer.history["step"], 
                                        epoch=trainer.history["epoch"])

        