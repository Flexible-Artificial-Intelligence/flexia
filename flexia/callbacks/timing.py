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


from typing import Union
from datetime import timedelta
import logging

from .callback import Callback
from ..trainer.enums import TrainerStates


logger = logging.getLogger(__name__)

class Timing(Callback):
    def __init__(self, 
                 duration:Union[str, timedelta]="01:00:00:00", 
                 duration_separator:str=":"):
        
        super().__init__()
        
        self.duration = duration
        self.duration_separator = duration_separator

        if isinstance(self.duration, str):
            duration_values = self.duration.strip().split(self.duration_separator)
            duration_values = tuple([int(value) for value in duration_values])
            days, hours, minutes, seconds = duration_values 
            
            self.duration = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

        elif isinstance(self.duration, dict):
            self.duration = timedelta(**self.duration)
        elif not isinstance(self.duration, timedelta):
            raise TypeError(f"Type of given `duration` is not supported.")
            
        self.stop = False
        

    def on_epoch_end(self, trainer):
        elapsed = trainer.history["elapsed"]
        self.stop = self.check(elapsed)
        
        if self.stop:
            trainer.state = TrainerStates.TRAINING_STOP
        
    def check(self, elapsed) -> bool:       
        stop = elapsed > self.duration
        
        if stop:
            case = "The time reaches duration limit."
            logger.info(case)
        
        return stop