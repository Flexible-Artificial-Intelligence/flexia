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


from datetime import timedelta, datetime
from typing import Tuple
import logging


logger = logging.getLogger(__name__)


class Timer:
    def __init__(self):
        self.start = datetime.now()
        
        self.elapsed_time = timedelta(seconds=0)
        self.remain_time = timedelta(seconds=0)
    

    def state_dict(self) -> dict:
        state = {
            "start": str(self.start),
            "elapsed_time": self.elapsed_time.total_seconds(),
            "remain_time": None if self.remain_time is None else self.remain_time.total_seconds(),
        }
        
        return state
    
    def load_state_dict(self, state_dict:dict) -> "Timer":
        self.start = datetime.fromisoformat(state_dict["start"])
        self.elapsed_time = timedelta(seconds=state_dict["elapsed_time"])
        self.remain_time = timedelta(seconds=state_dict["remain_time"])
        
        return self
    
    
    def __call__(self, fraction:float) -> Tuple[str, str]:                
        self.elapsed_time = datetime.now() - self.start
        elapsed_seconds = self.elapsed_time.total_seconds()        
        total_seconds = timedelta(seconds=round(elapsed_seconds / fraction))
        self.remain_time = abs(total_seconds - self.elapsed_time)
        
        return self.elapsed_time, self.remain_time