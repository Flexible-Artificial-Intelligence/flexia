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


class Timer:
    def __init__(self):
        self.start = datetime.now()
        
        self.remain_time = timedelta(seconds=0)

    
    @property
    def elapsed_time(self):
        return datetime.now() - self.start

    def __call__(self, fraction:float) -> Tuple[str, str]:                
        elapsed_seconds = self.elapsed_time.total_seconds()        
        total_seconds = timedelta(seconds=round(elapsed_seconds / fraction))
        self.remain_time = abs(total_seconds - self.elapsed_time)
        
        return self.elapsed_time, self.remain_time