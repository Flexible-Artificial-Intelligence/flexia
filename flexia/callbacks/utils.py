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


from .enums import Mode


def get_delta_value(value, delta=0.0, mode="min"):
    mode = Mode(mode)
    delta_value = (value - delta) if mode == Mode.MIN else (value + delta)
    
    return delta_value


def compare(value, other, mode="min"):
    mode = Mode(mode)
    condition = (value < other) if mode == Mode.MIN else (value > other)
    
    return condition