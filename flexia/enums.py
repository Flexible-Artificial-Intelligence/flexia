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


from enum import Enum


class ExplicitEnum(Enum):
    @classmethod
    def _missing_(cls, value):
        keys = list(cls._value2member_map_.keys())
        raise ValueError(f"`{value}` is not a valid `{cls.__name__}`, select one of `{keys}`.")


class SchedulerLibraries(ExplicitEnum):
    TRANSFORMERS = "transformers"
    TORCH = "torch"


class OptimizerLibraries(ExplicitEnum):
    TRANSFORMERS = "transformers"
    TORCH = "torch"
    BITSANDBYTES = "bitsandbytes"


class Precision(ExplicitEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"