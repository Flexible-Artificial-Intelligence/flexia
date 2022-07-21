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


class TrainerState(Enum):
    INIT_START = "on_init_start"
    INIT_END = "on_init_end"
    TRAINING_START = "on_training_start"
    TRAINING_END = "on_training_end"
    TRAINING_STEP_START = "on_training_step_start"
    TRAINING_STEP_END = "on_training_step_end"
    VALIDATION_START = "on_validation_start"
    VALIDATION_END = "on_validation_end"
    EPOCH_START = "on_epoch_start"
    VALIDATION_STEP_START = "on_validation_step_start"
    VALIDATION_STEP_END = "on_validation_step_end"
    EPOCH_END = "on_epoch_end"
    TRAINING_STOP = "on_training_stop"
    CHECKPOINT_SAVE = "on_checkpoint_save"
    EXCEPTION = "on_exception"
    PREDICTION_START = "on_prediction_start"
    PREDICTION_END = "on_prediction_end"
    PREDICTION_STEP_START = "on_prediction_step_start"
    PREDICTION_STEP_END = "on_prediction_step_end"