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


import logging


logger = logging.getLogger(__name__)

class LambdaCallback:
    def __init__(self, 
                 on_init=lambda trainer: None, 
                 on_training_step_start=lambda trainer: None, 
                 on_training_step_end=lambda trainer: None, 
                 on_validation_step_start=lambda trainer: None, 
                 on_validation_step_end=lambda trainer: None, 
                 on_epoch_start=lambda trainer: None, 
                 on_epoch_end=lambda trainer: None, 
                 on_validation_start=lambda trainer: None, 
                 on_validation_end=lambda trainer: None, 
                 on_training_start=lambda trainer: None, 
                 on_training_end=lambda trainer: None,  
                 on_training_stop=lambda trainer: None, 
                 on_checkpoint_save=lambda trainer: None, 
                 on_exception=lambda exception, trainer: None,
                 on_prediction_start=lambda inferencer: None,
                 on_prediction_end=lambda inferencer: None,
                 on_prediction_step_start=lambda inferencer: None,
                 on_prediction_step_end=lambda inferencer: None):

        self.on_init = on_init
        self.on_training_step_start = on_training_step_start
        self.on_training_step_end = on_training_step_end
        self.on_epoch_start = on_epoch_start
        self.on_epoch_end = on_epoch_end
        self.on_validation_start = on_validation_start
        self.on_validation_end = on_validation_end
        self.on_validation_step_start = on_validation_step_start
        self.on_validation_step_end = on_validation_step_end
        self.on_training_start = on_training_start
        self.on_training_end = on_training_end
        self.on_training_stop = on_training_stop
        self.on_checkpoint_save = on_checkpoint_save
        self.on_exception = on_exception
        self.on_prediction_start = on_prediction_start
        self.on_prediction_end = on_prediction_end
        self.on_prediction_step_start = on_prediction_step_start
        self.on_prediction_step_end = on_prediction_step_end
