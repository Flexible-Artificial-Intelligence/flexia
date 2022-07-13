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

from .logger import Logger
from .utils import get_logger, format_metrics


logger = logging.getLogger(__name__)


class LoggingLogger(Logger):
    def __init__(self, 
                 name="logger", 
                 path="logs.log", 
                 logs_format:str="%(message)s", 
                 verbose:int=1, 
                 decimals=4, 
                 level=logging.INFO) -> None:

        super().__init__()

        self.path = path
        self.name = name
        self.logs_format = logs_format
        self.verbose = verbose
        self.decimals = decimals
        self.level = level

        self.logger = None

    def on_training_start(self, trainer):
        self.logger = get_logger(name=self.name, 
                                 logs_format=self.logs_format, 
                                 path=self.path, 
                                 level=self.level) 

    def on_training_step_end(self, trainer):
        step = trainer.history["step_epoch"]
        steps = trainer.history["steps_epoch"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            elapsed = trainer.history["elapsed_epoch"]
            remain = trainer.history["remain_epoch"]
            train_loss_epoch = trainer.history["train_loss_epoch"]
            train_metrics_epoch = trainer.history["train_metrics"]
            lr = trainer.history["lr"]
            
            metrics_string = format_metrics(metrics=train_metrics_epoch, decimals=self.decimals)
            log_message = f"{step}/{steps} - elapsed: {elapsed} - remain: {remain} - loss: {train_loss_epoch:.{self.decimals}}{metrics_string} - lr: {lr:.{self.decimals}}"
            self.logger.info(log_message)


    def on_validation_step_end(self, trainer):
        step = trainer.history["validation_step"]
        steps = trainer.history["steps_validation"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            loss = trainer.history["validation_loss"]
            metrics = trainer.history["validation_metrics"]
            elapsed = trainer.history["validation_elapsed"]
            remain = trainer.history["validation_remain"]
            
            metrics_string = format_metrics(metrics=metrics, decimals=self.decimals)
            log_message = f"[Validation] {step}/{steps} - elapsed: {elapsed} - remain: {remain} - loss: {loss:.{self.decimals}}{metrics_string}"
            self.logger.info(log_message)


    def on_epoch_start(self, trainer):
        epoch = trainer.history["epoch"]
        epochs = trainer.history["epochs"]

        log_message = f"Epoch {epoch}/{epochs}"
        self.logger.info(log_message)

    def on_prediction_step_end(self, inferencer):
        step = inferencer.history["step"]
        steps = inferencer.history["steps"]
        elapsed = inferencer.history["elapsed"]
        remain = inferencer.history["remain"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            log_message = f"[Prediction] {step}/{steps} - elapsed: {elapsed} - remain: {remain}"
            self.logger.info(log_message)