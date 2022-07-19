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
from .utils import format_metrics


logger = logging.getLogger(__name__)


class PrintLogger(Logger):
    def __init__(self, verbose:int=1, decimals:int=3, sep=" - ") -> None:
        super().__init__()

        self.verbose = verbose
        self.decimals = decimals
        self.sep = sep

    def on_training_step_end(self, trainer):
        step = trainer.history["step_epoch"]
        steps = trainer.history["steps_epoch"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            elapsed = trainer.history["elapsed_epoch"]
            remain = trainer.history["remain_epoch"]
            train_loss_epoch = trainer.history["train_loss_epoch"]
            train_metrics_epoch = trainer.history["train_metrics"]
            lr = trainer.history["lr"]
            epoch = trainer.history["epoch"]
            epochs = trainer.history["epochs"]
            
            steps_margin = len(str(steps))
            epochs_margin = len(str(epochs))

            metrics_string = format_metrics(metrics=train_metrics_epoch, decimals=self.decimals, sep=self.sep)

            print(f"epoch: {epoch:{epochs_margin}d}/{epochs:{epochs_margin}d}{self.sep}"
                  f"step: {step:{steps_margin}d}/{steps:{steps_margin}d}{self.sep}"
                  f"elapsed: {elapsed}{self.sep}"
                  f"remain: {remain}{self.sep}"
                  f"loss: {train_loss_epoch:.{self.decimals}f}"
                  f"{metrics_string}{self.sep}"
                  f"lr: {lr:.{self.decimals}f}")


    def on_validation_step_end(self, trainer):
        step = trainer.history["validation_step"]
        steps = trainer.history["steps_validation"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            loss = trainer.history["validation_loss"]
            metrics = trainer.history["validation_metrics"]
            elapsed = trainer.history["validation_elapsed"]
            remain = trainer.history["validation_remain"]
            
            steps_margin = len(str(steps))

            metrics_string = format_metrics(metrics=metrics, decimals=self.decimals, sep=self.sep)

            print(f"[Validation] {step:{steps_margin}d}/{steps:{steps_margin}d}{self.sep}"
                  f"elapsed: {elapsed}{self.sep}"
                  f"remain: {remain}{self.sep}"
                  f"loss: {loss:.{self.decimals}f}"
                  f"{metrics_string}")


    def on_prediction_step_end(self, inferencer):
        step = inferencer.history["step"]
        steps = inferencer.history["steps"]
        elapsed = inferencer.history["elapsed"]
        remain = inferencer.history["remain"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            steps_margin = len(str(steps))
            print(f"[Prediction] {step:{steps_margin}d}/{steps:{steps_margin}d}{self.sep}"
                  f"elapsed: {elapsed}{self.sep}"
                  f"remain: {remain}{self.sep}")