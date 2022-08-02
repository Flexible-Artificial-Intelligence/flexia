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


from .logger import Logger
from .utils import format_metrics, format_time


class PrintLogger(Logger):
    def __init__(self, 
                 verbose:int=1, 
                 decimals:int=3, 
                 sep=" - ", 
                 time_format:str="{hours:02d}:{minutes:02d}:{seconds:02d}", 
                 log_accelerator_stats=False) -> None:
        
        super().__init__()

        self.verbose = verbose
        self.decimals = decimals
        self.time_format = time_format
        self.sep = sep
        self.log_accelerator_stats = log_accelerator_stats

    def on_training_step_end(self, trainer):
        step = trainer.history["step_epoch"]
        steps = trainer.history["steps_epoch"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            elapsed = format_time(trainer.history["elapsed_epoch"], time_format=self.time_format)
            remain = format_time(trainer.history["remain_epoch"], time_format=self.time_format)
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
            elapsed = format_time(trainer.history["validation_elapsed"], time_format=self.time_format)
            remain = format_time(trainer.history["validation_remain"], time_format=self.time_format)
            
            steps_margin = len(str(steps))

            metrics_string = format_metrics(metrics=metrics, decimals=self.decimals, sep=self.sep)

            print(f"[Validation] "
                  f"step: {step:{steps_margin}d}/{steps:{steps_margin}d}{self.sep}"
                  f"elapsed: {elapsed}{self.sep}"
                  f"remain: {remain}{self.sep}"
                  f"loss: {loss:.{self.decimals}f}"
                  f"{metrics_string}")


    def on_prediction_step_end(self, trainer):
        step = trainer.history["prediction_step"]
        steps = trainer.history["prediction_steps"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            elapsed = format_time(trainer.history["prediction_elapsed"], time_format=self.time_format)
            remain = format_time(trainer.history["prediction_remain"], time_format=self.time_format)

            steps_margin = len(str(steps))
            print(f"[Prediction] " 
                  f"step: {step:{steps_margin}d}/{steps:{steps_margin}d}{self.sep}"
                  f"elapsed: {elapsed}{self.sep}"
                  f"remain: {remain}")