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


from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm

from .logger import Logger
from .utils import format_metrics


class TQDMLogger(Logger):
    def __init__(self, 
                 bar_format: str = "{l_bar} {bar} {n_fmt}/{total_fmt} - elapsed: {elapsed} - remain: {remaining}{postfix}", 
                 color: str = "#000", 
                 decimals: int = 4, 
                 notebook: bool = False, 
                 sep: str = " - ", 
                 log_accelerator_stats: bool = False):
        
        super().__init__()
        
        self.bar_format = bar_format
        self.color = color
        self.decimals = decimals
        self.notebook = notebook
        self.sep = sep
        self.log_accelerator_stats = log_accelerator_stats

    def on_epoch_start(self, trainer) -> None:
        epoch = trainer.history["epoch"]
        epochs = trainer.history["epochs"]

        description = f"Epoch {epoch}/{epochs}"
        trainer.train_loader = self.__loader_wrapper(loader=trainer.train_loader, description=description)

    def on_validation_start(self, trainer) -> None:
        description = "Validation"
        trainer.validation_loader = self.__loader_wrapper(loader=trainer.validation_loader, description=description)

    def on_training_step_end(self, trainer) -> None:
        train_loss_epoch = trainer.history["train_loss_epoch"] 
        train_metric_epoch = trainer.history["train_metrics_epoch"]
        lr = trainer.history["lr"]
        
        metrics_string = format_metrics(metrics=train_metric_epoch, decimals=self.decimals)

        string = f"loss: {train_loss_epoch:.{self.decimals}f}{metrics_string}{self.sep}lr: {lr:.{self.decimals}}"
        trainer.train_loader.set_postfix_str(string)

    def on_validation_step_end(self, trainer) -> None:
        validation_loss = trainer.history["validation_loss"]
        validation_metrics = trainer.history["validation_metrics"]

        metrics_string = format_metrics(metrics=validation_metrics, decimals=self.decimals)
        
        string = f"loss: {validation_loss:.{self.decimals}}{metrics_string}"
        trainer.validation_loader.set_postfix_str(string)

    def on_training_end(self, trainer) -> None:
        trainer.train_loader.close()

    def on_validation_end(self, trainer) -> None:
        trainer.validation_loader.close()

    def on_prediction_start(self, trainer) -> None:
        description = "Prediction"
        trainer.prediction_loader = self.__loader_wrapper(loader=trainer.prediction_loader, description=description)

    def on_prediction_end(self, trainer) -> None:
        trainer.prediction_loader.close()

    def __loader_wrapper(self, loader: DataLoader, description: str = ""): 
        steps = len(loader)
        tqdm_wrapper = notebook_tqdm if self.notebook else tqdm
        loader = tqdm_wrapper(iterable=loader, 
                              total=steps,
                              colour=self.color,
                              bar_format=self.bar_format)

        loader.set_description_str(description)

        return loader