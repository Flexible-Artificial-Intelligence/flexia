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
from typing import Any, Dict, List, Optional

from ..import_utils import is_wandb_available


if is_wandb_available():
    import wandb


class WANDBLogger(Logger):
    def __init__(self, 
                 api_key:Optional[str]=None, 
                 finish:bool=True, 
                 summary_values:Dict[str, Dict[str, Any]]=None,
                 on_epoch:bool=True,
                 on_batch:bool=True,
                 **kwargs):

        super().__init__()     
            
        self.api_key = api_key
        self.finish = finish
        self.summary_values = summary_values
        self.on_epoch = on_epoch
        self.on_batch = on_batch
        self.kwargs = kwargs

        if self.api_key is not None:
            wandb.login(key=self.api_key)

        if self.summary_values is not None:
            for summary_name, summary_arguments in self.summary_values.items():
                wandb.define_metric(name=summary_name, **summary_arguments)


        wandb.init(**self.kwargs)
        print(f"Weights & Biases Run URL: {wandb.run.get_url()}")

    def on_training_step_end(self, trainer):
        logs = {
            "train/loss": trainer.history["train_loss"], 
            "lr": trainer.history["lr"],
        }

        if self.on_batch:
            batch_logs = {"train/loss vs batch": trainer.history["train_loss_batch"]}
            logs.update(batch_logs)

        if self.on_epoch:
            epoch_logs = {"train/loss vs epoch": trainer.history["train_loss_epoch"],}
            logs.update(epoch_logs)


        train_metrics = trainer.history["train_metrics"]
        train_metrics_batch = trainer.history["train_metrics_batch"]
        train_metrics_epoch = trainer.history["train_metrics_epoch"]
        step = trainer.history["step"]

        for metric in train_metrics.keys():
            logs.update({
                f"train/{metric}": train_metrics[metric],
            })

            if self.on_batch:
                batch_logs = {f"train/{metric} vs batch": train_metrics_batch[metric]}
                logs.update(batch_logs)

            if self.on_epoch:
                epoch_logs = {f"train/{metric} vs epoch": train_metrics_epoch[metric]}
                logs.update(epoch_logs)

        wandb.log(logs, step=step) 

    def on_validation_end(self, trainer):
        logs = {
            "validation/loss": trainer.history["validation_loss"], 
        }

        validation_metrics = trainer.history["validation_metrics"]
        step = trainer.history["step"]

        for metric in validation_metrics.keys():
            logs.update({
                f"validation/{metric}": validation_metrics[metric], 
            })

        wandb.log(logs, step=step)

    def on_epoch_end(self, trainer):
        if self.on_epoch:
            logs = {
                "epoch": trainer.history["epoch"]
            }
            
            step = trainer.history["step"]

            wandb.log(logs, step=step)

    def on_training_end(self, trainer):
        if self.finish:
            wandb.finish()

    def __wandb_run_exists() -> bool:
        return wandb.run is not None