import logging

from . import Logger
from .utils import format_metrics


logger = logging.getLogger(__name__)


class PrintLogger(Logger):
    def __init__(self, verbose:int=1, decimals:int=4) -> None:
        super().__init__()

        self.verbose = verbose
        self.decimals = decimals

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
            print(log_message)


    def on_validation_step_end(self, trainer):
        step = trainer.history["validation_step"]
        steps = trainer.history["steps_validation"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            loss = trainer.history["validation_loss"]
            metrics = trainer.history["validation_metrics"]
            elapsed = trainer.history["elapsed_epoch"]
            remain = trainer.history["remain_epoch"]
            
            metrics_string = format_metrics(metrics=metrics, decimals=self.decimals)
            log_message = f"[Validation] {step}/{steps} - elapsed: {elapsed} - remain: {remain} - loss: {loss:.{self.decimals}}{metrics_string}"
            print(log_message)


    def on_epoch_start(self, trainer):
        epoch = trainer.history["epoch"]
        epochs = trainer.history["epochs"]
        log_message = f"Epoch {epoch}/{epochs}"
        print(log_message)


    def on_prediction_step_end(self, inferencer):
        step = inferencer.history["step"]
        steps = inferencer.history["steps"]
        elapsed = inferencer.history["elapsed"]
        remain = inferencer.history["remain"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            log_message = f"[Prediction] {step}/{steps} - elapsed: {elapsed} - remain: {remain}"
            print(log_message)