from .logger import Logger
from ..third_party.pkbar import Kbar, Pbar


class KerasLogger(Logger):
    def __init__(self, 
                 width=20, 
                 verbose=1, 
                 interval=0.05, 
                 stateful_metrics=None, 
                 always_stateful=False, 
                 unit_name="step", 
                 log_accelerator_stats=False):

        super().__init__()

        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.always_stateful = always_stateful
        self.stateful_metrics = stateful_metrics
        self.unit_name = unit_name
        self.log_accelerator_stats = log_accelerator_stats
        self.bar = None

    def on_epoch_start(self, trainer) -> None:
        steps_epoch = trainer.history["steps_epoch"]
        epoch = trainer.history["epoch"] - 1
        epochs = trainer.history["epochs"]

        self.bar = Kbar(target=steps_epoch, 
                        epoch=epoch, 
                        num_epochs=epochs, 
                        width=self.width, 
                        verbose=self.verbose, 
                        interval=self.interval, 
                        stateful_metrics=self.stateful_metrics, 
                        always_stateful=self.always_stateful,
                        unit_name=self.unit_name)


    def on_training_step_end(self, trainer) -> None:
        step = trainer.history["step_epoch"]
        train_loss_epoch = trainer.history["train_loss_epoch"]
        train_metrics_epoch = trainer.history["train_metrics_epoch"]
        train_metrics_epoch = [(k, v) for k, v in train_metrics_epoch.items()]

        values = [("loss", train_loss_epoch)] + train_metrics_epoch
        self.bar.update(step, values=values)


    def on_validation_end(self, trainer) -> None:
        validation_loss = trainer.history["validation_loss"]
        validation_metrics = trainer.history["validation_metrics"]
        validation_metrics = [(f"val_{k}", v) for k, v in validation_metrics.items()]

        values = [("val_loss", validation_loss)] + validation_metrics
        self.bar.add(1, values=values)

        
    def on_prediction_start(self, trainer) -> None:
        prediction_steps = trainer.history["prediction_steps"]
        self.prediction_bar = Pbar(name="Prediction", target=prediction_steps, width=self.width)


    def on_prediction_step_end(self, trainer) -> None:
        prediction_step = trainer.history["prediction_step"]
        self.prediction_bar.update(prediction_step)
