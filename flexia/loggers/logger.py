class Logger:    
    def on_init(self, trainer):
        pass

    def on_training_step_start(self, trainer):
        pass

    def on_training_step_end(self, trainer):
        pass

    def on_validation_step_start(self, trainer):
        pass

    def on_validation_step_end(self, trainer):
        pass

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_validation_start(self, trainer):
        pass

    def on_validation_end(self, trainer):
        pass

    def on_validation_step_start(self, trainer):
        pass

    def on_validation_step_end(self, trainer):
        pass

    def on_training_start(self, trainer):
        pass

    def on_training_end(self, trainer):
        pass

    def on_training_stop(self, trainer):
        pass

    def on_checkpoint_save(self, trainer):
        pass

    def on_exception(self, exception, trainer):
        pass

    def on_prediction_start(self, inferencer):
        pass

    def on_prediction_end(self, inferencer): 
        pass

    def on_prediction_step_start(self, inferencer):
        pass

    def on_prediction_step_end(self, inferencer):
        pass