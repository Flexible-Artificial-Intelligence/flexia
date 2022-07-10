from . import Trainer, Inferencer


class Logger:    
    def on_init(self, trainer:"Trainer") -> None:
        """
        Called when the Trainer initialization ends.
        """

        pass

    def on_training_step_start(self, trainer:"Trainer") -> None:
        """
        Called when the training step starts.
        """
        
        pass

    def on_training_step_end(self, trainer:"Trainer") -> None:
        """
        Called when the training step ends.
        """
        
        pass

    def on_validation_step_start(self, trainer:"Trainer") -> None:
        """
        Called when the validation step starts.
        """
        
        pass

    def on_validation_step_end(self, trainer:"Trainer") -> None:
        """
        Called when the validation step ends.
        """

        pass

    def on_epoch_start(self, trainer:"Trainer") -> None:
        """
        Called when the training epoch starts.
        """
        
        pass

    def on_epoch_end(self, trainer:"Trainer") -> None:
        """
        Called when the training epoch ends.
        """

        pass

    def on_validation_start(self, trainer:"Trainer") -> None:
        """
        Called when the validation loop starts.
        """

        pass

    def on_validation_end(self, trainer:"Trainer") -> None:
        """
        Called when the validation loop ends.
        """

        pass

    def on_validation_step_start(self, trainer:"Trainer") -> None:
        """
        Called when the validation step starts.
        """

        pass

    def on_validation_step_end(self, trainer:"Trainer") -> None:
        """
        Called when the validation step ends.
        """

        pass

    def on_training_start(self, trainer:"Trainer") -> None:
        """
        Called when the training loop starts.
        """

        pass

    def on_training_end(self, trainer:"Trainer") -> None:
        """
        Called when the training loop ends.
        """
        
        pass

    def on_training_stop(self, trainer:"Trainer") -> None:
        """
        Called when the training loop stops.
        """

        pass

    def on_checkpoint_save(self, trainer:"Trainer") -> None:
        """
        Called when the checkpoint is saved.
        """

        pass
    
    def on_exception(self, exception:Exception, trainer:"Trainer") -> None:
        """
        Called when the exception raises.
        """

        pass

    def on_prediction_start(self, inferencer:"Inferencer") -> None:
        """
        Called when the prediction loop starts.
        """

        pass

    def on_prediction_end(self, inferencer:"Inferencer") -> None: 
        """
        Called when the prediction loop ends.
        """

        pass

    def on_prediction_step_start(self, inferencer:"Inferencer") -> None:
        """
        Called when the prediction step starts.
        """

        pass

    def on_prediction_step_end(self, inferencer:"Inferencer") -> None:
        """
        Called when the prediction step ends.
        """

        pass