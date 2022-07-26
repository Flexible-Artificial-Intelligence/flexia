from .hook import Hook



class Hooks:
    def __init__(self, hooks=[]):
        self.hooks = hooks

        if isinstance(self.hooks, list):
            for hook in self.hooks:
                if not isinstance(hook, Hook):
                    raise TypeError(f"{hook} must be subclass of {self.__class__.__name__}")
        else:
            raise TypeError("`hooks` must be list")
        

        
    def on_init_start(self, trainer) -> None:
        """
        Called when the Trainer initialization starts.
        """

        for hook in self.hooks:
            hook.on_init_start(trainer)

    def on_init_end(self, trainer) -> None:
        """
        Called when the Trainer initialization ends.
        """
        
        for hook in self.hooks:
            hook.on_init_end(trainer)

    def on_training_step_start(self, trainer) -> None:
        """
        Called when the training step starts.
        """
        
        for hook in self.hooks:
            hook.on_training_step_start(trainer)

    def on_training_step_end(self, trainer) -> None:
        """
        Called when the training step ends.
        """
        
        for hook in self.hooks:
            hook.on_training_step_end(trainer)

    def on_validation_step_start(self, trainer) -> None:
        """
        Called when the validation step starts.
        """
        
        for hook in self.hooks:
            hook.on_validation_step_start(trainer)

    def on_validation_step_end(self, trainer) -> None:
        """
        Called when the validation step ends.
        """

        for hook in self.hooks:
            hook.on_validation_step_end(trainer)

    def on_epoch_start(self, trainer) -> None:
        """
        Called when the training epoch starts.
        """
        
        for hook in self.hooks:
            hook.on_epoch_start(trainer)

    def on_epoch_end(self, trainer) -> None:
        """
        Called when the training epoch ends.
        """

        for hook in self.hooks:
            hook.on_epoch_end(trainer)

    def on_validation_start(self, trainer) -> None:
        """
        Called when the validation loop starts.
        """

        for hook in self.hooks:
            hook.on_validation_start(trainer)

    def on_validation_end(self, trainer) -> None:
        """
        Called when the validation loop ends.
        """

        for hook in self.hooks:
            hook.on_validation_end(trainer)

    def on_validation_step_start(self, trainer) -> None:
        """
        Called when the validation step starts.
        """

        for hook in self.hooks:
            hook.on_validation_step_start(trainer)

    def on_validation_step_end(self, trainer) -> None:
        """
        Called when the validation step ends.
        """

        for hook in self.hooks:
            hook.on_validation_step_end(trainer)

    def on_training_start(self, trainer) -> None:
        """
        Called when the training loop starts.
        """

        for hook in self.hooks:
            hook.on_training_start(trainer)

    def on_training_end(self, trainer) -> None:
        """
        Called when the training loop ends.
        """
        
        for hook in self.hooks:
            hook.on_training_end(trainer)

    def on_training_stop(self, trainer) -> None:
        """
        Called when the training loop stops.
        """

        self.on_training_end(trainer=trainer)

    def on_checkpoint_save(self, trainer) -> None:
        """
        Called when the checkpoint is saved.
        """

        for hook in self.hooks:
            hook.on_checkpoint_save(trainer)
    
    def on_exception(self, trainer) -> None:
        """
        Called when the exception raises.
        """

        self.on_training_end(trainer=trainer)

    def on_prediction_start(self, trainer) -> None:
        """
        Called when the prediction loop starts.
        """

        for hook in self.hooks:
            hook.on_prediction_start(trainer)

    def on_prediction_end(self, trainer) -> None: 
        """
        Called when the prediction loop ends.
        """

        for hook in self.hooks:
            hook.on_prediction_end(trainer)

    def on_prediction_step_start(self, trainer) -> None:
        """
        Called when the prediction step starts.
        """

        for hook in self.hooks:
            hook.on_prediction_step_start(trainer)


    def on_prediction_step_end(self, trainer) -> None:
        """
        Called when the prediction step ends.
        """

        for hook in self.hooks:
            hook.on_prediction_step_end(trainer)