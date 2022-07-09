from typing import Union
from datetime import timedelta


from .callback import Callback
from ..third_party.pytimeparse.timeparse import timeparse
from ..trainer.trainer_enums import TrainingStates


class Timing(Callback):
    def __init__(self, 
                 duration:Union[str, timedelta]="01:00:00:00", 
                 duration_separator:str=":"):
        
        super().__init__()
        
        self.duration = duration
        self.duration_separator = duration_separator

        if isinstance(self.duration, str):
            try:
                duration_values = self.duration.strip().split(self.duration_separator)
                duration_values = tuple([int(value) for value in duration_values])
                days, hours, minutes, seconds = duration_values 
            except:
                seconds = timeparse(self.duration)
                days, hours, minutes = 0, 0, 0
            
            self.duration = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

        elif isinstance(self.duration, dict):
            self.duration = timedelta(**self.duration)
        elif not isinstance(self.duration, timedelta):
            raise TypeError(f"Type of given `duration` is not supported.")
            
        self.stop = False
        

    def on_epoch_end(self, trainer):
        elapsed = trainer.history["elapsed"]
        self.stop = self.check(elapsed)
        
        if self.stop:
            trainer.state = TrainingStates.TRAINING_STOP
        
    def check(self, elapsed) -> bool:       
        stop = elapsed > self.duration
        
        return stop