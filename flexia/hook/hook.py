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


class Hook:    
    def on_init_start(self, trainer) -> None:
        """
        Called when the Trainer initialization starts.
        """

        pass

    def on_init_end(self, trainer) -> None:
        """
        Called when the Trainer initialization ends.
        """
        
        pass

    def on_training_step_start(self, trainer) -> None:
        """
        Called when the training step starts.
        """
        
        pass

    def on_training_step_end(self, trainer) -> None:
        """
        Called when the training step ends.
        """
        
        pass

    def on_validation_step_start(self, trainer) -> None:
        """
        Called when the validation step starts.
        """
        
        pass

    def on_validation_step_end(self, trainer) -> None:
        """
        Called when the validation step ends.
        """

        pass

    def on_epoch_start(self, trainer) -> None:
        """
        Called when the training epoch starts.
        """
        
        pass

    def on_epoch_end(self, trainer) -> None:
        """
        Called when the training epoch ends.
        """

        pass

    def on_validation_start(self, trainer) -> None:
        """
        Called when the validation loop starts.
        """

        pass

    def on_validation_end(self, trainer) -> None:
        """
        Called when the validation loop ends.
        """

        pass

    def on_validation_step_start(self, trainer) -> None:
        """
        Called when the validation step starts.
        """

        pass

    def on_validation_step_end(self, trainer) -> None:
        """
        Called when the validation step ends.
        """

        pass

    def on_training_start(self, trainer) -> None:
        """
        Called when the training loop starts.
        """

        pass

    def on_training_end(self, trainer) -> None:
        """
        Called when the training loop ends.
        """
        
        pass

    def on_training_stop(self, trainer) -> None:
        """
        Called when the training loop stops.
        """

        self.on_training_end(trainer=trainer)

    def on_checkpoint_save(self, trainer) -> None:
        """
        Called when the checkpoint is saved.
        """

        pass
    
    def on_exception(self, trainer) -> None:
        """
        Called when the exception raises.
        """

        self.on_training_end(trainer=trainer)

    def on_prediction_start(self, inferencer) -> None:
        """
        Called when the prediction loop starts.
        """

        pass

    def on_prediction_end(self, inferencer) -> None: 
        """
        Called when the prediction loop ends.
        """

        pass

    def on_prediction_step_start(self, inferencer) -> None:
        """
        Called when the prediction step starts.
        """

        pass

    def on_prediction_step_end(self, ) -> None:
        """
        Called when the prediction step ends.
        """

        pass