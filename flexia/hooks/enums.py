from ..enums import EqualEnum


class State(EqualEnum):
    INIT_START = "on_init_start"
    INIT_END = "on_init_end"
    TRAINING_START = "on_training_start"
    TRAINING_END = "on_training_end"
    TRAINING_STEP_START = "on_training_step_start"
    TRAINING_STEP_END = "on_training_step_end"
    VALIDATION_START = "on_validation_start"
    VALIDATION_END = "on_validation_end"
    EPOCH_START = "on_epoch_start"
    VALIDATION_STEP_START = "on_validation_step_start"
    VALIDATION_STEP_END = "on_validation_step_end"
    EPOCH_END = "on_epoch_end"
    TRAINING_STOP = "on_training_stop"
    CHECKPOINT_SAVE = "on_checkpoint_save"
    EXCEPTION = "on_exception"
    PREDICTION_START = "on_prediction_start"
    PREDICTION_END = "on_prediction_end"
    PREDICTION_STEP_START = "on_prediction_step_start"
    PREDICTION_STEP_END = "on_prediction_step_end"