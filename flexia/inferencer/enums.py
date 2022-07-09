from ..enums import ExplicitEnum


class InferencerStates(ExplicitEnum):
    INIT = "on_init"
    PREDICTION_START = "on_prediction_start"
    PREDICTION_END = "on_prediction_end"
    PREDICTION_STEP_START = "on_prediction_step_start"
    PREDICTION_STEP_END = "on_prediction_step_end"