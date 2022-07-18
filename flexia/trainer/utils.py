from .enums import TrainerStates


def exception_handler(function):
    def handler(*args, **kwargs):
        instance = args[0]
        try:
            output = function(*args, **kwargs)
            return output
        except Exception as error:
            instance.state = TrainerStates.EXCEPTION
            instance.history["exception"] = error
            return instance.history

    return handler