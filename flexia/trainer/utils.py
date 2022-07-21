from .enums import TrainerState


def exception_handler(function):
    def handler(*args, **kwargs):
        instance = args[0]
        try:
            output = function(*args, **kwargs)
            return output
        except Exception as error:
            instance.state = TrainerState.EXCEPTION
            raise error

    return handler