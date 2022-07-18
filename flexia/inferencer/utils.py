from .enums import InferencerStates


def exception_handler(function):
    def handler(*args, **kwargs):
        instance = args[0]
        try:
            output = function(*args, **kwargs)
            return output
        except Exception as error:
            instance.state = InferencerStates.EXCEPTION
            instance.history["exception"] = error
            return instance.history

    return handler