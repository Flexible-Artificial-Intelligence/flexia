from .enums import InferencerStates


def exception_handler(function):
    def handler(*args, **kwargs):
        instance = args[0]
        try:
            output = function(*args, **kwargs)
        except Exception as error:
            instance.state = InferencerStates.EXCEPTION
            return error

    return handler