from .enums import State


def run_hook(hook, trainer, *args, **kwargs) -> None:
    method = getattr(hook, trainer.state.value)
    method(trainer, *args, **kwargs)


def run_hooks(hooks, trainer, *args, **kwargs) -> None:
    for hook in hooks:
        run_hook(hook=hook, trainer=trainer, *args, **kwargs)


def exception_handler(function):
    def handler(*args, **kwargs):
        instance = args[0]
        try:
            output = function(*args, **kwargs)
            return output
        except Exception as error:
            instance.state = State.EXCEPTION
            raise error

    return handler