def run_hook(hook, trainer, *args, **kwargs) -> None:
    method = getattr(hook, trainer.state.value)
    method(trainer, *args, **kwargs)


def run_hooks(hooks, trainer, *args, **kwargs) -> None:
    for hook in hooks:
        run_hook(hook=hook, trainer=trainer, *args, **kwargs)