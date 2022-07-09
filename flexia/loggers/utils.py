import logging


def get_logger(name="logger", path="logs.log", logs_format="[%(asctime)s][%(levelname)s]: %(message)s") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(logs_format)

    if path is not None:
        file_handler = logging.FileHandler(name)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
        
    logger.propagate = False
        
    return logger


def format_metrics(metrics, sep=" - ", decimals=4) -> str:
    string = sep.join([f"{metric}: {value:.{decimals}}" for metric, value in metrics.items()])
    string = " - " + string if string != "" else string
    return string