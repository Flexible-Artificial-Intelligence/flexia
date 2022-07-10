import logging


def get_logger(name=__name__, 
               path="logs.log", 
               logs_format="%(message)s", 
               stream_handler=False, 
               level=logging.INFO) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(logs_format)

    if path is not None:
        file_handler = logging.FileHandler(name)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        

    return logger


def format_metrics(metrics, sep=" - ", decimals=4) -> str:
    string = sep.join([f"{metric}: {value:.{decimals}}" for metric, value in metrics.items()])
    string = " - " + string if string != "" else string
    return string