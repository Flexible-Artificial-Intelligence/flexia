# Copyright 2022 The Flexia Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from datetime import timedelta


def get_logger(name=__name__, 
               path="logs.log", 
               logs_format="%(message)s", 
               stream_handler=True, 
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
    string = sep.join([f"{metric}: {value:.{decimals}f}" for metric, value in metrics.items()])
    string = sep + string if string != "" else string
    return string


def format_time(time:timedelta, time_format:str="{hours:02d}:{minutes:02d}:{seconds:02d}") -> str:
    """
    Formats `timedelta` to user's time format.
    """
    time = get_time_from_timedelta(time)
    return time_format.format(**time)

def get_time_from_timedelta(delta:timedelta) -> dict:
    time = {"days": delta.days}
    time["hours"], rem = divmod(delta.seconds, 3600)
    time["minutes"], time["seconds"] = divmod(rem, 60)

    return time