import os
from setuptools import setup, find_packages


NAME = "flexia"
DESCRIPTION = """Flexia (Flexible Artificial Intelligence) is an open-source library, which provides high-level functionality for developing accurate Deep Learning models. There is a variety of methods for controlling (e.g Early Stopping, Model Checkpointing, Timing, etc.) and monitoring (e.g Weights & Biases, Print, Logging, and Taqadum) training, validation, and inferencing loops respectively."""
VERSION = "1.0.0"
AUTHOR = "Vadim Irtlach"
AUTHOR_EMAIL = "vadimirtlach@gmail.com"
URL = "https://github.com/vad13irt/flexia"
REQUIRED = ["pytimeparse", "numpy", "torch"]
PACKAGES = find_packages("flexia", "flexia/*")

DIRECTORY = os.getcwd()
README_PATH = os.path.join(DIRECTORY, "README.md")

with open(README_PATH, "r", encoding="utf-8") as readme:
    README = str(readme.read())

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=PACKAGES,
    include_package_data=True,
    install_requires=REQUIRED,
)