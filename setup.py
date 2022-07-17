from setuptools import setup, find_packages


install_requires = ["tqdm", "numpy", "wandb", "torch", "transformers"]

setup(
    name="flexia",
    version="1.0.3",
    description="Flexia (Flexible Artificial Intelligence) is an open-source library that provides a high-level API for developing accurate Deep Learning models for all kinds of Deep Learning tasks such as Classification, Regression, Object Detection, Image Segmentation, etc.",
    author="Vadim Irtlach",
    author_email="vadimirtlach@gmail.com",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/Flexible-Artificial-Intelligence/flexia",
    keywords="python data-science machine-learning natural-language-processing library computer-vision deep-learning pytorch artificial-intelligence neural-networks",
    python_requires=">=3.7.0",
    install_requires=install_requires,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)