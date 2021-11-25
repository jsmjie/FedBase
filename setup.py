import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fedbase",
    version="0.1.1",
    author="Jie MA",
    author_email="ustcmj@gmail.com, jie.ma-5@student.uts.edu.au",
    description="An easy, accessible Federated Learning framework for individual researchers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jie-ma-ai/FedBase",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)