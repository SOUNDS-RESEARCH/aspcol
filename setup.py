import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aspcol-PVJesper",
    version="0.0.1",
    author="Jesper Brunnström",
    author_email="jesper.brunnstroem@kuleuven.be",
    description="Audio signal processing collection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
