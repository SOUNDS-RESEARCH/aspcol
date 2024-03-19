# ASPCOL : Audio Signal Processing COLlection

## Introduction
A collection of functions and classes that can be useful for audio signal processing. More info can be found in the [documentation](https://sounds-research.github.io/aspcol/)

## Dependencies
All obligatory dependencies are listed in requirements.txt, and can be installed with pip:
```
pip install -r requirements.txt
```
The only non-standard dependency is [aspcore](https://github.com/SOUNDS-RESEARCH/aspcore) which is optional. It is required for 
all functionality of the adaptivefilter.py module, as well as the function power_of_filtered_signal in utilities.py. 