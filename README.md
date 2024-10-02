# ASPCOL : Audio Signal Processing COLlection
ASPCOL is a collection of functions and classes for audio signal processing. The package contains routines for state-of-the-art sound field estimation and sound field reproduction methods. It also contains functions for classic audio signal processing such as designing and using filters, adaptive filtering, statistical signal processing, matrix manipulations, and more. 

**[More info and complete API documentation](https://sounds-research.github.io/aspcol/)**

## Installation
The package does not exist on PyPi. It can be installed by cloning the repository and installing via pip from the downloaded folder. 
```
pip install ./path/to/cloned/aspcol/folder
```

All obligatory dependencies are listed in requirements.txt, and can be installed with pip:
```
pip install -r requirements.txt
```
The only non-standard dependency is [aspcore](https://github.com/SOUNDS-RESEARCH/aspcore) which is optional. It is required for all functionality of the adaptivefilter.py module, as well as the function power_of_filtered_signal in utilities.py. 

## Contents
### Sound field estimation
The content is contained in the modules 
- kernelinterpolation
- movingmicrophones
- planewaves
- soundfieldestimation
- sphericalharmonics

### Sound field reproduction
The content is contained in the modules
- soundfieldcontrol

### Linear filtering
The content is contained in the modules
- adaptivefilter
- filterclasses
- filterdesign
- fouriertransform
- lowrank


## References
The package was developed in the course of the following research. Please consider citing any of the following papers if relevant to your work. 

**Bayesian sound field estimation using uncertain data**, J. Brunnström, M. B. Møller, J. Østergaard, and M. Moonen
```
@inproceedings{brunnstromBayesian2024,
    title = {Bayesian Sound Field Estimation Using Uncertain Data},
    author = {Brunnstr{\"o}m, Jesper and M{\o}ller, Martin Bo and {\O}stergaard, Jan and Moonen, Marc},
    year = {2024},
    month = sep,
    langid = {english},
    booktitle = {Proc. Int. Workshop Acoust. Signal Enhancement (IWAENC).},
}
```

**Sound zone control for arbitrary sound field reproduction methods**, J. Brunnström, T. van Waterschoot, and M. Moonen
```
@inproceedings{brunnstromSound2023,
    title = {Sound Zone Control for Arbitrary Sound Field Reproduction Methods},
    author = {Brunnstr{\"o}m, Jesper and van Waterschoot, Toon and Moonen, Marc},
    year = {2023},
    month = sep,
    doi = {10.23919/EUSIPCO58844.2023.10289995},
    booktitle = {Proc. European Signal Process. Conf. (EUSIPCO),},
}
```

**Signal-to-interference-plus-noise ratio based optimization for sound zone control**, J. Brunnström, T. van Waterschoot, and M. Moonen
```
@article{brunnstromSignaltointerferenceplusnoise2023,
    title = {Signal-to-Interference-plus-Noise Ratio Based Optimization for Sound Zone Control},
    author = {Brunnstr{\"o}m, Jesper and {van Waterschoot}, Toon and Moonen, Marc},
    year = {2023},
    journal = {IEEE Open J. Signal Process.},
    volume = {4},
    pages = {257--266},
    issn = {2644-1322},
    doi = {10.1109/OJSP.2023.3246398},
}
```

**Variable span trade-off filter for sound zone control with kernel interpolation weighting**, J. Brunnström, S. Koyama, and M. Moonen
```
@inproceedings{brunnstromVariable2022,
    title = {Variable Span Trade-off Filter for Sound Zone Control with Kernel Interpolation Weighting},
    author = {Brunnstr{\"o}m, Jesper and Koyama, Shoichi and Moonen, Marc},
    year = {2022},
    month = may,
    pages = {1071--1075},
    issn = {2379-190X},
    doi = {10.1109/ICASSP43922.2022.9746550},
    booktitle = {Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)},
}
```

## License
The software is distributed under the MIT license. See the LICENSE file for more information.

## Acknowledgements
The software has been developed during a PhD project as part of the [SOUNDS ETN](https://www.sounds-etn.eu) at KU Leuven. The SOUNDS project has recieved funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 956369.