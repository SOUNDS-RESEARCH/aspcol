Audio Signal Processing Collection (aspcol)
===========================================
This is a package for audio signal processing. Contains implementations of state-of-the-art sound field estimation and sound field reproduction methods, as well as some basic audio signal processing tools.

API Reference
=============
.. autosummary::
   :toctree: _autosummary
   :template: new-module-template.rst

   aspcol.distance
   aspcol.kernelinterpolation
   aspcol.kernelinterpolation_jax
   aspcol.planewaves
   aspcol.plot
   aspcol.soundfieldcontrol
   aspcol.soundfieldestimation
   aspcol.soundfieldestimation_jax
   aspcol.spatialcovarianceestimation
   aspcol.sphericalharmonics
   aspcol.utilities

References
==========
A full list of the papers relevant to the implemented algorithms in this package can be found at the following link. 

.. toctree::

   references

The package was developed in the course of the following research. Please consider citing any of the following papers if relevant to your work. 

*Bayesian sound field estimation using uncertain data*, J. Brunnström, M. B. Møller, J. Østergaard, and M. Moonen

.. code-block:: bibtex

   @inproceedings{brunnstromBayesian2024,
      title = {Bayesian Sound Field Estimation Using Uncertain Data},
      author = {Brunnstr{\"o}m, Jesper and M{\o}ller, Martin Bo and {\O}stergaard, Jan and Moonen, Marc},
      year = {2024},
      month = sep,
      langid = {english},
      booktitle = {Proc. Int. Workshop Acoust. Signal Enhancement (IWAENC).},
   }

*Sound zone control for arbitrary sound field reproduction methods*, J. Brunnström, T. van Waterschoot, and M. Moonen

.. code-block:: bibtex

   @inproceedings{brunnstromSound2023,
      title = {Sound Zone Control for Arbitrary Sound Field Reproduction Methods},
      author = {Brunnstr{\"o}m, Jesper and van Waterschoot, Toon and Moonen, Marc},
      year = {2023},
      month = sep,
      doi = {10.23919/EUSIPCO58844.2023.10289995},
      booktitle = {Proc. European Signal Process. Conf. (EUSIPCO),},
   }

*Signal-to-interference-plus-noise ratio based optimization for sound zone control*, J. Brunnström, T. van Waterschoot, and M. Moonen

.. code-block:: bibtex

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

*Variable span trade-off filter for sound zone control with kernel interpolation weighting*, J. Brunnström, S. Koyama, and M. Moonen

.. code-block:: bibtex

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






License
=======
The software is distributed under the MIT license. See the LICENSE file for more information.

If you use this software in your research, please cite the relevant paper(s). Most functions has a reference to the paper in which the method is described.

Acknowledgements
================
The software has been developed during a PhD project as part of the `SOUNDS <https://www.sounds-etn.eu/>`_ ETN project at KU Leuven. The SOUNDS project has recieved funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 956369.

