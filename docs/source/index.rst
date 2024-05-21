.. aspcol documentation master file, created by
   sphinx-quickstart on Wed Sep 13 10:22:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Audio Signal Processing Collection (aspcol)
===========================================
This is a package for audio signal processing. Contains implementations of state-of-the-art sound field estimation and sound field reproduction methods, as well as some basic audio signal processing tools.

API Reference
=============
.. autosummary::
   :toctree: _autosummary
   :template: new-module-template.rst

   aspcol.adaptivefilter
   aspcol.correlation
   aspcol.distance
   aspcol.filterclasses
   aspcol.filterdesign
   aspcol.kernelinterpolation
   aspcol.lowrank
   aspcol.matrices
   aspcol.montecarlo
   aspcol.pseq
   aspcol.soundfieldcontrol
   aspcol.soundfieldestimation
   aspcol.utilities


License
=======
The software is distributed under the MIT license. See the LICENSE file for more information.

If you use this software in your research, please cite the relevant paper(s). Most functions has a reference to the paper in which the method is described.

Acknowledgements
================
The software has been developed during a PhD project as part of the `SOUNDS <https://www.sounds-etn.eu/>`_ ETN project at KU Leuven. The SOUNDS project has recieved funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 956369.

