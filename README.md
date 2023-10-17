# ASPCOL : Audio Signal Processing COLlection

## Introduction
A collection of functions and classes that can be useful for audio signal processing. More info can be found in the [documentation](https://sounds-research.github.io/aspcol/)

## Dependencies
The only non-standard dependency is [aspcore](https://github.com/SOUNDS-RESEARCH/aspcore). The other dependencies are listed in requirements.txt, and can be installed with pip:
```
pip install -r requirements.txt
```


# Modules
## adaptivefilter
* Least mean squares (LMS)
* Recursive least squares (RLS)
* Fast block LMS
* Fast block weighted LMS

[P. S. R. Diniz, Adaptive filtering: algorithms and practical implementation. Cham: Springer International Publishing, 2020. doi: 10.1007/978-3-030-29057-3.](https://link.springer.com/book/10.1007/978-3-030-29057-3)  

## correlation
* Estimation of correlation function and covariance matrix of signal.  
* Sample covariance for vector-valued variable, including the non-zero mean case, even in a streaming processing.  
* Covariance estimation with optimal linear shrinkage [1] and almost optimal non-linear shrinkage [2],

[1] [Y. Chen, A. Wiesel, Y. C. Eldar, and A. O. Hero, “Shrinkage Algorithms for MMSE Covariance Estimation,” IEEE Trans. Signal Process., vol. 58, no. 10, pp. 5016–5029, Oct. 2010, doi: 10.1109/TSP.2010.2053029.](doi.org/10.1109/TSP.2010.2053029)  
[2] [O. Ledoit and M. Wolf, “Quadratic shrinkage for large covariance matrices,” Dec. 2020, doi: 10.5167/UZH-176887.](doi.org/10.5167/UZH-176887)  

## distance
A collection of distance measures for different types of quantities

##### Any array:  
* Mean square error
##### Vectors:   
* Angular distance
* Cosine similarity 
##### PSD matrices:  
* Correlation matrix distance [1]
* Affine invariant Riemannian metric [2]
* Kullback Leibler divergence between zero-mean Gaussian densities described by the compared matrices [3]


[1] [M. Herdin, N. Czink, H. Ozcelik, and E. Bonek, “Correlation matrix distance, a meaningful measure for evaluation of non-stationary MIMO channels,” in 2005 IEEE 61st Vehicular Technology Conference, May 2005, pp. 136-140 Vol. 1. doi: 10.1109/VETECS.2005.1543265.](https://doi.org/10.1109/VETECS.2005.1543265)  
[2] [W. Förstner and B. Moonen, “A metric for covariance matrices,” in Geodesy-The Challenge of the 3rd Millennium, E. W. Grafarend, F. W. Krumm, and V. S. Schwarze, Eds., Berlin, Heidelberg: Springer Berlin Heidelberg, 2003, pp. 299–309. doi: 10.1007/978-3-662-05296-9_31.](doi.org/10.1007/978-3-662-05296-9_31)  
[3] [J. Duchi, "Derivations for Linear Algebra and Optimization"](https://web.stanford.edu/~jduchi/projects/general_notes.pdf)  

## filterclasses
* Weighted overlap-add (WOLA) [1,2]
* IIR filter
* Mean with forgetting factor

[1] [R. Crochiere, “A weighted overlap-add method of short-time Fourier analysis/synthesis,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 28, no. 1, pp. 99–102, Feb. 1980, doi: 10.1109/TASSP.1980.1163353.](doi.org/10.1109/TASSP.1980.1163353)  
[2] [S. Ruiz, T. Dietzen, T. van Waterschoot, and M. Moonen, “A comparison between overlap-save and weighted overlap-add filter banks for multi-channel Wiener filter based noise reduction,” in 2021 29th European Signal Processing Conference (EUSIPCO), Aug. 2021, pp. 336–340. doi: 10.23919/EUSIPCO54536.2021.9616352.](doi.org/10.23919/EUSIPCO54536.2021.9616352)  


## filterdesign
Helper functions to obtain FIR filters from frequency values, for less risk of errors. 


## kernelinterpolation
Interpolation of a sound field taking physical properties of sound into account. [1, 2]

A number of kernels are implemented:
* Gaussian kernel
* Diffuse kernel in 2D [6] and 3D [1]
* Directional kernel in 3D [2, 4]
* Reciprocal kernel for RIR estimation [3]

The estimation methods are written generally to allow for any kernel function to be given as argument, and then functions implementing the kernel functions associated with the papers below are provided. 

[1] [N. Ueno, S. Koyama, and H. Saruwatari, “Kernel ridge regression with constraint of Helmholtz equation for sound field interpolation,” in 2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC), Tokyo, Japan: IEEE, Sep. 2018, pp. 436–440. doi: 10.1109/IWAENC.2018.8521334.](doi.org/10.1109/IWAENC.2018.8521334)  
[2] [N. Ueno, S. Koyama, and H. Saruwatari, “Directionally weighted wave field estimation exploiting prior information on source direction,” IEEE Transactions on Signal Processing, vol. 69, pp. 2383–2395, Apr. 2021, doi: 10.1109/TSP.2021.3070228.](doi.org/10.1109/TSP.2021.3070228)  
[3] [J. G. C. Ribeiro, N. Ueno, S. Koyama, and H. Saruwatari, “Kernel interpolation of acoustic transfer function between regions considering reciprocity,” in 2020 IEEE 11th Sensor Array and Multichannel Signal Processing Workshop (SAM), Jun. 2020, pp. 1–5. doi: 10.1109/SAM48682.2020.9104256.](doi.org/10.1109/SAM48682.2020.9104256)  
[4] [S. Koyama, J. Brunnström, H. Ito, N. Ueno, and H. Saruwatari, “Spatial active noise control based on kernel interpolation of sound field,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3052–3063, Aug. 2021, doi: 10.1109/TASLP.2021.3107983.](doi.org/10.1109/TASLP.2021.3107983)  
[5] [J. Brunnström, S. Koyama, and M. Moonen, “Variable span trade-off filter for sound zone control with kernel interpolation weighting,” in ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), May 2022, pp. 1071–1075. doi: 10.1109/ICASSP43922.2022.9746550.](doi.org/10.1109/ICASSP43922.2022.9746550)  
[6] [H. Ito, S. Koyama, N. Ueno, and H. Saruwatari, “Feedforward spatial active noise control based on kernel interpolation of sound field,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, United Kingdom: IEEE, May 2019, pp. 511–515. doi: 10.1109/ICASSP.2019.8683067.](doi.org/10.1109/ICASSP.2019.8683067)


## lowrank
* Decomposes and reconstructs any impulse response with singular value decomposition or canonical polyadic decomposition for a low-rank approximation [1,2]
* Implements a low-cost convolution by directly using the low-rank representation [3,4]

[1] [M. Jälmby, F. Elvander, and T. van Waterschoot, “Low-rank tensor modeling of room impulse responses,” in 2021 29th European Signal Processing Conference (EUSIPCO), Aug. 2021, pp. 111–115. doi: 10.23919/EUSIPCO54536.2021.9616075.](doi.org/10.23919/EUSIPCO54536.2021.9616075)  
[2] [C. Paleologu, J. Benesty, and S. Ciochină, “Linear system identification based on a Kronecker product decomposition,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 10, pp. 1793–1808, Oct. 2018, doi: 10.1109/TASLP.2018.2842146.](doi.org/10.1109/TASLP.2018.2842146)  
[3] [J. Atkins, A. Strauss, and C. Zhang, “Approximate convolution using partitioned truncated singular value decomposition filtering,” in 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, May 2013, pp. 176–180. doi: 10.1109/ICASSP.2013.6637632.](doi.org/10.1109/ICASSP.2013.6637632)  
[4] [M. Jälmby, F. Elvander, and T. van Waterschoot, “Fast low-latency convolution by low-rank tensor approximation,” in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes, Greece, Jun. 2023.](doi.org/10.1109/ICASSP49357.2023.10095908)  


## matrices
Helper functions for dealing with matrices. Some examples include constructing a block matrix, ensure positive definiteness, apply function to individual blocks of a block matrix. 

## montecarlo
Helper functions for naive monte carlo sampling of a given function

## soundfieldestimation
Directionally weighted wave field estimation. Can be used for directional microphones

Relevant papers
Ueno, Koyama, Saruwatari - Directionally weighted wave field estimation exploiting prior information on source direction
### utilities
Helper functions for various tasks
