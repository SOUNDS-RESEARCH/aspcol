# ASPCOL : Audio Signal Processing COLlection

## Introduction
A collection of functions and classes that can be useful for audio signal processing. The code is written generally to facilitate audio processing in a streaming context, where only a block of the signal is available at any given time. 


## Dependencies
The package makes heavy use of numpy and scipy. 


## List of algorithms
# adaptivefilter
Least mean squares (LMS), Recursive least squares (RLS), Fast block LMS, Fast block weighted LMS
# correlation
Estimation of correlation function and covariance matrix of signal. Sample covariance for vector-valued variable, including the non-zero mean case, even in a streaming processing. Covariance estimation with optimal linear shrinkage, covariance estimation with optimal non-linear shrinkage

Relevant papers
Shrinkage Algorithms for MMSE Covariance Estimation
Ledoit, Wolf, "Quadratic shrinkage for large covariance matrices"
# distance
Distance measures for any variable: Mean square error
Distance measures for vectors: Angular distance, Cosine similarity, 
Distance measures for matrices: Correlation matrix distance, Affine invariant Riemannian metric, Kullback Leibler divergence

Relevant papers
Correlation matrix distaince, a meaningful measure for evaluation of non-stationary MIMO channels - Herdin, Czink, Ozcelik, Bonek
A Metric for Covariance Matrices - Wolfgang Förstner, Boudewijn Moonen
# filterclasses
Weighted overlap-add, IIR filter, mean with forgetting factor, 
# filterdesign
Helper functions to obtain FIR filters from frequency values, for less risk of errors. 
# kernelinterpolation
Kernel interpolation of sound field

Relevant papers
N. Ueno, S. Koyama, and H. Saruwatari, “Kernel ridge regression with constraint of Helmholtz equation for sound field interpolation,”
N. Ueno, S. Koyama, and H. Saruwatari, “Directionally weighted wave field estimation exploiting prior information on source direction,” 
J. G. C. Ribeiro, N. Ueno, S. Koyama, and H. Saruwatari, “Kernel interpolation of acoustic transfer function between regions considering reciprocity,”
S. Koyama, J. Brunnström, H. Ito, N. Ueno, and H. Saruwatari, “Spatial active noise control based on kernel interpolation of sound field,”
J. Brunnström, S. Koyama, and M. Moonen, “Variable span trade-off filter for sound zone control with kernel interpolation weighting,”

# matrices
Helper functions for dealing with matrices. Some examples include constructing a block matrix, ensure positive definiteness, apply function to individual blocks of a block matrix. 
# montecarlo
Functions for naive monte carlo sampling of any given function
# polynomialmatrix
Implements Polynomial Eigenvalue Decomposition (PEVD)

Relevant papers
# soundfieldestimation
Directionally weighted wave field estimation. Can be used for directional microphones

Relevant papers
Ueno, Koyama, Saruwatari - Directionally weighted wave field estimation exploiting prior information on source direction
# utilities
Helper functions for anything