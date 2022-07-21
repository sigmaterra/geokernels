# geokernels: Geodesic kernels for Gaussian Process regression and classification 

This package is an extension of scikit-learn's gaussian_process kernel package
sklearn/gaussian_process/kernels.py (see 
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process)

The following kernels are added to the default kernels as geodesic kernel versions:

- 'RBF_geo' (RBF kernel with geodesic distance metric)
- 'Matern_geo' (Matern kernel with geodesic distance metric)
- 'RationalQuadratic_geo' (Rational Quadratic kernel with geodesic distance metric)

All kernels can be used as drop-in replacement for the scikit-learn kernels.

The geodesic kernels are build for the use-case that the spatial dimensions of a dataset are given
as Latitude and Longitude (WGS84 coordinate system). This avoids the problem of having to transform 
the data coordinates into a local projected cartesian system beforehand, which can lead to inaccuracies
at larger distances (e.g., problem of overlapping coordinate reference zones). 

Solving the geodesic problem is accomplished by defining a kernel that combines the 
geodesic distance metric for the spatial part with the Euclidean distance metric
for all non-spatial features (e.g., for spatial-temporal modeling).

The geodesic distance is computed via Vincenty's solution to the inverse geodetic problem, 
which is based on the WGS84 reference ellipsoid and is accurate to within 1 mm or better.
For details, please see references or documentation in geokernels.geodisics.py.

Both, anisotropic (one length-scale per feature) and isotropic (same length-scale for all features) kernels 
are supported. Note that given the non-euclidean metric for the geodesic distance, only one length-scale 
parameter is required instead of two separate for the two spatial dimensions (Latitude, Longitude) of the dataset. 
Thus, for an anisotropic kernel, the number of length-scales is one less than the number of dimensions of the data. 


## Installation

``` sh
python -m pip install geokernels
```

## Requirements

- scikit-learn
- numba

## Example

``` python

```

## References

- Carl Edward Rasmussen, Christopher K. I. Williams (2006). “Gaussian Processes for Machine Learning”. The MIT Press.
- David Duvenaud (2014). “The Kernel Cookbook: Advice on Covariance functions”.
- Vincenty, Thaddeus (August 1975b). Geodetic inverse solution between antipodal points. DMAAC Geodetic Survey Squadron. doi:10.5281/zenodo.32999.
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process
- https://en.wikipedia.org/wiki/Vincenty's_formulae


## Contributors

The code was written by:

Sebastian Haan (contact: sebhaan@sigmaterra.com)

