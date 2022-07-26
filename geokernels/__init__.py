"""geokernels: fast geospatial distance and geodesic kernel computation for machine learning  

This Python package provides fast geospatial distance computation and geodesic distance 
kernels to accelerate geospatial machine learning and distance matrix calculations.

The included geodesic kernel package accepts WGS84 coordinates (Latitude, Longitude) and 
extends scikit-learn's Gaussian Process kernels with geodesic kernels as drop-in replacement. 
This solves the problem of continental scale modeling and requires no transformation into 
suitable local projected coordinate systems beforehand.

## Functionality

The core functionalities are:
- fast distance calculations (geodesic and Great-Circle) for coordinate arrays, including pairwise
distance matrixes:
    - `geokernels.distance.geodist`
    - `geokernels.distance.geodist_matrix`
    - `geokernels.distance.greatcircle`
    - `geokernels.distance.greatcircle_matrix`
- geodesic kernel package (see `geokernels.kernels`) 

Improvements over current geodesic distance implementations:
- computational speed improvement of a factor of 50 to 150 in comparison to alternative Python packages 
for geodesic distances (geopy/geographilib), which is achieved via a numba accelerated inverse method of 
Vincenty's distance formula.
- Support of Numpy arrays as input for multiple coordinates and distance matrix calculations.
- Integration into scikit-learn Gaussian Process sklearn kernels.

The following geodesic kernels are added to the default Gaussian Process sklearn kernels:

- 'RBF_geo' (RBF kernel with geodesic distance metric)
- 'Matern_geo' (Matern kernel with geodesic distance metric)
- 'RationalQuadratic_geo' (Rational Quadratic kernel with geodesic distance metric)

All kernels can be used as drop-in replacement for the scikit-learn kernels RBF, Matern, 
and RationalQuadratic, respectively. 

The geodesic kernels address the problem if spatial coordinates of a dataset are given
as Latitude and Longitude (WGS84 coordinate system). This avoids the typical geospatial issue 
of having to transform the data coordinates into a local projected cartesian system beforehand, 
which can lead to inaccuracies at larger distances (e.g., continental scale or overlapping 
projected coordinate reference zones). 

Solving the geodesic problem is accomplished by defining a kernel that combines the 
geodesic distance metric for the spatial part with the Euclidean distance metric
for all non-spatial features (e.g., for spatial-temporal modeling). For more implementation
details, see sklearn_geokernels.kernels.py

The geodesic distance is computed via Vincenty's solution to the inverse geodetic problem, 
which is based on the WGS84 reference ellipsoid and is accurate to within 1 mm or better.
While the accuracy is comparable with other libraries for geodesic distance calculation,
such as GeographicLib/geopy, the geodesic distance computation implemented here is optimized 
for speed and more suitable for computing large arrays such as needed for Gaussian Process 
regression with scikit-learn. This implementation includes an automatic fallback option to 
the slower geographiclib algorithm in case of non-convergence of Vincenty's method (<0.01% of cases).


Both, anisotropic (one length-scale per feature) and isotropic (same length-scale for all features) 
kernels are supported. One important difference in comparison to the default sklearn kernels is the 
settings for the length-scale parameters in case of anisotropic kernel: Due to the non-euclidean metric 
for the geodesic distance, only one length-scale parameter is required instead of two separate for the two 
spatial dimensions (Latitude, Longitude) of the dataset. Thus, for an anisotropic kernel, the number 
of length-scales is one less than the number of dimensions of the data. 

Examples: see README: https://github.com/sigmaterra/geokernels
"""

__version__ = '0.2.2'
__author__ = 'Sebastian Haan'
__title__ = "geokernels"
__description__ = "fast geospatial distance and geodesic kernel computation for machine learning"
__uri__ = "https://github.com/sigmaterra/geokernels"
__doc__ = __description__ + " <" + __uri__ + ">"
__license__ = "MIT License"
__copyright__ = "Copyright (c) 2022 Sebastian Haan"