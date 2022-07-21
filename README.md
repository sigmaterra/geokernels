# geokernels: Geodesic kernels for Gaussian Process regression and classification 

This package is an extension of scikit-learn's gaussian_process kernel package
sklearn.gaussian_process.kernels.py (see 
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
which is based on the WGS84 reference ellipsoid and is accurate to within 1 cm or better.
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
#import standard libraries
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
# Import kernels similar to importing sklearn.gaussian_process.kernels :
from geokernels.sklearn_geokernels.kernels import RBF_geo, Matern_geo, RationalQuadratic_geo, WhiteKernel
from geokernels.tests.test_kernels import make_simdata1

# Add data: needs to include in the first two columns Latitude (first) and Longitude (second) coordinates.
# for testing use function make_simdata1 to generate 3 dim dataset (first two dimensions are Latitude, Longitude):
X, y = make_simdata1(n_samples = 100, noise = 0.1) 

# Split in train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define Kernel, e.g. anisotropic, which is one length-scale for geodesic distance given by coordinates
# and one length-scale for each other feature.
kernel = 1.0 * (
    RBF_geo(length_scale = [1e6, 1], length_scale_bounds = [(1e4, 1e7),(0.1, 1e4)]) 
    + WhiteKernel(noise_level_bounds=(1e-4, 10)))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=7)
# Fit GP, which includes default hyperparameter optimisation
gp.fit(X_train, y_train)

# Make predictions on test data
y_pred, y_std = gp.predict(X_test, return_std=True)
gp.score(X_test, y_test)
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