# geokernels: fast geo-spatial distance and geodesic kernel computation for machine learning 
This Python package provides fast geo-spatial distance computation and geodesic distance 
kernels (e.g. for distance pairwise matrix calculation and Gaussian Process regressions). 
The geodesic kernel package is tailored to integrate geodesic kernels for scikit-learn's 
Gaussian Process models and can be used as drop-in replacement for sklearn.gaussian_process.kernels. 

Improvements over current geodesic distance implementations:
- computational speed improvement of a factor of 50 to 150 in comparison to alternative Python packages 
(geopy/geographilib), which is achieved via a numba accelerated inverse method of Vincenty's distance formula
(see Examples below). This implementation includes an automatic fallback option to the slower geographiclib 
algorithm in case of non-convergence of Vincenty's method (<0.01% of cases).
- Support of Numpy arrays as input for multiple coordinates and distance matrix calculations.
- Integration into scikit-learn Gaussian Process sklearn kernels

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
regression with scikit-learn.For more details, please see references and documentation in 
sklearn_geokernels.geodesics.py.

Both, anisotropic (one length-scale per feature) and isotropic (same length-scale for all features) 
kernels are supported. One important difference in comparison to the default sklearn kernels is the 
settings for the length-scale parameters in case of anisotropic kernel: Due to the non-euclidean metric 
for the geodesic distance, only one length-scale parameter is required instead of two separate for the two 
spatial dimensions (Latitude, Longitude) of the dataset. Thus, for an anisotropic kernel, the number 
of length-scales is one less than the number of dimensions of the data. 



## Installation

``` sh
pip install geokernels
```

## Requirements

- scikit-learn
- numba

## Examples

### Geodesic Distance Computation

``` python
import numpy as np
from geokernels.distance import geodist, geodist_matrix

# Calculate Distance between two points:
newport_ri = (41.49008, -71.312796)
cleveland_oh = (41.499498, -81.695391)
dist_km = geodist(newport_ri, cleveland_oh, metric = 'km')
dist_miles = geodist(newport_ri, cleveland_oh, metric = 'mile')
print(f"Geodesic distance: {dist_km:.3f} km or {dist_miles:.3f} miles")
#Out: Geodesic distance: 866.455 km or 538.390 miles

# Calculate distances between two numpy arrays of coordinates (row-wise).
# Distances are calculated as: dist[i] = distance(XA[i], XB[i])
XA, XB = np.random.rand(1000, 2) * 180 - 90, np.random.rand(1000, 2) * 180 - 90 
XA[:,1] *= 2 # longitude array A
XB[:,1] *= 2 # longitude array B
dists = geodist(XA, XB, metric = 'km')
print(f'Computed {len(dists)} distances with a mean distance of {dists.mean():.3f} km')
#Out: Computed 1000 distances with a mean distance of 10019.851 km

# Calculate distance matrix between all possible pairs of distances in array A.
# Note: upper and lower triangel of matrix are the same and diagonal is zero
dist_matrixA = geodist_matrix(XA, metric = 'meter')
print(f'Computed {dist_matrixA.shape[0]} x {dist_matrixA.shape[1]} distance matrix')
#Out: Computed 1000 x 1000 distance matrix

# Calculate distance matrix between all possible pairs between array A and B.
dist_matrixAB = geodist_matrix(XA, XB, metric = 'km')
print(f'Computed {dist_matrixAB.shape[0]} x {dist_matrixAB.shape[1]} distance matrix between XA and XB.') 
#Out Computed 1000 x 1000 distance matrix between XA and XB.
```

### Scikit-learn Gaussian Process Regression

``` python
#import standard libraries
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
# Import kernels similar to importing sklearn.gaussian_process.kernels :
from geokernels.kernels import RBF_geo, Matern_geo, RationalQuadratic_geo, WhiteKernel
from geokernels.test_kernels import make_simdata1

# Add data: needs to include in the first two columns Latitude (first) and Longitude (second) coordinates.
# for testing use function make_simdata1 to generate 3 dim dataset (first two dimensions are Latitude, Longitude):
X, y = make_simdata1(n_samples = 100, noise = 0.1) 

# Split in train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define Kernel. Here we choose an anisotropic RBF kernel. Note that there is only one length-scale parameter for
# the geodesic distance based on the combined first two features, Latitude and Longitude, and one length-scale
# per remaining feature (in this case only one more). See description above
kernel = 1.0 * (
    RBF_geo(length_scale = [1e6, 1], length_scale_bounds = [(1e4, 1e7),(0.1, 1e4)]) 
    + WhiteKernel(noise_level_bounds=(1e-4, 10)))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=7)
# Fit GP, which includes default hyperparameter optimization
gp.fit(X_train, y_train)

# Make predictions on test data
y_pred, y_std = gp.predict(X_test, return_std=True)
gp.score(X_test, y_test)
```

## Speed Comparison

Speed comparison between geokernels distance computation, 
geopy's: https://pypi.org/project/geopy/,
and geographiclib's: https://pypi.org/project/geographiclib/. 
Note: only geopy need to be installed since it requires geographiclib.
``` python
import numpy as np
from scipy.spatial.distance import pdist
import timeit

# Import all three methods:
from geopy.distance import geodesic as geodesic_geopy
from geographiclib.geodesic import Geodesic as geodesic_gglib
from geokernels.geodesics import geodesic_vincenty

# Generate 1000 random samples of coordinates (X[:,0] = Latitude, X[:,1] = Longitude)
X = np.random.rand(1000, 2) * 180 - 90
X[:,1] *= 2

# Calculate all pairwise distances with scipy pdist (here in total: 499,500 distances)

# with Geopy (distance calculation based on Geographiclib ):
start = timeit.default_timer()
dist_geopy = pdist(X, metric = lambda u, v: geodesic_geopy(u, v).meters)
stop = timeit.default_timer()
print(f'Time for Geopy computation: {(stop - start):.3f} seconds')
#Out: Time for Geopy computation: 53.356 seconds

# with Geographiclib, should be faster than geopy:
start = timeit.default_timer()
dist_gglib = pdist(X, metric = lambda u, v: geodesic_gglib.WGS84.Inverse(u[0], u[1], v[0], v[1])['s12'])
stop = timeit.default_timer()
print(f'Time for Geographiclib computation: {(stop - start):.3f} seconds')
#Out: Time for Geographiclib computation: 36.824 seconds

# geokernels (using our accelerated vincenty's inverse method):
start = timeit.default_timer()
dist_vincenty = pdist(X, metric = lambda u, v: geodesic_vincenty(u, v))
stop = timeit.default_timer()
print(f'Time for geokernels computation: {(stop - start):.3f} seconds')
#Out: Time for geokernels computation: 0.701 seconds (incl numba's compilation time)
#Out: Time for geokernels computation: 0.393 seconds (repeat after numba compilation)

# test accuracy in comparison to the standard by geographiclib
dist_mean = np.mean(np.abs(dist_vincenty - dist_gglib))
dist_max = np.max(np.abs(dist_vincenty - dist_gglib))
print(f'Mean difference in absolute distance [meters]: {dist_mean:.3e}') 
print(f'Max difference in absolute distance [meters]: {dist_max:.3e}')
#Out: Mean difference in absolute distance [meters]: 8.546e-06
#Out: Max difference in absolute distance [meters]: 1.269e-04
```
Speed improvement to geopy: factor of 78 to 142 times faster.
Speed improvement to geographiclib: factor of 53 to 94 times faster.



## Testing

Test functions and more examples can be found in geokernels.test_geodesics and geokernels.test_kernels. 

Test for all kernels:
``` python
from geokernels.test_kernels import test_allkernels
test_allkernels()
```
Comparison between geokernel's geodesic distance implementation and geographiclib/geopy:
(Note that this requires installation of at least geographiclib: pip install geographiclib)
``` python
from geokernels.test_geodesics import test_geodesic_vincenty, test_geodesic_geographiclib, test_accuracy
dist_vincenty = test_geodesic_vincenty()
dist_geographiclib = test_geodesic_geographiclib()
test_accuracy()
```

## References

- Vincenty, Thaddeus (August 1975b). Geodetic inverse solution between antipodal points. DMAAC Geodetic Survey Squadron. doi:10.5281/zenodo.32999.
- Carl Edward Rasmussen, Christopher K. I. Williams (2006). “Gaussian Processes for Machine Learning”. The MIT Press.
- David Duvenaud (2014). “The Kernel Cookbook: Advice on Covariance functions”.
- https://en.wikipedia.org/wiki/Vincenty's_formulae
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process
- Karney, Charles F. F. (January 2013). "Algorithms for geodesics". Journal of Geodesy. 87 (1): 43–55. 
arXiv:1109.4448. Bibcode:2013JGeod..87...43K. doi:10.1007/s00190-012-0578-z. Addenda.


## Contributors

Written by: Sebastian Haan
