# Test Functions for geodesic distance computations
import os, sys
import timeit
import numpy as np
from scipy.spatial.distance import pdist, cdist

sys.path.append('..')
# import functions from geokernels package
from sklearn_geokernels.geodesics import geodesic_vincenty, geodist_dimwise


def gen_testdata(nsample = 100):
    """
    Generate random 2dim test data.

    Parameters:
    -----------
    nsample: int, number of samples

    Return
    ------
    X: numpy array, shape (nsample, 2)
        X[:,0] : Latitude
        X[:,1] : Longitude
    """
    X = np.random.rand(nsample, 2) * 180 - 90
    X[:,1] *= 2
    return X


def test_geodesic_vincenty():
    """
    Test the geodesic distance computation using the Vincenty formula.
    """
    X = gen_testdata()
    start = timeit.default_timer()
    dist = pdist(X, metric = lambda u, v: geodesic_vincenty(u, v))
    stop = timeit.default_timer()
    print(f'Time for distance matrix computation: {(stop - start):.3f} seconds')
    assert np.size(dist) == X.shape[0] * (X.shape[0] - 1) / 2
    return dist


def test_geodist_dimwise():
    """
    Test the pairwise geodesic distances between the data for each dimension.
    """
    X = gen_testdata()
    start = timeit.default_timer()
    dist = geodist_dimwise(X)
    stop = timeit.default_timer()
    print(f'Time for dimension-wsie distance matrix computation: {(stop - start):.3f} seconds')
    assert dist.shape[0] == dist.shape[1] == X.shape[0] 
    assert dist.shape[2] ==  X.shape[1] - 1
    return dist


def test_geodesic_geopy():
    """
    Test the geodesic distance computation using geopy.
    Note: Distant computation based on geographiclib.

    Requirement: geopy
    """
    try:
        from geopy.distance import geodesic as geodesic_geopy
    except:
        print("Package geopy not found. Please install it using pip install geopy")
        return 0
    X = gen_testdata()
    start = timeit.default_timer()
    dist = pdist(X, metric = lambda u, v: geodesic_geopy(u, v).meters)
    stop = timeit.default_timer()
    print(f'Time for distance matrix computation: {(stop - start):.3f} seconds')
    assert np.size(dist) == X.shape[0] * (X.shape[0] - 1) / 2
    return dist


def test_geodesic_geographiclib():
    """
    Test the geodesic distance computation using geographiclib.

    Requirement: geographiclib
    """
    try:
        from geographiclib.geodesic import Geodesic
    except:
        print("Package geographiclib not found. Please install it using pip install geographiclib")
        return 0
    X = gen_testdata()
    start = timeit.default_timer()
    dist = pdist(X, metric = lambda u, v: geodesic_geographiclib(u, v))
    stop = timeit.default_timer()
    print(f'Time for distance matrix computation: {(stop - start):.3f} seconds')
    assert np.size(dist) == X.shape[0] * (X.shape[0] - 1) / 2
    return dist


if __name__ == '__main__':
    dist = test_geodesic_vincenty()
    dist = test_geodist_dimwise()
    #dist = test_geodesic_geopy()
    #dist = test_geodesic_geographiclib()
    print('All tests passed')