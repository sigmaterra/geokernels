"""Computation of geospatial distances (WGS84).

Coordinates are assumed to be in Latitude and Longitude (WGS 84).
Accepting numpy arrays as input.

The geospatial distance calculation is based on Vincenty's inverse method formula
and accelerated with Numba (see geokernels.geodesics.geodesic_vincenty and references).

In a few cases (<0.01%) Vincenty's inverse method can fail to converge, and
a fallback option using the slower geographiclib solution is implemented.


Functions included:
- geodist: returns list of distances between points of two lists: 
dist[i] = distance(XA[i], XB[i])

geodist_matrix: returns distance matrix between all possible combinations of 
pairwise distances (either between all points in one list or points between two lists).
dist[i,j] = distance(XA[i], XB[j]) or distance(X[i], X[j])

This implementation provides a fast computation of geo-spatial distances in comparison to 
alternative methods for computing geodesic distance (tested:  geopy and GeographicLib, 
see geokernels.test_geodesics for test functions).

References:

- https://en.wikipedia.org/wiki/Vincenty's_formulae
- https://geographiclib.sourceforge.io/
- Karney, Charles F. F. (January 2013). "Algorithms for geodesics". Journal of Geodesy. 87 (1): 43–55. 
arXiv:1109.4448. Bibcode:2013JGeod..87...43K. doi:10.1007/s00190-012-0578-z. Addenda.
"""

# Author: Sebastian Haan

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

from .geodesics import geodesic_vincenty, great_circle, great_circle_array


def geodist(coords1, coords2, metric = 'meter'):
    """
    Return distances between two coordinates or two lists of coordinates.
    Distances are calculated as: dist[i] = distance(XA[i], XB[i])

    Coordinates are assumed to be in Latitude, Longitude (WGS 84).

    For distances between all pair combintaions see geo_pdist and geo_cdist.

    Parameters
    -----------
    coords1: (lat, long), list or array with shape (n_points1, 2) for multiple points
    coords2: (lat, long), list or array with shape (n_points2, 2) for multiple points
        coords1.shape = coords2.shape
    metric: 'meter', 'km', 'mile', 'nmi'

    Return
    ------
    dist: float or numpy array, distance(s) between points, length = n_points
    """
    coords1 = np.asarray(coords1)
    coords2 = np.asarray(coords2)
    assert coords1.shape == coords2.shape
    if metric == 'meter':
        conv_fac = 1
    elif metric == 'km':
        conv_fac = 1e-3
    elif metric == 'mile':
        conv_fac = 1 / 1609.344
    elif metric == 'nmi':
        conv_fac = 1 / 1852.
    else:
        raise ValueError(f'Metric {metric} not supported')
    if np.size(coords1) == 2:
        return geodesic_vincenty(coords1, coords2) * conv_fac
    if coords1.shape[1] != 2:
        raise ValueError('coords1 and coords2 must have at two dimensions: Latitude, Longitude ')
    if (abs(coords1[:,0]) > 90).any() | (abs(coords2[:,0]) > 90).any():
        raise ValueError('First dimension must be Latitude: -90 < lat < 90')
    if (abs(coords1[:,1]) > 180).any() | (abs(coords2[:,1]) > 180).any():
        raise ValueError('Second dimension must be Longitude: -180 < long < 180')
    n_points = len(coords1)
    dist = np.asarray([geodesic_vincenty(coords1[i], coords2[i]) for i in range(n_points)])
    return dist * conv_fac


def geodist_matrix(coords1, coords2 = None, metric = 'meter'):
    """
    Compute distance between each pair of possible combinations.
    
    If coords2 = None, compute distance between all possible pair combinations in coords1.
    dist[i,j] = distance(XA[i], XB[j])

    If coords2 is given, compute distance between each possible pair of the two collections 
    of inputs: dist[i,j] = distance(X[i], X[j])

    Coordinates are assumed to be in Latitude, Longitude (WGS 84).

    Parameters
    -----------
    coords1: [(lat, long)], array with shape (n_points1, 2)
    coords2: [(lat, long)], array with shape (n_points2, 2)
        if coords not None: coords1.shape = coords2.shape
    metric: 'meter', 'km', 'mile', 'nmi'

    Return
    ------
    dist: distance matrix is returned. 
    if only coords1 are given: for each i and j the metric dist(u=XA[i], v=XA[j]) is computed
    If coords2 i not None: for each i and j, the metric dist(u=XA[i], v=XB[j]) 
    is computed and stored in the ij-th entry.
    """
    if metric == 'meter':
        conv_fac = 1
    elif metric == 'km':
        conv_fac = 1e-3
    elif metric == 'mile':
        conv_fac = 1 / 1609.344
    elif metric == 'nmi':
        conv_fac = 1 / 1852.
    else:
        print(f'Metric {metric} not supported')
        return None
    coords1 = np.asarray(coords1)
    if coords1.shape[1] != 2:
        raise ValueError('coords1 and coords2 must have at two dimensions: Latitude, Longitude')
    if (abs(coords1[:,0]) > 90).any():
        raise ValueError('First dimension must be Latitude: -90 < lat < 90')
    if (abs(coords1[:,1]) > 180).any():
        raise ValueError('Second dimension must be Longitude: -180 < lng < 180')
    if coords2 is None:
        coords2 = np.asarray(coords2)
        # if only one list of coordinates is given:
        dist = pdist(coords1, metric = lambda u, v: geodesic_vincenty(u, v))
        dist = squareform(dist)
    else:
        # if two lists of coordinates are given
        assert coords1.shape == coords2.shape
        if (abs(coords2[:,0]) > 90).any():
            raise ValueError('First dimension must be Latitude: -90 < lat < 90')
        if (abs(coords2[:,1]) > 180).any():
            raise ValueError('Second dimension must be Longitude: -180 < lng < 180')
        dist = cdist(coords1, coords2, metric = lambda u, v: geodesic_vincenty(u, v))
    return dist * conv_fac


def greatcircle(coords1, coords2, metric = 'meter'):
    """
    Return distances between two coordinates or two lists of coordinates
    using spherical asymmetry (Great Circle approximation).
    Distances are calculated as: dist[i] = distance(XA[i], XB[i])

    Coordinates are assumed to be in Latitude, Longitude (WGS 84).

    For distances between all pair combinations see geo_pdist and geo_cdist.

    Parameters
    -----------
    coords1: (lat, long), list or array with shape (n_points1, 2) for multiple points
    coords2: (lat, long), list or array with shape (n_points2, 2) for multiple points
        coords1.shape = coords2.shape
    metric: 'meter', 'km', 'mile', 'nmi'

    Return
    ------
    dist: float or numpy array, distance(s) between points, length = n_points
    """
    coords1 = np.asarray(coords1)
    coords2 = np.asarray(coords2)
    assert coords1.shape == coords2.shape
    if metric == 'meter':
        conv_fac = 1
    elif metric == 'km':
        conv_fac = 1e-3
    elif metric == 'mile':
        conv_fac = 1 / 1609.344
    elif metric == 'nmi':
        conv_fac = 1 / 1852.
    else:
        raise ValueError(f'Metric {metric} not supported')
    if np.size(coords1) == 2:
        return geodesic_vincenty(coords1, coords2) * conv_fac
    if coords1.shape[1] != 2:
        raise ValueError('coords1 and coords2 must have at two dimensions: Latitude, Longitude ')
    if (abs(coords1[:,0]) > 90).any() | (abs(coords2[:,0]) > 90).any():
        raise ValueError('First dimension must be Latitude: -90 < lat < 90')
    if (abs(coords1[:,1]) > 180).any() | (abs(coords2[:,1]) > 180).any():
        raise ValueError('Second dimension must be Longitude: -180 < long < 180')
    n_points = len(coords1)
    dist = great_circle_array(coords1, coords2)
    return dist * conv_fac


def greatcircle_matrix(coords1, coords2 = None, metric = 'meter'):
    """
    Compute distance between each pair of possible combinations
    using spherical asymmetry (Great Circle approximation).
    
    If coords2 = None, compute distance between all possible pair combinations in coords1.
    dist[i,j] = distance(XA[i], XB[j])

    If coords2 is given, compute distance between each possible pair of the two collections 
    of inputs: dist[i,j] = distance(X[i], X[j])

    Coordinates are assumed to be in Latitude, Longitude (WGS 84).

    Parameters
    -----------
    coords1: [(lat, long)], array with shape (n_points1, 2)
    coords2: [(lat, long)], array with shape (n_points2, 2)
        if coords not None: coords1.shape = coords2.shape
    metric: 'meter', 'km', 'mile', 'nmi'

    Return
    ------
    dist: distance matrix is returned. 
    if only coords1 are given: for each i and j the metric dist(u=XA[i], v=XA[j]) is computed
    If coords2 i not None: for each i and j, the metric dist(u=XA[i], v=XB[j]) 
    is computed and stored in the ij-th entry.
    """
    if metric == 'meter':
        conv_fac = 1
    elif metric == 'km':
        conv_fac = 1e-3
    elif metric == 'mile':
        conv_fac = 1 / 1609.344
    elif metric == 'nmi':
        conv_fac = 1 / 1852.
    else:
        print(f'Metric {metric} not supported')
        return None
    coords1 = np.asarray(coords1)
    if coords1.shape[1] != 2:
        raise ValueError('coords1 and coords2 must have at two dimensions: Latitude, Longitude')
    if (abs(coords1[:,0]) > 90).any():
        raise ValueError('First dimension must be Latitude: -90 < lat < 90')
    if (abs(coords1[:,1]) > 180).any():
        raise ValueError('Second dimension must be Longitude: -180 < lng < 180')
    if coords2 is None:
        coords2 = np.asarray(coords2)
        # if only one list of coordinates is given:
        dist = pdist(coords1, metric = lambda u, v: great_circle(u, v))
        dist = squareform(dist)
    else:
        # if two lists of coordinates are given
        assert coords1.shape == coords2.shape
        if (abs(coords2[:,0]) > 90).any():
            raise ValueError('First dimension must be Latitude: -90 < lat < 90')
        if (abs(coords2[:,1]) > 180).any():
            raise ValueError('Second dimension must be Longitude: -180 < lng < 180')
        dist = cdist(coords1, coords2, metric = lambda u, v: great_circle(u, v))
    return dist * conv_fac
