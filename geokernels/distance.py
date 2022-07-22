"""Computation of geospatial distances (WGS84).

The geospatial distance calculation is based on Vincenty's inverse method formula
(see geokernels.geodesics.geodesic_vincenty).

Coordinates are assumed to be in Latitude and Longitude (WGS 84).

geodist: return a list of distances between point to point
geodist_matrix: returns distance matrix between all possible combinations of 
pairwise distances.

This implementationprovides a fast computation of geo-spatial distances in comparison to 
alternative methods for computing geodesic distance (tested:  geopy and GeographicLib, 
see geokernels.test_geodesics for test functions).

"""

# Author: Sebastian Haan

import numpy as np
from scipy.spatial.distance import pdist, cdist

from .geodesics import geodesic_vincenty


def geodist(coords1, coords2, metric = 'meter'):
    """
    Return distances between two coordinates or two lists of coordinates.
    Distances are calculated row by row: dist_i = distance(coord1_i, coord2_i)

    Coordinates are assumed to be in Latitude, Longitude (WGS 84).

    For distances between all pair combintaions see geo_pdist and geo_cdist.

    Parameters
    -----------
    coords1: (lat, long), floats or array with shape (n_points1, 2)
    coords2: (lat, long), floats or array with shape (n_points2, 2)
        coords1.shape = coords2.shape
    metric: 'meter', 'km', 'mile', 'nmi'

    Return
    ------
    dist: float or numpy array, distance(s) between points, shape = (n_points,)
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
        print(f'Metric {metric} not supported')
        return None
    if np.size(coords1) == 2:
        return geodesic_vincenty(coords1, coords2) * conv_fac
    if coords1.shape[1] != 2:
        print('coords1 and coords2 must have at two dimensions: Latitude, Longitude ')
        return None
    if (abs(coords1[:,0]) > 90).any() | (abs(coords2[:,0]) > 90).any():
        print('First dimension must be Latitude: -90 < lat < 90')
        return None
    if (abs(coords1[:,1]) > 180).any() | (abs(coords2[:,1]) > 180).any():
        print('Second dimension must be Longitude: -180 < long < 180')
        return None
    n_points = len(coords1)
    dist = np.asarray([geodesic_vincenty(coords1[i], coords2[i]) for i in range(n_points)])
    return dist * conv_fac


def geodist_matrix(coords1, coords2 = None, metric = 'meter'):
    """
    Compute distance between each pair of possible combinations.
    If coords2 = None, compute distance between all possible pair combinations in coords1.
    if coords2 is given, Compute distance between each possible pair of the two collections 
    of inputs.

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
        print('coords1 and coords2 must have at two dimensions: Latitude, Longitude ')
        return None
    if (abs(coords1[:,0]) > 90).any():
        print('First dimension must be Latitude: -90 < lat < 90')
        return None
    if (abs(coords1[:,1]) > 180).any():
        print('Second dimension must be Longitude: -180 < lng < 180')
        return None
    if coords2 == None:
        # if only one list of coordinates is given:
        dist = pdist(X, metric = lambda u, v: geodesic_vincenty(u, v))
        dist = squareform(dist)
        np.fill_diagonal(dist, 1)
    else:
        # if two lists of coordinates are given
        assert coords1.shape == coords2.shape
        if (abs(coords2[:,0]) > 90).any():
            print('First dimension must be Latitude: -90 < lat < 90')
            return None
        if (abs(coords2[:,1]) > 180).any():
            print('Second dimension must be Longitude: -180 < lng < 180')
            return None
        dist = cdist(X, Y, metric = lambda u, v: geodesic_vincenty(u, v))
    return dist * conv_fac