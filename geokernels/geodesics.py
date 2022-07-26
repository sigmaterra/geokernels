"""Geodesic distance calculation functions on a spheroid (WGS84).

The recommended function is based on Vincenty's inverse method formula
as implemented in the function geodesic_vincenty_inverse and accelerated with numba.

Alternative methods for computing geodesic distance via geopy or GeographicLib
are much slower (see README and test_geodesics.py).

However, in a few cases (<0.01%) Vincenty's inverse method can fail to converge, and
a fallback option using the slower geographiclib solution is implemented. 


Requirements:
- numpy
- numba
- scipy


Potential future upgrade: 
- function geodist_dimwise() might be accelerated
by adding jit decorator, however scipy's pdist and cdist seem not to be supported 
by numba-scipy yet.
- geographiclib numba acceleration is not yet implemented.


References:

- https://en.wikipedia.org/wiki/Vincenty's_formulae
- https://en.wikipedia.org/wiki/World_Geodetic_System
- https://en.wikipedia.org/wiki/Great-circle_distance
- https://geographiclib.sourceforge.io/
- Karney, Charles F. F. (January 2013). "Algorithms for geodesics". Journal of Geodesy. 87 (1): 43–55. 
arXiv:1109.4448. Bibcode:2013JGeod..87...43K. doi:10.1007/s00190-012-0578-z. Addenda.

"""

# Author: Sebastian Haan

import math 
import numpy as np
from scipy.spatial.distance import pdist, cdist
from numba import jit

from .geographiclib2.geodesic import Geodesic as geodesic_gglib


@jit(nopython=True)
def geodesic_vincenty_inverse(point1, point2):
    """
    Compute the geodesic distance between two points on the 
    surface of a spheroid (WGS84) based on Vincenty's formula 
    for the inverse geodetic problem.

    Parameters
    ----------
    point1 : (latitude_1, longitude_1)
    point2 : (latitude_2, longitude_2)

    Returns
    -------
    distance : float, in meters

    Note: this function is an optimized implementation of the 
    vincenty python package https://github.com/maurycyp/vincenty
    """

    # WGS84 ellipsoid parameters:
    a = 6378137  # meters
    f = 1 / 298.257223563
    # b = (1 - f)a, in meters
    b = 6356752.314245

    # Inverse method parameters:
    MAX_ITERATIONS = 200
    CONVERGENCE_THRESHOLD = 1e-11

    # short-circuit coincident points
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0.0

    U1 = math.atan((1 - f) * math.tan(math.radians(point1[0])))
    U2 = math.atan((1 - f) * math.tan(math.radians(point2[0])))
    L = math.radians(point2[1] - point1[1])
    Lambda = L

    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)
    sinU2 = math.sin(U2)
    cosU2 = math.cos(U2)

    for iteration in range(MAX_ITERATIONS):
        sinLambda = math.sin(Lambda)
        cosLambda = math.cos(Lambda)
        sinSigma = math.sqrt((cosU2 * sinLambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
        if sinSigma == 0:
            return 0.0  # coincident points
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2 
        if cosSqAlpha != 0.:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
            C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        else:
            cos2SigmaM = 0
            C = 0 
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM ** 2)))
        if abs(Lambda - LambdaPrev) < CONVERGENCE_THRESHOLD:
            break 
    else:
        #print('convergence', abs(Lambda - LambdaPrev)/ CONVERGENCE_THRESHOLD)
        return None  # no convergence

    uSq = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma *
                 (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM *
                 (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
    s = b * A * (sigma - deltaSigma)

    return round(s, 6)


def geodesic_vincenty(p1, p2):
    """
    Compute the geodesic distance between two points on the
    surface of a spheroid (WGS84) based on Vincenty's formula
    for the inverse geodetic problem[0].

    In the unlikely case Vincenty's inverse method fails to converge,
    the geographiclib algorithm is used instead.

    Parameters
    ----------
    p1 : (latitude_1, longitude_1)
    p2 : (latitude_2, longitude_2)

    Returns
    -------
    distance : float, in meters
    """
    d = geodesic_vincenty_inverse(p1, p2)
    if d is None:
        # in case vincenty fails to converge, use geographiclib
        return geodesic_gglib.WGS84.Inverse(p1[0], p1[1], p2[0], p2[1])['s12']
    else:
        return d



def geodist_dimwise(X):
    """
    Compute the pairwise geodesic distances between the data for each dimension.
    The distance for the first two dimensions is computed as combined geodesic distance,
    resulting in a distance metric that is spatially isotropic and has one 
    less dimension than the data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Returns
    -------
    distances : array-like, shape (n_samples, n_samples, n_features - 1), in meters squared.
    """
    # Initialise distances to zero
    dist = np.zeros((X.shape[0], X.shape[0],X.shape[1] - 1))
    # Compute geodesic distance for latitude and longitude
    dist[:,:,0] = cdist(X[:,:2], X[:,:2], metric = lambda u, v: geodesic_vincenty(u, v))
    # compute Euclidean distance for remaining dimensions
    dist[:,:,1:] = X[:, np.newaxis, 2:] - X[np.newaxis, :, 2:]

    return dist



@jit(nopython=True)
def harvesine_array(u,v):
    """
    Compute the geodesic distance squared between two points
    based on Harvesine formula using spherical geometry. 
    
    less accurate than Vincenty's inverse method.
    Only used if Vincenty does not converge

    Parameters
    ----------
    u : (latitude_1, longitude_1)
    v : (latitude_2, longitude_2)

    Returns
    -------
    distance : float, in meters
    """
    # Compute the haversine formula for latitude and longitude
    dlat = abs(np.radians(u[0] - v[0]))
    dlng = abs(np.radians(u[1] - v[1]))
    d = (np.sin(dlat/2)**2 + 
        (1 - np.sin(dlat/2)**2
        - np.sin(np.radians(u[0] + v[0]) / 2)**2)
        * np.sin(dlng/2)**2
        )
    return 6371009 * 2 * np.arcsin(np.sqrt(d))



@jit(nopython=True)
def harvesine(u,v):
    """
    Compute the geodesic distance squared between two points
    based on Harvesine formula using spherical geometry
    
    less accurate than Vincenty's inverse method.
    Only used if Vincenty does not converge

    Parameters
    ----------
    u : (latitude_1, longitude_1)
    v : (latitude_2, longitude_2)

    Returns
    -------
    distance : float, in meters
    """
    # Compute the haversine formula for latitude and longitude
    lat1, lng1 = math.radians(u[0]), math.radians(u[1])
    lat2, lng2 = math.radians(v[0]), math.radians(v[1])

    dlat = abs(lat2 - lat1)
    dlng = abs(lng2 - lng1)
    d = (math.sin(dlat/2)**2 +
        (1 - math.sin(dlat/2)**2
        - math.sin((lat1 + lat2) / 2)**2)
        * math.sin(dlng/2)**2
        )

    return 6371009 * 2 * math.asin(math.sqrt(d))


@jit(nopython=True)
def great_circle(u, v):
    """
    Use spherical geometry to calculate the surface distance between
    points.

    Parameters
    ----------
    u : (latitude_1, longitude_1)
    v : (latitude_2, longitude_2)

    Returns
    -------
    distance : float, in meters
    """

    lat1, lng1 = math.radians(u[0]), math.radians(u[1])
    lat2, lng2 = math.radians(v[0]), math.radians(v[1])

    sin_lat1, cos_lat1 = math.sin(lat1), math.cos(lat1)
    sin_lat2, cos_lat2 = math.sin(lat2), math.cos(lat2)

    delta_lng = abs(lng2 - lng1)
    cos_delta_lng, sin_delta_lng = math.cos(delta_lng), math.sin(delta_lng)

    d = math.atan2(math.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                    (cos_lat1 * sin_lat2 -
                    sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)
    
    return 6371009 * d


@jit(nopython=True)
def great_circle_array(u, v):
    """
    Use spherical geometry to calculate the surface distance between
    points.

    Parameters
    ----------
    u : (latitude_1, longitude_1), floats or arrays of floats
    v : (latitude_2, longitude_2), floats or arrays of floats

    Returns
    -------
    distance : float, in meters
    """

    lat1, lng1 = np.radians(u[0]), np.radians(u[1])
    lat2, lng2 = np.radians(v[0]), np.radians(v[1])

    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

    delta_lng = abs(lng2 - lng1)
    cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)

    d = np.arctan2(np.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                    (cos_lat1 * sin_lat2 -
                    sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)
    
    return 6371009 * d


@jit(nopython=True)
def geodesic_vincenty_harvesine(p1, p2):
    """
    Compute the geodesic distance between two points on the
    surface of a spheroid (WGS84) based on Vincenty's formula.

    This method uses here Harvesine solution as fallback for 
    Vincenty's inverse method in case of non-convergence.

    Parameters
    ----------
    p1 : (latitude_1, longitude_1)
    p2 : (latitude_2, longitude_2)

    Returns
    -------
    distance : float, in meters
    """
    d = geodesic_vincenty_start(p1, p2)
    if d is None:
        return harvesine(p1, p2)
    else:
        return d


def geodist_dimwise_harvesine(X):
    """
    Compute the squared pairwise geodesic distances between the data for each dimension.

    The dimension wise distances are approximated using the haversine formula 
    to split distance metric in latitudinal and longitudinal component.
    Spherical geometry is used to approximate the surface distance wuth a
    mean earth radius of 6371.009 km is used.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Returns
    -------
    distances : array-like, shape (n_samples, n_samples, n_features), in meters squared.
    """
    # Initialise distances to zero
    sdist = np.zeros((X.shape[0], X.shape[0],X.shape[1]))
    # Compute the haversine formula for latitude and longitude
    dlat = abs(np.radians(X[:, np.newaxis, 0] - X[np.newaxis, :, 0]))
    dlng = abs(np.radians(X[:, np.newaxis, 1] - X[np.newaxis, :, 1]))

    # delta latitude to meter
    sdist[:,:,0] = (6371009 * 2 * np.arcsin(abs(np.sin(dlat/2))))**2

    # delta longitude to meter:
    sdist[:,:,1] = (6371009 * 2 * np.arcsin(np.sqrt(
        (1 - np.sin(dlat/2)**2
        - np.sin(np.radians(X[:, np.newaxis, 0] + X[np.newaxis, :, 0]) / 2)**2)
        * np.sin(dlng /2)**2
        )))**2
    # Compute the pairwise squared Euclidean distances between the data for any remaining dimensions
    sdist[:,:,2:] = (X[:, np.newaxis, 2:] - X[np.newaxis, :, 2:]) ** 2

    return sdist



