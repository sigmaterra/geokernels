import os, sys
import timeit
import numpy as np
from scipy.spatial.distance import pdist, cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
try:
    import matplotlib.pyplot as plt
    _plot = True
except:
    print('WARNING: Plotting not available. Install matplotlib for plotting test results.')
    _plot = False

# import functions from geokernels package
sys.path.append('..')
from sklearn_geokernels.geodesics import geodesic_vincenty
# import geodesic kernels from sklearn_geokernels (as drop in for sklearn.gaussian_process.kernels):
from sklearn_geokernels.kernels import RBF_geo, Matern_geo, RationalQuadratic_geo, WhiteKernel, RBF, Matern


def make_simdata1(n_samples, noise = 0., random_state = None):
    """Generate Gaussian Process regression problem.
    Inputs `X` are independent features uniformly distributed. 

    The output `y` is created according to the formula::
        y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + noise * N(0, 1).

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.
    noise : float, default=0.0
        The standard deviation of the gaussian noise applied to the output.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset noise. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray of shape (n_samples,)
        The output values.

    """
    np.random.seed(random_state)
    X = np.random.rand(n_samples, 3)
    X[:,0] = (X[:,0] - 0.5) * 0.1 * np.pi  
    X[:,1] = (X[:,1] - 0.5) * 0.1 * np.pi
    #X[:, 2] = X[:, 2] * 1.0
    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + noise * np.random.rand(n_samples)
    return X, y


def test_gp(kernel_name = 'RBF_geo', n_samples= 200, plot=True):
    """
    Test of Gaussian Process regression with geodesic kernels

    This test automatically generates a 3D dataset and fits a Gaussian Process.

    Parameters
    ----------
    kernel_name : str, accepted: 'RBF_geo' (default), 'Matern_geo', or 'RationalQuadratic_geo'
    n_samples : int, default=200
    plot : bool, default=False
    """
    # Generate test data
    X, y = make_simdata1(n_samples, noise = 0.2, random_state =0)
    # Convert X to Latitude and Longitude coordinates in degree
    X[:,0:2] *= 180/np.pi
    # split in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # test anisotropic length_scale
    if kernel_name == 'RBF_geo':
        print('Test: RBF_geo')
        kernel = 1.0 * RBF_geo(length_scale = [1e5,1], 
        length_scale_bounds = [(1e4, 1e7),(0.1, 1e4)]) + WhiteKernel(noise_level_bounds=(1e-3, 1e2))
    elif kernel_name == 'Matern_geo':
        print('Test: Matern_geo')
        kernel = 1.0 * Matern_geo(length_scale = [1e5,1], 
        length_scale_bounds = [(1e4, 1e7),(0.1, 1e4)]) + WhiteKernel(noise_level_bounds=(1e-3, 1e2))
    elif kernel_name == 'RationalQuadratic_geo':
        print('Test: RationalQuadratic_geo')
        kernel = 1.0 * RationalQuadratic_geo(length_scale = 1, 
        length_scale_bounds = (1e-1, 1e4)) + WhiteKernel(noise_level_bounds=(1e-3, 1e2))
    else:
        print(f'Kernel name {kernel_name} not accepted. \
            Please choose from: RBF_geo, Matern_geo, or RationalQuadratic_geo.')
    #kernel = 1.0 * RBF(length_scale = [1,1,1], length_scale_bounds = (0.1, 1e4)) + WhiteKernel(noise_level_bounds=(1e-3, 1e2))
    start = timeit.default_timer()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
    gp.fit(X_train, y_train)
    fitstop = timeit.default_timer()
    print(f'Fitting time: {(fitstop - start):.2f} seconds')
    y_pred, y_std = gp.predict(X_test, return_std=True)
    stop = timeit.default_timer()
    print(f'Prediction time: {(stop - fitstop):.2f} seconds')
    print(f'RMSE: {np.sqrt(np.nanmean((y_pred - y_test)**2)):.4f}')
    print(f'R^2 : {gp.score(X_test, y_test):.4f}')
    print(f'Test completed')
    if _plot & plot:
        plt.clf()
        plt.errorbar(y_test, y_pred, yerr = y_std, label='Test data', fmt="o")
        plt.xlabel('Y true')
        plt.xlabel('Y prediction')
        plt.legend()
        plt.show()
    return True


def test_RBF_geo():
    """
    Test of Gaussian Process regression with geodesic RBF kernel
    """
    try:
        res_ok = test_gp(kernel_name = 'RBF_geo', n_samples= 100, plot=False)
    except:
        res_ok = False
    assert res_ok


def test_Matern_geo():
    """
    Test of Gaussian Process regression with geodesic RBF kernel
    """
    try:
        res_ok = test_gp(kernel_name = 'Matern_geo', n_samples= 100, plot=False)
    except:
        res_ok = False
    assert res_ok

    
def test_Quadratic_geo():
    """
    Test of Gaussian Process regression with geodesic RBF kernel
    """
    try:
        res_ok = test_gp(kernel_name = 'RationalQuadratic_geo', n_samples= 100, plot=False)
    except:
        res_ok = False
    assert res_ok


if __name__ == '__main__':
    res_ok = test_gp()
    assert res_ok