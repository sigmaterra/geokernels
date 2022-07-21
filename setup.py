try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from os import path
import io

## in development set version
PYPI_VERSION = '0.1.1'

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if __name__ == "__main__":
    setup(name = 'geokernels',
          author            = "Sebastian Haan",
          author_email      = "sebhaan@sigmaterra.com",
          url               = "https://github.com/sigmaterra/geokernels",
          version           = PYPI_VERSION ,
          description       = "Geodesic Gaussian Process kernels for scikit-learn",
          long_description  = long_description,
          long_description_content_type='text/markdown',
          license           = 'MIT',
          install_requires  = ['scikit_learn>=1.0',
                                'numba>=0.53'],
          python_requires   = '>=3.8',
          packages          = ['geokernels'],
          classifiers       = ['Programming Language :: Python :: 3',
                                'Operating System :: OS Independent',
                               ]
          )