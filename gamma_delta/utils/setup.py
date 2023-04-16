from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np

setup(
    name='mcmc_cython',
    ext_modules=cythonize([Extension("mcmc_cython", ["mcmc_cython.pyx"], include_dirs=[np.get_include()])],
                          compiler_directives={'language_level' : "3"},
                          annotate=True),
    zip_safe=False,
)
