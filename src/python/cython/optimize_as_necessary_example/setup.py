from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


ext_modules = [
    Extension('mcmc', sources=['mcmc.pyx'], libraries=['m']),
    Extension('functions', sources=['functions.pyx'])
]


setup(
    ext_modules = cythonize(ext_modules,
                           include_path=[numpy.get_include()])
)
