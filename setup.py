#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Python distutils setup for PyMVPA"""

from distutils.core import setup, Extension
import numpy as N
import os
from glob import glob

# find numpy headers
numpy_headers = os.path.join(os.path.dirname(N.__file__),'core','include')

# Notes on the setup
# Version scheme is: major.minor.patch<suffix>

# define the extension modules
libsvmc_ext = Extension(
    'mvpa.clfs.libsvm.svmc',
    sources = [ 'mvpa/clfs/libsvm/svmc.i' ],
    include_dirs = [ '/usr/include/libsvm-2.0/libsvm', numpy_headers ],
    libraries    = [ 'svm' ],
    language     = 'c++',
    swig_opts    = [ '-c++', '-noproxy',
                     '-I/usr/include/libsvm-2.0/libsvm',
                     '-I' + numpy_headers ] )

smlrc_ext = Extension(
    'mvpa.clfs.libsmlr.smlrc',
    sources = [ 'mvpa/clfs/libsmlr/smlr.c' ],
    libraries = ['m'],
    # extra_compile_args = ['-O0'],
    language = 'c')

ext_modules = [smlrc_ext]

if 'PYMVPA_LIBSVM' in os.environ.keys():
    ext_modules.append(libsvmc_ext)

# define the setup
setup(name       = 'pymvpa',
      version      = '0.2.0-pre1',
      author       = 'Michael Hanke, Yaroslav Halchenko, Per B. Sederberg',
      author_email = 'pkg-exppsy-pymvpa@lists.alioth.debian.org',
      license      = 'MIT License',
      url          = 'http://pkg-exppsy.alioth.debian.org/pymvpa',
      description  = 'Multivariate pattern analysis',
      long_description = """\
PyMVPA is a Python module intended to ease pattern classification analyses
of large datasets. It provides high-level abstraction of typical processing
steps and a number of implementations of some popular algorithms. While it is
not limited to neuroimaging data it is eminently suited for such datasets.
PyMVPA is truely free software (in every respect) and additonally requires
nothing but free-software to run.""",
      packages     = [ 'mvpa',
                       'mvpa.datasets',
                       'mvpa.clfs',
                       'mvpa.clfs.libsvm',
                       'mvpa.clfs.libsmlr',
                       'mvpa.algorithms',
                       'mvpa.measures',
                       'mvpa.misc',
                       'mvpa.misc.fsl' ],
      ext_modules  = ext_modules
      )

