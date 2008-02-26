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

# C++ compiler is needed for the extension
os.environ['CC'] = 'g++'

# find numpy headers
numpy_headers = os.path.join(os.path.dirname(N.__file__),'core','include')

# Notes on the setup
# Version scheme is: major.minor.patch<suffix>

setup(name       = 'pymvpa',
    version      = '0.1.0',
    author       = 'Michael Hanke, Yaroslav Halchenko',
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
                     'mvpa.algorithms',
                     'mvpa.misc',
                     'mvpa.misc.fsl' ],
    ext_modules  = [ Extension( 'mvpa.clfs.libsvm.svmc',
                                [ 'mvpa/clfs/libsvm/svmc.i' ],
            include_dirs = [ '/usr/include/libsvm-2.0/libsvm', numpy_headers ],
            libraries    = [ 'svm' ],
            swig_opts    = [ '-c++', '-noproxy',
                             '-I/usr/include/libsvm-2.0/libsvm',
                             '-I' + numpy_headers ] ) ]
    )
