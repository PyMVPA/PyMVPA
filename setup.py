#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Python distutils setup for PyMVPA"""

from distutils.core import setup, Extension
import numpy as N
import os
from glob import glob

# C++ compiler is needed for the extension
os.environ['CC'] = 'g++'

# find numpy headers
numpy_headers = os.path.join(os.path.dirname(N.__file__),'core','include')

# Notes on the setup
# Version scheme is:
# 0.<4-digit-year><2-digit-month><2-digit-day>.<ever-increasing-integer>

setup(name       = 'pymvpa',
    version      = '0.20070829.1',
    author       = 'Michael Hanke',
    author_email = 'michael.hanke@gmail.com',
    license      = 'MIT License',
    url          = 'http://apsy.gse.uni-magdeburg.de/hanke',
    description  = 'Multivariate pattern analysis',
    long_description = """ """,
    packages     = [ 'mvpa',
                     'mvpa.datasets',
                     'mvpa.clf',
                     'mvpa.clf.libsvm',
                     'mvpa.algorithms',
                     'mvpa.misc',
                     'mvpa.misc.fsl' ],
    ext_modules  = [ Extension( 'mvpa.clf.libsvm.svmc',
                                [ 'mvpa/clf/libsvm/svmc.i' ],
            include_dirs = [ '/usr/include/libsvm-2.0/libsvm', numpy_headers ],
            libraries    = [ 'svm' ],
            swig_opts    = [ '-c++', '-noproxy',
                             '-I/usr/include/libsvm-2.0/libsvm',
                             '-I' + numpy_headers ] ) ]
    )
