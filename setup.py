#!/usr/bin/env python

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#
#    Python distutils setup for PyMVPA
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

from distutils.core import setup, Extension
import numpy
import os
from glob import glob

# C++ compiler is needed for the extension
os.environ['CC'] = 'g++'

# find numpy headers
numpy_headers = os.path.join(os.path.dirname(numpy.__file__),'core','include')

# Notes on the setup
# Version scheme is:
# 0.<4-digit-year><2-digit-month><2-digit-day>.<ever-increasing-integer>

setup(name       = 'pymvpa',
    version      = '0.20070829.1',
    author       = 'Michael Hanke',
    author_email = 'michael.hanke@gmail.com',
    license      = 'LGPL',
    url          = 'http://apsy.gse.uni-magdeburg.de/hanke',
    description  = 'Multivariate pattern analysis',
    long_description = """ """,
    packages     = [ 'mvpa', 'mvpa.libsvm', 'mvpa.fsl' ],
    ext_modules  = [ Extension( 'mvpa.libsvm.svmc', [ 'mvpa/libsvm/svmc.i' ],
            include_dirs = [ '/usr/include/libsvm-2.0/libsvm', numpy_headers ],
            libraries    = [ 'svm' ],
            swig_opts    = [ '-c++', '-noproxy',
                             '-I/usr/include/libsvm-2.0/libsvm',
                             '-I' + numpy_headers ] ) ]
    )
