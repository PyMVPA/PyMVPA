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

from distutils.core import setup
import os
from glob import glob

# Notes on the setup
# Version scheme is:
# 0.<4-digit-year><2-digit-month><2-digit-day>.<ever-increasing-integer>

setup(name       = 'pymvpa',
    version      = '0.20070524.1',
    author       = 'Michael Hanke',
    author_email = 'michael.hanke@gmail.com',
    license      = 'LGPL',
    url          = 'http://apsy.gse.uni-magdeburg.de/hanke',
    description  = 'Multivariate pattern analysis',
    long_description = """ """,
    packages     = [ 'mvpa' ],
    )

