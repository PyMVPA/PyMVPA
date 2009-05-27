#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Python distutils setup for PyMVPA"""

from numpy.distutils.core import setup, Extension
import os
import sys
from glob import glob

# some config settings
have_libsvm = False
extra_link_args = []
include_dirs = []
library_dirs = []

# platform-specific settings
if sys.platform == "darwin":
    extra_link_args.append("-bundle")

if sys.platform.startswith('linux'):
    # need to look for numpy (header location changes with v1.3)
    include_dirs += ['/usr/include/numpy']

# only if libsvm.a is available
if os.path.exists(os.path.join('build', 'libsvm', 'libsvm.a')):
    include_dirs += [os.path.join('3rd', 'libsvm')]
    library_dirs += [os.path.join('build', 'libsvm')]
    have_libsvm = True

# when libsvm is forced
if sys.argv.count('--with-libsvm'):
    # clean argv if necessary (or distutils will complain)
    sys.argv.remove('--with-libsvm')
    # look for libsvm in some places, when local one is not used
    if not have_libsvm:
        if not sys.platform.startswith('win'):
            include_dirs += ['/usr/include/libsvm-2.0/libsvm',
                             '/usr/include/libsvm',
                             '/usr/local/include/libsvm',
                             '/usr/local/include/libsvm-2.0/libsvm',
                             '/usr/local/include']
        else:
            # no clue on windows
            pass
    have_libsvm = True


# define the extension modules
libsvmc_ext = Extension(
    'mvpa.clfs.libsvmc._svmc',
    sources = ['mvpa/clfs/libsvmc/svmc.i'],
    include_dirs = include_dirs,
    library_dirs = library_dirs,
    libraries    = ['svm'],
    language     = 'c++',
    extra_link_args = extra_link_args,
    swig_opts    = ['-I' + d for d in include_dirs])

smlrc_ext = Extension(
    'mvpa.clfs.libsmlrc.smlrc',
    sources = [ 'mvpa/clfs/libsmlrc/smlr.c' ],
    library_dirs = library_dirs,
    libraries = ['m'],
    # extra_compile_args = ['-O0'],
    extra_link_args = extra_link_args,
    language = 'c')

ext_modules = [smlrc_ext]

if have_libsvm:
    ext_modules.append(libsvmc_ext)

# Notes on the setup
# Version scheme is: major.minor.patch<suffix>

# define the setup
setup(name         = 'pymvpa',
      version      = '0.4.2',
      author       = 'Michael Hanke, Yaroslav Halchenko, Per B. Sederberg',
      author_email = 'pkg-exppsy-pymvpa@lists.alioth.debian.org',
      license      = 'MIT License',
      url          = 'http://www.pymvpa.org',
      description  = 'Multivariate pattern analysis',
      long_description = \
          "PyMVPA is a Python module intended to ease pattern classification " \
          "analyses of large datasets. It provides high-level abstraction of " \
          "typical processing steps and a number of implementations of some " \
          "popular algorithms. While it is not limited to neuroimaging data " \
          "it is eminently suited for such datasets.\n" \
          "PyMVPA is truely free software (in every respect) and " \
          "additionally requires nothing but free-software to run.",
      # please maintain alphanumeric order
      packages     = [ 'mvpa',
                       'mvpa.algorithms',
                       'mvpa.atlases',
                       'mvpa.base',
                       'mvpa.clfs',
                       'mvpa.clfs.libsmlrc',
                       'mvpa.clfs.libsvmc',
                       'mvpa.clfs.sg',
                       'mvpa.datasets',
                       'mvpa.featsel',
                       'mvpa.mappers',
                       'mvpa.measures',
                       'mvpa.misc',
                       'mvpa.misc.bv',
                       'mvpa.misc.fsl',
                       'mvpa.misc.io',
                       'mvpa.misc.plot',
                       'mvpa.support',
                       'mvpa.tests',
                       ],
      data_files = [('mvpa/data',
                     [f for f in glob(os.path.join('mvpa', 'data', '*'))
                         if os.path.isfile(f)]),
                    ('mvpa/data/bv',
                     [f for f in glob(os.path.join('mvpa', 'data', 'bv', '*'))
                         if os.path.isfile(f)])],
      scripts      = glob(os.path.join('bin', '*')),
      ext_modules  = ext_modules
      )

