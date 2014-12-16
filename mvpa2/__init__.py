# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Framework for multivariate pattern analysis (MVPA)


Package Organization
====================
The mvpa2 package contains the following subpackages and modules:

:group Algorithms: algorithms
:group Anatomical Atlases: atlases
:group Basic Data Structures: datasets
:group Classifiers (supervised learners): clfs
:group Feature Selections: featsel
:group Mappers (usually unsupervised learners): mappers
:group Measures: measures
:group Miscellaneous: base misc support
:group Unittests: tests

:author: `Michael Hanke <michael.hanke@gmail.com>`__,
         `Yaroslav Halchenko <debian@onerussian.com>`__,
         `Per B. Sederberg <persed@princeton.edu>`__
         `Nikolaas N. Oosterhof <n.n.oosterhof@googlemail.com>`__
:requires: Python 2.6+
:version: 2.3.1
:see: `The PyMVPA webpage <http://www.pymvpa.org>`__
:see: `GIT Repository Browser <http://github.com/PyMVPA/PyMVPA>`__

:license: The MIT License <http://www.opensource.org/licenses/mit-license.php>
:copyright: |copy| 2006-2014 Michael Hanke <michael.hanke@gmail.com>
:copyright: |copy| 2007-2014 Yaroslav O. Halchenko <debian@onerussian.com>
:copyright: |copy| 2012-2014 Nikolaas N. Oosterhof <n.n.oosterhof@googlemail.com>

.. |copy| unicode:: 0xA9 .. copyright sign
"""

__docformat__ = 'restructuredtext'

# canonical PyMVPA version string
__version__ = '2.3.1'

import os
import random
import numpy as np
from mvpa2.base import cfg
from mvpa2.base import externals
from mvpa2.base.info import wtf

# commit hash to be filled in by Git upon export/archive
hashfilename = os.path.join(os.path.dirname(__file__), 'COMMIT_HASH')
__hash__ = ''
if os.path.exists(hashfilename):
    hashfile = open(hashfilename, 'r')
    __hash__ = hashfile.read().strip()
    hashfile.close()

#
# Data paths
#

# locate data root -- data might not be installed, but if it is, it should be at
# this location
pymvpa_dataroot = \
        cfg.get('data', 'root',
                default=os.path.join(os.path.dirname(__file__), 'data'))
# locate PyMVPA data database root -- also might not be installed, but if it is,
# it should be at this location
pymvpa_datadbroot = \
        cfg.get('datadb', 'root',
                default=os.path.join(os.getcwd(), 'datadb'))


#
# Debugging and optimization
#

if not __debug__:
    try:
        import psyco
        psyco.profile()
    except ImportError:
        from mvpa2.base import verbose
        verbose(2, "Psyco online compilation is not enabled")
else:
    # Controllable seeding of random number generator
    from mvpa2.base import debug
    debug('INIT', 'mvpa')

#
# RNGs control
#

from mvpa2._random import _random_seed, seed, get_random_seed

#
# Testing
#

# import the main unittest interface
from mvpa2.tests import run as test

#
# Externals-dependent tune ups
#

# PyMVPA is useless without numpy
# Also, this check enforcing population of externals.versions
# for possible later version checks, hence don't remove
externals.exists('numpy', force=True, raise_=True)
# We might need to suppress the warnings:

# If instructed -- no python or numpy warnings (like ctypes version
# for slmr), e.g. for during doctests
if cfg.getboolean('warnings', 'suppress', default=False):
    import warnings
    warnings.simplefilter('ignore')
    # NumPy
    np.seterr(**dict([(x, 'ignore') for x in np.geterr()]))

if externals.exists('scipy'):
    externals._suppress_scipy_warnings()

# And check if we aren't under IPython so we could pacify completion
# a bit
externals.exists('running ipython env', force=True, raise_=False)
# Check for matplotlib so matplotlib backend becomes set according to
# our configuration
externals.exists('matplotlib', force=True, raise_=False)

#
# Hooks
#

# Attach custom top-level exception handler
if cfg.getboolean('debug', 'wtf', default=False):
    import sys
    _sys_excepthook = sys.excepthook
    def _pymvpa_excepthook(*args):
        """Custom exception handler to report also pymvpa's wtf

        Calls original handler, and then collects WTF and spits it out
        """
        ret = _sys_excepthook(*args)
        sys.stdout.write("PyMVPA's WTF: collecting information... hold on...")
        sys.stdout.flush()
        wtfs = wtf()
        sys.stdout.write("\rPyMVPA's WTF:                                       \n")
        sys.stdout.write(str(wtfs))
        return ret
    sys.excepthook = _pymvpa_excepthook

# Attach custom top-level exception handler
if cfg.getboolean('debug', 'pdb', default=False):
    import sys
    _sys_excepthook = sys.excepthook
    def _pymvpa_pdb_excepthook(type, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type, value, tb)
        else:
            import traceback, pdb
            traceback.print_exception(type, value, tb)
            print
            pdb.post_mortem(tb)
    sys.excepthook = _pymvpa_pdb_excepthook

if __debug__:
    debug('INIT', 'mvpa end')
