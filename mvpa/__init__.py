# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""MultiVariate Pattern Analysis


Package Organization
====================
The mvpa package contains the following subpackages and modules:

.. packagetree::
   :style: UML

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
:requires: Python 2.4+
:version: 0.4.5
:see: `The PyMVPA webpage <http://www.pymvpa.org>`__
:see: `GIT Repository Browser <http://git.debian.org/?p=pkg-exppsy/pymvpa.git;a=summary>`__

:license: The MIT License <http://www.opensource.org/licenses/mit-license.php>
:copyright: |copy| 2006-2010 Michael Hanke <michael.hanke@gmail.com>
:copyright: |copy| 2007-2010 Yaroslav O. Halchenko <debian@onerussian.com>

:newfield contributor: Contributor, Contributors (Alphabetical Order)
:contributor: `Emanuele Olivetti <emanuele@relativita.com>`__
:contributor: `Per B. Sederberg <persed@princeton.edu>`__

.. |copy| unicode:: 0xA9 .. copyright sign
"""

__docformat__ = 'restructuredtext'

# canonical PyMVPA version string
__version__ = '0.4.5'

import os
import random
import numpy as N
from mvpa.base import cfg
from mvpa.base import externals
from mvpa.base.info import wtf

# locate data root -- data might not be installed, but if it is, it should be at
# this location
pymvpa_dataroot = os.path.join(os.path.dirname(__file__), 'data')

if not __debug__:
    try:
        import psyco
        psyco.profile()
    except ImportError:
        from mvpa.base import verbose
        verbose(2, "Psyco online compilation is not enabled")
else:
    # Controllable seeding of random number generator
    from mvpa.base import debug

    debug('INIT', 'mvpa')

if cfg.has_option('general', 'seed'):
    _random_seed = cfg.getint('general', 'seed')
else:
    _random_seed = int(N.random.uniform()*(2**31-1))

def seed(random_seed):
    """Uniform and combined seeding of all relevant random number
    generators.
    """
    N.random.seed(random_seed)
    random.seed(random_seed)

seed(_random_seed)

# import the main unittest interface
from mvpa.tests import run as test

# PyMVPA is useless without numpy
# Also, this check enforcing population of externals.versions
# for possible later version checks, hence don't remove
externals.exists('numpy', force=True, raiseException=True)
# We might need to suppress the warnings so enforcing check here,
# it is ok if it would fail
externals.exists('scipy', force=True, raiseException=False)

if __debug__:
    debug('RANDOM', 'Seeding RNG with %d' % _random_seed)
    debug('INIT', 'mvpa end')

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

