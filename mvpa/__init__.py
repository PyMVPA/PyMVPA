#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
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

:group Basic Data Structures: datasets
:group Classifiers: clf
:group Algorithms: algorithms
:group Miscellaneous: misc

:author: `Michael Hanke <michael.hanke@gmail.com>`__,
         `Yaroslav Halchenko <debian@onerussian.com>`__,
         `Per B. Sederberg <persed@princeton.edu>`__
:requires: Python 2.4+
:version: 0.2.2
:see: `The PyMVPA webpage <http://pkg-exppsy.alioth.debian.org/pymvpa>`__
:see: `GIT Repository Browser <http://git.debian.org/?p=pkg-exppsy/pymvpa.git;a=summary>`__

:license: The MIT License
:copyright: |copy| 2006-2008 Michael Hanke <michael.hanke@gmail.com>

:newfield contributor: Contributor, Contributors (Alphabetical Order)
:contributor: `Per B. Sederberg <persed@princeton.edu>`__

.. |copy| unicode:: 0xA9 .. copyright sign
"""

__docformat__ = 'restructuredtext'

import os
import random
import numpy as N

if not __debug__:
    try:
        import psyco
        psyco.profile()
    except:
        from mvpa.misc import verbose
        verbose(2, "Psyco online compilation is not enabled")
else:
    # Controllable seeding of random number generator
    from mvpa.misc import debug

    debug('INIT', 'mvpa')

if os.environ.has_key('MVPA_SEED'):
    __random_seed = int(os.environ['MVPA_SEED'])
else:
    __random_seed = int(N.random.uniform()*(2**31-1))

N.random.seed(__random_seed)
random.seed(__random_seed)

if __debug__:
    debug('RANDOM', 'Seeding RNG with %d' % __random_seed)
    debug('INIT', 'mvpa end')
