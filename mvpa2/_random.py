# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helper module for control of RNGs (numpy and stock python)"""

import random
import numpy as np

from mvpa2.base import cfg
if __debug__:
    from mvpa2.base import debug

#
# RNG seeding
#

def get_random_seed():
    """Generate a random int good for seeding RNG via `seed` function"""
    return int(np.random.uniform()*(2**31-1))

if cfg.has_option('general', 'seed'):
    _random_seed = cfg.getint('general', 'seed')
else:
    _random_seed = get_random_seed()

def seed(random_seed=_random_seed):
    """Uniform and combined seeding of all relevant random number
    generators.
    """
    if __debug__:
        debug('RANDOM', 'Reseeding RNGs with %s' % random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

seed(_random_seed)
