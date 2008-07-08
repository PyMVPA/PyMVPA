#!/usr/bin/env python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Example of fitting an HRF model to noisy peristimulus data"""

import numpy as N
import pylab as P

from mvpa.misc.plot import errLinePlot
from mvpa.misc.fx import singleGammaHRF, leastSqFit
from mvpa import cfg

# make dataset
# 40 identical 'trial time courses' generated from a simple gamma function
#   time-to-peak: 6s
#   FWHM: 7s
#   Scaling: 1
a = N.asarray([singleGammaHRF(N.arange(20), A=6, W=7, K=1)] * 40)
# get closer to reality
a += N.random.normal(size=a.shape)


# now fit a gamma function, parameter start values:
#   time-to-peak: 5s
#   FWHM: 5s
#   Scaling: 1
fpar, succ = leastSqFit(singleGammaHRF, [5,5,1], a)

# generate high-resultion curves for the 'true' time course
# and the fitted one
curves = [singleGammaHRF(N.linspace(0,20), 6, 7, 1),
          singleGammaHRF(N.linspace(0,20), *fpar)]

# plot data (with error bars) and both curves
errLinePlot(a, curves=curves, linestyle='-')

# add legend to plot
P.legend(('original', 'fit'))

if cfg.getboolean('examples', 'interactive', True):
    # show the cool figure
    P.show()
