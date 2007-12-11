#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Simple preprocessing utilities"""

__docformat__ = 'restructuredtext'


from scipy import signal

from mvpa.misc.support import getBreakPoints

def detrend(data, perchunk=False, type='linear'):
    """
    Given a dataset, detrend the data inplace either entirely or per each chunk

    :Parameters:
      `data` : `Dataset`
        dataset to operate on
      `perchunk` : bool
        either to operate on whole dataset at once or on each chunk
        separately
      `type`
        type accepted by scipy.signal.detrend. Currently only
        'linear' or 'constant' (which is just demeaning)

    """

    bp = 0                              # no break points by default

    if perchunk:
        bp = getBreakPoints(data.chunks)

    data.samples[:] = signal.detrend(data.samples, axis=0, type=type, bp=bp)
