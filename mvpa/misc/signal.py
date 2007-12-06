#!/usr/bin/python
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


def detrend(data, perchunk=False):
    """
    Given a dataset, detrend the data either entirely or per each chunk

    :Parameters:
      `data` : `Dataset`
        dataset to operate on
      `perchunk` : bool
        either to operate on whole dataset at once or on each chunk
        separately
    :TODO:
     * make possible to detrend providing chunk boundaries as breakpoints
       for detrend
    """

    if perchunk:
        for chunk in data.uniquechunks:
            ids = data.idsbychunks(chunk)
            detrended = signal.detrend(data.samples[ids, :], axis=0)
            data.samples[ids, :] = detrended[:]
    else:
        data.samples[:] = signal.detrend(data.samples, axis=0)
