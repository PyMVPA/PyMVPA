# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wrapper around the output of MELODIC (part of FSL)"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import externals
if externals.exists('nibabel', raise_=True):
    import nibabel as nb


class MelodicResults( object ):
    """Easy access to MELODIC output.

    Only important information is available (important as judged by the
    author).
    """
    def __init__( self, path, fext='.nii.gz'):
        """Reads all information from the given MELODIC output path.
        """
        self._fext = fext
        rpath = None
        lookup = ['', 'filtered_func_data.ica']
        for lu in lookup:
            if os.path.exists(pathjoin(path, lu, 'melodic_IC' + fext)):
                rpath = pathjoin(path, lu)
                break
        if rpath is None:
            raise ValueError("Cannot find Melodic results at '%s'" % path)
        else:
            self._rpath = rpath
        self._ic = nb.load(pathjoin(rpath, 'melodic_IC' + fext))
        self._icshape = self._ic.shape
        self._mask = nb.load(pathjoin(rpath, 'mask' + fext))
        self._tmodes = np.loadtxt(pathjoin(rpath, 'melodic_Tmodes' ))
        self._smodes = np.loadtxt(pathjoin(rpath, 'melodic_Smodes'))
        self._icstats = np.loadtxt(pathjoin(rpath, 'melodic_ICstats'))


    def _get_stat(self, type, ic):
        # melodic's IC number is one-based, we do zero-based
        img = nb.load(pathjoin(self._rpath, 'stats',
                                   '%s%i' % (type, ic + 1) + self._fext))
        return img.get_data()

    def get_probmap(self, ic):
        return self._get_stat('probmap_', ic)

    def get_thresh_zstat(self, ic):
        return self._get_stat('thresh_zstat', ic)

    def get_tmodes(self):
        ns = self.smodes.shape[1]
        if ns > 1:
            # in multisession ICA melodic creates rank-1 approximation of a
            # timeseries from all sessions and stores them in the first column
            return self._tmodes.T[::ns+1]
        else:
            return self._tmodes.T

    def get_raw_tmodes(self):
        return self._tmodes

    # properties
    path     = property(fget=lambda self: self._rpath )
    ic       = property(fget=lambda self: np.rollaxis(self._ic.get_data(), -1))
    mask     = property(fget=lambda self: self._mask.get_data())
    nic      = property(fget=lambda self: self._icshape()[3])
    extent   = property(fget=lambda self: self._icshape()[:3])
    tmodes   = property(fget=get_tmodes)
    smodes   = property(fget=lambda self: self._smodes.T )
    icstats = property(fget=lambda self: self._icstats,
            doc="""Per component statistics.

The first two values (from a set of four per component) correspond to the
explained variance of the component in the set of extracted components and
two the total variance in the whole dataset.""")
    relvar_per_ic  = property(fget=lambda self: self._icstats[:, 0])
    truevar_per_ic = property(fget=lambda self: self._icstats[:, 1])
