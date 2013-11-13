# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Mappers for Dataset scaling."""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base.node import Node
from mvpa2.mappers.base import Mapper, accepts_dataset_as_samples
from mvpa2.base.dochelpers import _str, _repr_attrs
from mvpa2.generators.splitters import mask2slice


class SensitivityBasedFeatureScaling(Mapper):
    """Mapper that weights the features of a dataset.
       This amounts to multiplying by a diagonal matrix on the right.
       The motivation behind this is to weight features based on an arbitrary
       senstivity map instead of eliminating features.  Thus as an option to
       replace feature selection.
    """
    
    def __init__(self, sensitivity_analyzer, norm='l2',**kwargs):
        """
        Parameters
        ----------
        sensitivity_analyzer    :  Callable Featurewise Measure 
        norm                    :  Normalize sensitivity map before scaling,
                                   Default 'l2' divides sensitity map by its
                                   l2 norm. 
        """
        super(type(self),self).__init__(**kwargs)
        self._sensitivity_map = None
        self._sens_anal = sensitivity_analyzer
        self._norm = norm
        
    def _forward_dataset(self,ds):
        mds = ds.copy(deep=False)
        scal = self._sensitivity_map.samples.flatten()
        scal[scal==0] = 1.*10e-16 # no zeros allowed
        if self._norm == 'l2':
            scal = scal/np.linalg.norm(scal)
        
        mds.samples = mds.samples * scal
        return mds

    def _forward_data(self, data):
        mds = ds.copy(deep=False)
        scal = self._sensitivity_map.samples.flatten()
        scal[scal==0] = 1.*10e-16 # no zeros allowed
        if self._norm == 'l2':
            scal = scal/np.linalg.norm(scal)
        mds = mds * scal

        return mds

    def _train(self,ds):
        self._sensitivity_map = self._sens_anal(ds)
        super(type(self), self)._train(ds)

    def _untrain(self):
        self._sensitivity_map = None
        super(type(self), self)._untrain()




