# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data mapper"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.base.dochelpers import enhancedDocString
from mvpa.mappers.base import ProjectionMapper

import mvpa.base.externals as externals
if externals.exists('mdp', raiseException=True):
    from mdp.nodes import FastICANode, CuBICANode


class ICAMapper(ProjectionMapper):
    """Mapper to project data onto ICA components estimated from some dataset.

    After the mapper has been instantiated, it has to be train first. The ICA
    mapper only handles 2D data matrices.
    """
    def __init__(self, algorithm='cubica', transpose=False, **kwargs):
        ProjectionMapper.__init__(self, **kwargs)

        self._algorithm = algorithm
        self._transpose = transpose

    __doc__ = enhancedDocString('ICAMapper', locals(), ProjectionMapper)


    def _train(self, dataset):
        """Determine the projection matrix onto the components from
        a 2D samples x feature data matrix.
        """
        white_param = {}

        # more features than samples? -> rank deficiancy
        # if not tranposing the data, MDP has to do SVD prior to ICA
        if dataset.samples.shape[1] > dataset.samples.shape[0] \
           and not self._transpose:
            white_param['svd'] = True

        if self._algorithm == 'fastica':
            node = FastICANode(white_parm=white_param,
                               dtype=dataset.samples.dtype)
        elif self._algorithm == 'cubica':
            node = CuBICANode(white_parm=white_param,
                              dtype=dataset.samples.dtype)
        else:
            raise NotImplementedError

#            node.train(dataset.samples.T)
#            self._proj = dataset.samples.T * N.asmatrix(node.get_projmatrix())
#            print self._proj.shape
#        else:
        node.train(dataset.samples)
        self._proj = N.asmatrix(node.get_projmatrix())
        self._recon = N.asmatrix(node.get_recmatrix())
