# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Emanuele Olivetti <emanuele@relativita.com>
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Similarity functions for prototype-based projection."""

import numpy as np

from mvpa2.clfs.distance import squared_euclidean_distance

if __debug__:
    from mvpa2.base import debug

class Similarity(object):
    """Similarity function base class.

    """

    def __repr__(self):
        return "Similarity()"

    def computed(self, data1, data2=None):
        raise NotImplementedError


class SingleDimensionSimilarity(Similarity):
    """TODO

    .. math:: e^{(-|data1_j - data2_j|_2)}

    """
    def __init__(self, d=0, **kwargs):
        """
        Parameters
        ----------
        d : int
          Dimension (feature) across which to compute similarity
        **kwargs
          Passed to Similarity
        """
        Similarity.__init__(self, **kwargs)
        self.d = d

    def computed(self, data1, data2=None):
        if data2 == None: data2 = data1
        self.similarity_matrix = np.exp(-np.abs(data1[:, self.d],
                                              data2[:, self.d]))
        return self.similarity_matrix


class StreamlineSimilarity(Similarity):
    """Compute similarity between two streamlines.
    """

    def __init__(self, distance, gamma=1.0):
        """
        Parameters
        ----------
        distance : func
          Distance measure
        gamma : float
          Exponent coefficient
        """
        Similarity.__init__(self)
        self.distance = distance
        self.gamma = gamma


    def computed(self, data1, data2=None):
        if data2 == None:
            data2 = data1
        self.distance_matrix = np.zeros((len(data1), len(data2)))

        # setup helpers to pull out content of object-type arrays
        if isinstance(data1, np.ndarray) and np.issubdtype(data1.dtype, np.object):
            d1extract = _pass_obj_content
        else:
            d1extract = lambda x: x

        if isinstance(data2, np.ndarray) and np.issubdtype(data2.dtype, np.object):
            d2extract = _pass_obj_content
        else:
            d2extract = lambda x: x

        # TODO: use np.fromfunction
        for i, d1 in enumerate(data1):
            for j, d2 in enumerate(data2):
                self.distance_matrix[i,j] = self.distance(d1extract(data1[i]),
                                                          d2extract(data2[j]))

        self.similarity_matrix = np.exp(-self.gamma*self.distance_matrix)
        return self.similarity_matrix


def _pass_obj_content(data):
    """Helper that can be used to return the content of a single-element
    array of type 'object' to access its real content.
    """
    return data[0]
