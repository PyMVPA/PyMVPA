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

import numpy as N

from mvpa.clfs.distance import squared_euclidean_distance

if __debug__:
    from mvpa.base import debug

class Similarity(object):
    """Similarity function base class.

    """

    def __init__(self):
        pass

    def __repr__(self):
        return "Similarity()"

    def compute(self, data1, data2=None):
        raise NotImplementedError


class SingleDimensionSimilarity(Similarity):
    """exp(-|data1_j-data2_j|_2)
    """
    def __init__(self, d=0, **kwargs):
        Similarity.__init__(self, **kwargs)
        self.d = d

    def compute(self, data1, data2=None):
        if data2 == None: data2 = data1
        self.similarity_matrix = N.exp(-N.abs(data1[:,self.d],data2[:,self.d]))
        return self.similarity_matrix
    

class StreamlineSimilarity(Similarity):
    """Compute similarity between two streamlines.
    """

    def __init__(self, distance, gamma=1.0):
        Similarity.__init__(self)
        self.distance = distance
        self.gamma = gamma
        
    
    def compute(self, data1, data2=None):
        if data2 == None: data2 = data1
        self.distance_matrix = N.zeros((len(data1), len(data2)))
        ## debug("MAP",str(len(data1))+" - "+str(len(data2)))
        ## debug("MAP","data1:"+str(data1))
        ## debug("MAP","data2:"+str(data2))
        for i in range(len(data1)):
            for j in range(len(data2)):
                self.distance_matrix[i,j] = self.distance(data1[i],data2[j])
                ## debug("MAP","data1_"+str(i)+":"+str(data1[i])+" - data2_"+str(j)+":"+str(data2[j])+" - distance="+str(self.distance_matrix[i,j]))
                pass
            pass
        self.similarity_matrix = N.exp(-self.gamma*self.distance_matrix)
        return self.similarity_matrix
