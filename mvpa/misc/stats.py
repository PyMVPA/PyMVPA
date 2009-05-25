# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Little statistics helper"""

__docformat__ = 'restructuredtext'

from mvpa.base import externals

if externals.exists('scipy', raiseException=True):
    import scipy.stats as stats

import numpy as N
import copy

def chisquare(obs, exp=None):
    """Compute the chisquare value of a contingency table with arbitrary
    dimensions.

    If no expected frequencies are supplied, the total N is assumed to be
    equally distributed across all cells.

    Returns: chisquare-stats, associated p-value (upper tail)
    """
    obs = N.array(obs)

    # get total number of observations
    nobs = N.sum(obs)

    # if no expected value are supplied assume equal distribution
    if exp == None:
        exp = N.ones(obs.shape) * nobs / N.prod(obs.shape)

    # make sure to have floating point data
    exp = exp.astype(float)

    # compute chisquare value
    chisq = N.sum((obs - exp )**2 / exp)

    # return chisq and probability (upper tail)
    return chisq, stats.chisqprob(chisq, N.prod(obs.shape) - 1)


class DSMatrix(object):
    """DSMatrix allows for the creation of dissilimarity matrices using
       arbitrary distance metrics.
    """

    # metric is a string
    def __init__(self, data_vectors, metric='spearman'):
        """Initialize DSMatrix

        :Parameters:
          data_vectors : ndarray
             m x n collection of vectors, where m is the number of exemplars
             and n is the number of features per exemplar
          metric : string
             Distance metric to use (e.g., 'euclidean', 'spearman', 'pearson',
             'confusion')
        """
        # init members
        self.full_matrix = []
        self.u_triangle = None
        self.vector_form = None

        # this one we know straight away, so set it
        self.metric = metric

        # size of dataset (checking if we're dealing with a column vector only)
        num_exem = N.shape(data_vectors)[0]
        flag_1d = False
        # changed 4/26/09 to new way of figuring out if array is 1-D
        #if (isinstance(data_vectors, N.ndarray)):
        if (not(num_exem == N.size(data_vectors))):
            num_features = N.shape(data_vectors)[1]
        else:
            flag_1d = True
            num_features = 1

        # generate output (dissimilarity) matrix
        dsmatrix = N.mat(N.zeros((num_exem, num_exem)))

        if (metric == 'euclidean'):
            #print 'Using Euclidean distance metric...'
            # down rows
            for i in range(num_exem):
                # across columns
                for j in range(num_exem):
                    if (not(flag_1d)):
                        dsmatrix[i,j] = N.linalg.norm(data_vectors[i,:] - data_vectors[j,:])
                    else:
                        dsmatrix[i,j] = N.linalg.norm(data_vectors[i] - data_vectors[j])

        elif (metric == 'spearman'):
            #print 'Using Spearman rank-correlation metric...'
            # down rows
            for i in range(num_exem):
                # across columns
                for j in range(num_exem):
                    dsmatrix[i,j] = 1 - stats.spearmanr(data_vectors[i,:],data_vectors[j,:])[0]

        elif (metric == 'pearson'):
            #print 'Using Pearson correlation metric...'
            # down rows
            for i in range(num_exem):
                # across columns
                for j in range(num_exem):
                    dsmatrix[i, j] = 1 - stats.pearsonr(
                        data_vectors[i,:],data_vectors[j,:])[0]

        elif (metric == 'confusion'):
            #print 'Using confusion correlation metric...'
            # down rows
            for i in range(num_exem):
                # across columns
                for j in range(num_exem):
                    if (not(flag_1d)):
                        dsmatrix[i, j] = 1 - int(
                            N.floor(N.sum((
                                data_vectors[i, :] == data_vectors[j, :]
                                ).astype(N.int32)) / num_features))
                    else:
                        dsmatrix[i, j] = 1 - int(
                            data_vectors[i] == data_vectors[j])

        self.full_matrix = dsmatrix

    def getTriangle(self):
        # if we need to create the u_triangle representation, do so
        if (self.u_triangle == None):
            self.u_triangle = N.triu(self.full_matrix)

        return self.u_triangle

    # create the dissimilarity matrix on the (upper) triangle of the two
    # two dissimilarity matrices; we can just reuse the same dissimilarity
    # matrix code, but since it will return a matrix, we need to pick out
    # either dsm[0,1] or dsm[1,0]
    # note:  this is a bit of a kludge right now, but it's the only way to solve
    # certain problems:
    #  1.  Set all 0-valued elements in the original matrix to -1 (an impossible
    #        value for a dissimilarity matrix)
    #  2.  Find the upper triangle of the matrix
    #  3.  Create a vector from the upper triangle, but only with the
    #      elements whose absolute value is greater than 0 -- this
    #      will keep everything from the original matrix that wasn't
    #      part of the zero'ed-out portion when we took the upper
    #      triangle
    #  4.  Set all the -1-valued elements in the vector to 0 (their
    #      original value)
    #  5.  Cast to numpy array
    def getVectorForm(self):
        if (not(self.vector_form == None)):
            return self.vector_form

        orig_dsmatrix = copy.deepcopy(self.getFullMatrix())

        orig_dsmatrix[orig_dsmatrix == 0] = -1

        orig_tri = N.triu(orig_dsmatrix)

        self.vector_form = orig_tri[abs(orig_tri) > 0]

        self.vector_form[self.vector_form == -1] = 0

        self.vector_form = N.asarray(self.vector_form)
        self.vector_form = self.vector_form[0]

        return self.vector_form

    # XXX is there any reason to have these get* methods
    #     instead of plain access to full_matrix and method?
    def getFullMatrix(self):
        return self.full_matrix

    def getMetric(self):
        return self.metric
