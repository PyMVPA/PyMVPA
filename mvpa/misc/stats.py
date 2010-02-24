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

import numpy as np
import copy

def chisquare(obs, exp=None):
    """Compute the chisquare value of a contingency table with arbitrary
    dimensions.

    If no expected frequencies are supplied, the total N is assumed to be
    equally distributed across all cells.

    Returns
    -------
    tuple
     chisquare-stats, associated p-value (upper tail)
    """
    obs = np.array(obs)

    # get total number of observations
    nobs = np.sum(obs)

    # if no expected value are supplied assume equal distribution
    if exp == None:
        exp = np.ones(obs.shape) * nobs / np.prod(obs.shape)

    # make sure to have floating point data
    exp = exp.astype(float)

    # compute chisquare value
    chisq = np.sum((obs - exp )**2 / exp)

    # return chisq and probability (upper tail)
    return chisq, stats.chisqprob(chisq, np.prod(obs.shape) - 1)


class DSMatrix(object):
    """DSMatrix allows for the creation of dissilimarity matrices using
       arbitrary distance metrics.
    """

    # metric is a string
    def __init__(self, data_vectors, metric='spearman'):
        """Initialize DSMatrix

        Parameters
        ----------
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
        num_exem = np.shape(data_vectors)[0]
        flag_1d = False
        # changed 4/26/09 to new way of figuring out if array is 1-D
        #if (isinstance(data_vectors, np.ndarray)):
        if (not(num_exem == np.size(data_vectors))):
            num_features = np.shape(data_vectors)[1]
        else:
            flag_1d = True
            num_features = 1

        # generate output (dissimilarity) matrix
        dsmatrix = np.mat(np.zeros((num_exem, num_exem)))

        if (metric == 'euclidean'):
            #print 'Using Euclidean distance metric...'
            # down rows
            for i in range(num_exem):
                # across columns
                for j in range(num_exem):
                    if (not(flag_1d)):
                        dsmatrix[i, j] = np.linalg.norm(
                            data_vectors[i, :] - data_vectors[j, :])
                    else:
                        dsmatrix[i, j] = np.linalg.norm(
                            data_vectors[i] - data_vectors[j])

        elif (metric == 'spearman'):
            #print 'Using Spearman rank-correlation metric...'
            # down rows
            for i in range(num_exem):
                # across columns
                for j in range(num_exem):
                    dsmatrix[i, j] = 1 - stats.spearmanr(
                        data_vectors[i,:], data_vectors[j,:])[0]

        elif (metric == 'pearson'):
            #print 'Using Pearson correlation metric...'
            # down rows
            for i in range(num_exem):
                # across columns
                for j in range(num_exem):
                    dsmatrix[i, j] = 1 - stats.pearsonr(
                        data_vectors[i,:], data_vectors[j,:])[0]

        elif (metric == 'confusion'):
            #print 'Using confusion correlation metric...'
            # down rows
            for i in range(num_exem):
                # across columns
                for j in range(num_exem):
                    if (not(flag_1d)):
                        dsmatrix[i, j] = 1 - int(
                            np.floor(np.sum((
                                data_vectors[i, :] == data_vectors[j, :]
                                ).astype(np.int32)) / num_features))
                    else:
                        dsmatrix[i, j] = 1 - int(
                            data_vectors[i] == data_vectors[j])

        self.full_matrix = dsmatrix

    ##REF: Name was automagically refactored
    def get_triangle(self):
        # if we need to create the u_triangle representation, do so
        if (self.u_triangle is None):
            self.u_triangle = np.triu(self.full_matrix)

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
    ##REF: Name was automagically refactored
    def get_vector_form(self):
        if (self.vector_form is not None):
            return self.vector_form

        orig_dsmatrix = copy.deepcopy(self.get_full_matrix())

        orig_dsmatrix[orig_dsmatrix == 0] = -1

        orig_tri = np.triu(orig_dsmatrix)

        vector_form = orig_tri[abs(orig_tri) > 0]

        vector_form[vector_form == -1] = 0

        vector_form = np.asarray(vector_form)
        self.vector_form = vector_form[0]

        return self.vector_form

    # XXX is there any reason to have these get* methods
    #     instead of plain access to full_matrix and method?
    ##REF: Name was automagically refactored
    def get_full_matrix(self):
        return self.full_matrix

    ##REF: Name was automagically refactored
    def get_metric(self):
        return self.metric
