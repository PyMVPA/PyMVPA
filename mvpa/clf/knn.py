#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""k-Nearest-Neighbour classification"""

import numpy as N

from mvpa.misc import verbose
from mvpa.misc.support import indentDoc


class kNN:
    """ k-nearest-neighbour classification.
    """

    __warned = False

    def __init__(self, k=2):
        """
        Parameters:
          k:       number of nearest neighbours to be used for voting
        """
        self.__k = k
        # XXX So is the voting function fixed forever?
        self.__votingfx = self.getWeightedVote
        self.__data = None


    def __repr__(self):
        """ String summary over the object
        """
        return """kNN / k=%d
 votingfx: TODO
 data: %s""" % (self.__k, indentDoc(self.__data))


    def train( self, data ):
        """ Train the classifier.

        For kNN it is degenerate -- just stores the data.
        """
        self.__data = data
        if __debug__:
            if not kNN.__warned and \
                str(data.samples.dtype).startswith('uint') \
                or str(data.samples.dtype).startswith('int'):
                kNN.__warned = True
                verbose(1, "kNN: input data is in integers. " + \
                        "Overflow on arithmetic operations might result in"+\
                        " errors. Please convert dataset's samples into" +\
                        " floating datatype if any error is reported.")
        self.__weights = None

    def predict(self, data):
        """ Predict the class labels for the provided data.

        Returns a list of class labels (one for each data sample).
        """
        # make sure we're talking about arrays
        data = N.array( data )

        if not data.ndim == 2:
            raise ValueError, "Data array must be two-dimensional."

        if not data.shape[1] == self.__data.nfeatures:
            raise ValueError, "Length of data samples (features) does " \
                              "not match the classifier."

        # predicted class labels will go here
        predicted = []

        # for all test pattern
        for p in data:
            # calc the euclidean distance of the pattern vector to all
            # patterns in the training data
            dists = N.sqrt(
                        N.sum(
                            (self.__data.samples - p )**2, axis=1
                            )
                        )
            # get the k nearest neighbours from the sorted list of distances
            knn = dists.argsort()[:self.__k]

            # finally get the class label
            predicted.append( self.__votingfx(knn) )

        return predicted


    def getMajorityVote(self, knn_ids):
        """TODO docstring
        """

        labels = self.__data.labels
        uniquelabels = self.__data.uniquelabels

        # create dictionary with an item for each condition
        votes = dict( zip ( uniquelabels, [0]*len(uniquelabels) ) )

        # add 1 to a certain condition per NN
        for nn in knn_ids:
            votes[labels[nn]] += 1

        # find the condition with most votes
        best_cond = None
        most_votes = None
        for cond, vote in votes.iteritems():
            if best_cond is None or vote > most_votes:
                best_cond = cond
                most_votes = vote

        return best_cond


    def getWeightedVote(self, knn_ids):
        """TODO docstring
        """

        # Lazy evaluation
        if self.__weights is None:
            #
            # It seemed to Yarik that this has to be evaluated just once per
            # training dataset.
            #
            self.__labels = self.__data.labels
            Nlabels = len(self.__labels)
            uniquelabels = self.__data.uniquelabels
            Nuniquelabels = len(uniquelabels)

            # create dictionary with an item for each condition
            self.__votes_init = dict( zip ( uniquelabels, [0]*Nuniquelabels ) )

            # TODO: To get proper speed up for the next line only,
            #       histogram should be computed
            #       via sorting + counting "same" elements while reducing.
            #       Guaranteed complexity is NlogN whenever now it is N^2
            self.__weights = {}

            for label in uniquelabels:
                self.__weights[label] = 0.0

            for label in self.__labels:
                self.__weights[label] += 1.0

            for k, v in self.__weights.iteritems():
                self.__weights[k] = 1.0 - (v / Nlabels)

        votes = self.__votes_init.copy()

        for nn in knn_ids:
            votes[self.__labels[nn]] += self.__weights[self.__labels[nn]]

        # find the condition with most votes
        best_cond = None
        most_votes = None
        for cond, vote in votes.iteritems():
            if best_cond is None or vote > most_votes:
                best_cond = cond
                most_votes = vote

        return best_cond
