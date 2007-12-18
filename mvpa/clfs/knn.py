#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""k-Nearest-Neighbour classifier."""

__docformat__ = 'restructuredtext'


import numpy as N

from mvpa.misc import verbose
from mvpa.misc.support import indentDoc
from mvpa.clfs.classifier import Classifier


class kNN(Classifier):
    """k-nearest-neighbour classifier.

    If enabled it stores the votes per class in the 'values' state after
    calling predict().
    """

    __warned = False

    def __init__(self, k=2):
        """
        Parameters:
          k:       number of nearest neighbours to be used for voting
        """
        # init base class first
        Classifier.__init__(self)

        self.__k = k
        # XXX So is the voting function fixed forever?
        self.__votingfx = self.getWeightedVote
        self.__data = None


    def __repr__(self):
        """String summary over the object
        """
        return """kNN / k=%d
 votingfx: TODO
 data: %s""" % (self.__k, indentDoc(self.__data))


    def train( self, data ):
        """Train the classifier.

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

        # create dictionary with an item for each condition
        uniquelabels = data.uniquelabels
        self.__votes_init = dict(zip(uniquelabels, 
                                     [0] * len(uniquelabels)))


    def predict(self, data):
        """Predict the class labels for the provided data.

        Returns a list of class labels (one for each data sample).
        """
        # make sure we're talking about arrays
        data = N.array(data)

        if not data.ndim == 2:
            raise ValueError, "Data array must be two-dimensional."

        if not data.shape[1] == self.__data.nfeatures:
            raise ValueError, "Length of data samples (features) does " \
                              "not match the classifier."

        # predicted class labels will go here
        predicted = []
        votes = []

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
            prediction, vote = self.__votingfx(knn)
            predicted.append(prediction)
            votes.append(vote)

        # store the predictions in the state. Relies on State._setitem to do
        # nothing if the relevant state member is not enabled
        self['predictions'] = predicted
        self['values'] = votes

        return predicted


    def getMajorityVote(self, knn_ids):
        """Simple voting by choosing the majority of class neighbours.
        """

        uniquelabels = self.__data.uniquelabels

        # translate knn ids into class labels
        knn_labels = N.array([ self.__data.labels[nn] for nn in knn_ids ])

        # number of occerences for each unique class in kNNs
        votes = self.__votes_init.copy()
        for nn in knn_ids:
            votes[self.__labels[nn]] += 1

        # find the class with most votes
        # return votes as well to store them in the state
        return uniquelabels[N.array(votes).argmax()], \
               votes


    def getWeightedVote(self, knn_ids):
        """Vote with classes weighted by the number of samples per class.
        """
        uniquelabels = self.__data.uniquelabels

        # Lazy evaluation
        if self.__weights is None:
            #
            # It seemed to Yarik that this has to be evaluated just once per
            # training dataset.
            #
            self.__labels = self.__data.labels
            Nlabels = len(self.__labels)
            Nuniquelabels = len(uniquelabels)

            # TODO: To get proper speed up for the next line only,
            #       histogram should be computed
            #       via sorting + counting "same" elements while reducing.
            #       Guaranteed complexity is NlogN whenever now it is N^2
            # compute the relative proportion of samples belonging to each
            # class (do it in one loop to improve speed and reduce readability
            self.__weights = \
                [ 1.0 - ((self.__labels == label).sum() / Nlabels) \
                    for label in uniquelabels ]
            self.__weights = dict(zip(uniquelabels, self.__weights))


        # number of occerences for each unique class in kNNs
        votes = self.__votes_init.copy()
        for nn in knn_ids:
            votes[self.__labels[nn]] += 1

        # weight votes
        votes = [ self.__weights[ul] * votes[ul] for ul in uniquelabels]

        # find the class with most votes
        # return votes as well to store them in the state
        return uniquelabels[N.array(votes).argmax()], \
               votes
