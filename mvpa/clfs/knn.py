# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""k-Nearest-Neighbour classifier."""

__docformat__ = 'restructuredtext'


import numpy as N

from mvpa.base import warning
from mvpa.misc.support import indentDoc
from mvpa.clfs.base import Classifier
from mvpa.base.dochelpers import enhancedDocString
from mvpa.clfs.distance import squared_euclidean_distance

if __debug__:
    from mvpa.base import debug


class kNN(Classifier):
    """k-Nearest-Neighbour classifier.

    This is a simple classifier that bases its decision on the distances between
    the training dataset samples and the test sample(s). Distances are computed
    using a customizable distance function. A certain number (`k`)of nearest
    neighbors is selected based on the smallest distances and the labels of this
    neighboring samples are fed into a voting function to determine the labels
    of the test sample.

    Training a kNN classifier is extremely quick, as no actuall training is
    performed as the training dataset is simply stored in the classifier. All
    computations are done during classifier prediction.

    .. note::
      If enabled, kNN stores the votes per class in the 'values' state after
      calling predict().
    """

    _clf_internals = ['knn', 'non-linear', 'binary', 'multiclass',
                      'notrain2predict' ]

    def __init__(self, k=2, dfx=squared_euclidean_distance,
                 voting='weighted', **kwargs):
        """
        :Parameters:
          k: unsigned integer
            Number of nearest neighbours to be used for voting.
          dfx: functor
            Function to compute the distances between training and test samples.
            Default: squared euclidean distance
          voting: str
            Voting method used to derive predictions from the nearest neighbors.
            Possible values are 'majority' (simple majority of classes
            determines vote) and 'weighted' (votes are weighted according to the
            relative frequencies of each class in the training data).
          **kwargs:
            Additonal arguments are passed to the base class.
        """

        # init base class first
        Classifier.__init__(self, **kwargs)

        self.__k = k
        self.__dfx = dfx
        self.__voting = voting
        self.__data = None


    def __repr__(self):
        """Representation of the object
        """
        return "kNN(k=%d, dfx=%s enable_states=%s)" % \
            (self.__k, self.__dfx, str(self.states.enabled))


    def __str__(self):
        return "%s\n data: %s" % \
            (Classifier.__str__(self), indentDoc(self.__data))


    def _train(self, data):
        """Train the classifier.

        For kNN it is degenerate -- just stores the data.
        """
        self.__data = data
        if __debug__:
            if str(data.samples.dtype).startswith('uint') \
                or str(data.samples.dtype).startswith('int'):
                warning("kNN: input data is in integers. " + \
                        "Overflow on arithmetic operations might result in"+\
                        " errors. Please convert dataset's samples into" +\
                        " floating datatype if any error is reported.")
        self.__weights = None

        # create dictionary with an item for each condition
        uniquelabels = data.uniquelabels
        self.__votes_init = dict(zip(uniquelabels,
                                     [0] * len(uniquelabels)))


    def _predict(self, data):
        """Predict the class labels for the provided data.

        Returns a list of class labels (one for each data sample).
        """
        # make sure we're talking about arrays
        data = N.asarray(data)

        # checks only in debug mode
        if __debug__:
            if not data.ndim == 2:
                raise ValueError, "Data array must be two-dimensional."

            if not data.shape[1] == self.__data.nfeatures:
                raise ValueError, "Length of data samples (features) does " \
                                  "not match the classifier."

        # compute the distance matrix between training and test data with
        # distances stored row-wise, ie. distances between test sample [0]
        # and all training samples will end up in row 0
        dists = self.__dfx(self.__data.samples, data).T

        # determine the k nearest neighbors per test sample
        knns = dists.argsort(axis=1)[:, :self.__k]

        # predicted class labels will go here
        predicted = []
        votes = []

        if self.__voting == 'majority':
            vfx = self.getMajorityVote
        elif self.__voting == 'weighted':
            vfx = self.getWeightedVote
        else:
            raise ValueError, "kNN told to perform unknown voting '%s'." % self.__voting

        # perform voting
        results = [vfx(knn) for knn in knns]

        # extract predictions
        predicted = [r[0] for r in results]

        # store the predictions in the state. Relies on State._setitem to do
        # nothing if the relevant state member is not enabled
        self.predictions = predicted
        self.values = [r[1] for r in results]

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
        return uniquelabels[N.asarray(votes).argmax()], \
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
        return uniquelabels[N.asarray(votes).argmax()], \
               votes


    def untrain(self):
        """Reset trained state"""
        self.__data = None
        super(kNN, self).untrain()
