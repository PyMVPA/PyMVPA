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

import sys
# not worthy of externals checking
_dict_has_key = sys.version_info >= (2, 5)

import numpy as np

from mvpa2.base import warning
from mvpa2.datasets.base import Dataset
from mvpa2.misc.support import indent_doc
from mvpa2.base.state import ConditionalAttribute

from mvpa2.clfs.base import Classifier, accepts_dataset_as_samples
from mvpa2.clfs.distance import squared_euclidean_distance

__all__ = [ 'kNN' ]

if __debug__:
    from mvpa2.base import debug


class kNN(Classifier):
    """
    k-Nearest-Neighbour classifier.

    This is a simple classifier that bases its decision on the distances
    between the training dataset samples and the test sample(s). Distances
    are computed using a customizable distance function. A certain number
    (`k`)of nearest neighbors is selected based on the smallest distances
    and the labels of this neighboring samples are fed into a voting
    function to determine the labels of the test sample.

    Training a kNN classifier is extremely quick, as no actual training
    is performed as the training dataset is simply stored in the
    classifier. All computations are done during classifier prediction.

    Ties
    ----

    In case if voting procedure results in a tie, it is broken by
    choosing a class with minimal mean distance to the corresponding
    k-neighbors.

    Notes
    -----
    If enabled, kNN stores the votes per class in the 'values' state after
    calling predict().

    """

    distances = ConditionalAttribute(enabled=False,
        doc="Distances computed for each sample")


    __tags__ = ['knn', 'non-linear', 'binary', 'multiclass', 'oneclass']

    def __init__(self, k=2, dfx=squared_euclidean_distance,
                 voting='weighted', **kwargs):
        """
        Parameters
        ----------
        k : unsigned integer
          Number of nearest neighbours to be used for voting.
        dfx : functor
          Function to compute the distances between training and test samples.
          Default: squared euclidean distance
        voting : str
          Voting method used to derive predictions from the nearest neighbors.
          Possible values are 'majority' (simple majority of classes
          determines vote) and 'weighted' (votes are weighted according to the
          relative frequencies of each class in the training data).
        **kwargs
          Additional arguments are passed to the base class.
        """

        # init base class first
        Classifier.__init__(self, **kwargs)

        self.__k = k
        self.__dfx = dfx
        self.__voting = voting
        self.__data = None
        self.__weights = None


    def __repr__(self, prefixes=[]): # pylint: disable-msg=W0102
        """Representation of the object
        """
        return super(kNN, self).__repr__(
            ["k=%d" % self.__k, "dfx=%s" % self.__dfx,
             "voting=%s" % repr(self.__voting)]
            + prefixes)


    ## def __str__(self):
    ##     return "%s\n data: %s" % \
    ##         (Classifier.__str__(self), indent_doc(self.__data))


    def _train(self, data):
        """Train the classifier.

        For kNN it is degenerate -- just stores the data.
        """
        self.__data = data
        labels = data.sa[self.get_space()].value
        uniquelabels = data.sa[self.get_space()].unique
        Nuniquelabels = len(uniquelabels)

        if __debug__:
            if str(data.samples.dtype).startswith('uint') \
                or str(data.samples.dtype).startswith('int'):
                warning("kNN: input data is in integers. " + \
                        "Overflow on arithmetic operations might result in"+\
                        " errors. Please convert dataset's samples into" +\
                        " floating datatype if any error is reported.")
        if self.__voting == 'weighted':
            self.__labels = labels.copy()
            Nlabels = len(labels)

            # TODO: To get proper speed up for the next line only,
            #       histogram should be computed
            #       via sorting + counting "same" elements while reducing.
            #       Guaranteed complexity is NlogN whenever now it is N^2
            # compute the relative proportion of samples belonging to each
            # class (do it in one loop to improve speed and reduce readability
            weights = \
                [ 1.0 - ((labels == label).sum() / Nlabels) \
                    for label in uniquelabels ]
            self.__weights = dict(zip(uniquelabels, weights))
        else:
            self.__weights = None

        # create dictionary with an item for each condition
        self.__votes_init = dict(zip(uniquelabels,
                                     [0] * Nuniquelabels))


    @accepts_dataset_as_samples
    def _predict(self, data):
        """Predict the class labels for the provided data.

        Returns a list of class labels (one for each data sample).
        """
        # make sure we're talking about arrays
        data = np.asanyarray(data)

        targets_sa_name = self.get_space()
        targets_sa = self.__data.sa[targets_sa_name]
        labels = targets_sa.value
        uniquelabels = targets_sa.unique

        # checks only in debug mode
        if __debug__:
            if not data.ndim == 2:
                raise ValueError, "Data array must be two-dimensional."

            if not data.shape[1] == self.__data.nfeatures:
                raise ValueError, "Length of data samples (features) does " \
                                  "not match the classifier."

        # compute the distance matrix between training and test data with
        # distances stored row-wise, i.e. distances between test sample [0]
        # and all training samples will end up in row 0
        dists = self.__dfx(self.__data.samples, data).T
        if self.ca.is_enabled('distances'):
            # .sa.copy() now does deepcopying by default
            self.ca.distances = Dataset(dists, fa=self.__data.sa.copy())

        # determine the k nearest neighbors per test sample
        knns = dists.argsort(axis=1)[:, :self.__k]

        # predictions and votes for all samples
        all_votes, predictions = [], []
        for inns, nns in enumerate(knns):
            votes = self.__votes_init.copy()
            # TODO: optimize!
            for nn in nns:
                votes[labels[nn]] += 1

            # optionally weight votes
            if self.__voting == 'majority':
                pass
            elif self.__voting == 'weighted':
                # TODO: optimize!
                for ul in uniquelabels:
                    votes[ul] *= self.__weights[ul]
            else:
                raise ValueError, "kNN told to perform unknown voting '%s'." \
                      % self.__voting

            # reverse dictionary items and sort them to get the
            # winners
            # It would be more expensive than just to look for
            # the maximum, but this piece should be the least
            # cpu-intensive while distances computation should consume
            # the most. Also it would allow to look and break the ties
            votes_reversed = sorted([(v, k) for k, v in votes.iteritems()],
                                    reverse=True)
            # check for ties
            max_vote, max_vote_label = votes_reversed[0]

            if len(votes_reversed) > 1 and max_vote == votes_reversed[1][0]:
                # figure out all ties and break them based on the mean
                # distance
                # TODO: theoretically we could break out of the loop earlier
                ties = [x[1] for x in votes_reversed if x[0] == max_vote]

                # compute mean distances to the corresponding clouds
                # restrict analysis only to k-nn's
                nns_labels = labels[nns]
                nns_dists = dists[inns][nns]
                ties_dists = [np.mean(nns_dists[nns_labels == t]) for t in ties]
                max_vote_label = ties[np.argmin(ties_dists)]
                if __debug__:
                    debug('KNN',
                          'Ran into the ties: %s with votes: %s, dists: %s, max_vote %r',
                          (ties, votes_reversed, ties_dists, max_vote_label))

            all_votes.append(votes)
            predictions.append(max_vote_label)

        # store the predictions in the state. Relies on State._setitem to do
        # nothing if the relevant state member is not enabled
        self.ca.predictions = predictions
        self.ca.estimates = all_votes # np.array([r[1] for r in results])

        return predictions

    def _untrain(self):
        """Reset trained state"""
        self.__data = None
        self.__weights = None
        super(kNN, self)._untrain()

    dfx = property(fget=lambda self: self.__dfx)
