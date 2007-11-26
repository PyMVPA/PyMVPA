#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Utility class to compute the transfer error of classifiers."""

__docformat__ = 'restructuredtext'


from mvpa.misc.errorfx import MeanMismatchErrorFx
from mvpa.misc.state import State

class TransferError(State):
    """Compute the transfer error of a (trained) classifier on a dataset.

    The actual error value is computed using a customizable error function.
    Optionally the classifier can be training by passing an additional
    training dataset to the __call__() method.
    """
    def __init__(self, clf, errorfx=MeanMismatchErrorFx()):
        """Cheap initialization.

        Parameters
        ----------
        - `clf`: Classifier instance.
                 Either trained or untrained.
        - `errorfx`: Functor that computes a scalar error value from the
                     vectors of desired and predicted values (e.g. subclass
                     of ErrorFx)
        """
        State.__init__(self)
        self.__clf = clf
        self.__errorfx = errorfx

        self._registerState('confusion')
        """TODO Think that labels might be also symbolic thus can't directly
                be indicies of the array
        """

    def __call__(self, testdata, trainingdata=None):
        """Compute the transfer error for a certain test dataset.

        If `trainingdata` is not `None` the classifier is trained using the
        provided dataset before computing the transfer error. Otherwise the
        classifier is used in it's current state to make the predictions on
        the test dataset.

        Returns a scalar value of the transfer error.
        """
        if not trainingdata == None:
            self.__clf.train(trainingdata)

        predictions = self.__clf.predict(testdata.samples)

        # compute confusion matrix
        # self['confusion'] = ....
        # TODO

        # compute error from desired and predicted values
        error = self.__errorfx(predictions,
                               testdata.labels)

        return error
