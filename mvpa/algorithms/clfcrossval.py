#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Cross-validate a classifier on a dataset"""

import numpy as N

from mvpa.algorithms.datameasure import DataMeasure
from mvpa.misc.errorfx import MeanMatchErrorFx
from mvpa.datasets.splitter import NoneSplitter


class ClfCrossValidation(DataMeasure):
    """ Cross validate a classifier on datasets generate by a splitter from a
    source dataset.

    Arbitrary performance/error values can be computed by specifying and error
    function (used to compute an error value for each cross-validation fold) and
    a combiner function that aggregates all computed error values across
    cross-validation folds.
    """
    def __init__(self,
                 clf,
                 splitter=NoneSplitter,
                 errorfx=MeanMatchErrorFx(),
                 combinerfx=N.mean):
        """
        Cheap initialization.

        Parameters:
            clf        - classifier instance
            splitter   - splitter instance used to split the dataset for
                         cross-validation folds. By convention the first dataset
                         in the tuple returned by the splitter is used to train
                         the provided classifier. If the first element is 'None'
                         no training is performed. The second dataset is used to
                         generate predictions with the (trained) classifier.
            errorfx    - function that computes a scalar error value from the
                         vectors of desired and predicted values (subclass of
                         ErrorFx)
            combinerfx - function that is used to aggregate the error values of
                         all cross-validation folds
        """
        DataMeasure.__init__(self)

        self.__splitter = splitter
        self.__clf = clf
        self.__errorfx = errorfx
        self.__combinerfx = combinerfx

    def __repr__(self):
        """ String summary over the object
        """
        return """ClfCrossValidation:
 splitter: %s
 classifier: %s
 errorfx: %s
 combinerfx: %s""" % (`self.__splitter`, `self.__clf`,
                     `self.__errorfx`, `self.__combinerfx`)


    def __call__(self, dataset, callbacks=[]):
        """ Perform cross-validation on a dataset.

        'dataset' is passed to the splitter instance and serves as the source
        dataset to generate split for the single cross-validation folds.
        """
        # store the results of the splitprocessor
        results = []

        # splitter
        for split in self.__splitter(dataset):
            # only train classifier if splitter provides something in first
            # element of tuple
            if not split[0] == None:
                self.__clf.train( split[0] )

            # compute error from desired and predicted values
            error = self.__errorfx(self.__clf.predict(split[1].samples),
                                   split[1].labels)
            results.append(error)

            # XXX add callbacks

        return self.__combinerfx(results)
