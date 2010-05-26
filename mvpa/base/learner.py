# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Implementation of a common trainable processing object (Learner)."""

__docformat__ = 'restructuredtext'

import time
from mvpa.base.node import Node
from mvpa.base.state import ConditionalAttribute
from mvpa.base.types import is_datasetlike

if __debug__:
    from mvpa.base import debug


class LearnerError(Exception):
    """Base class for exceptions thrown by the learners

    (classifiers, regressions)
    """
    pass


class DegenerateInputError(LearnerError):
    """Learner exception thrown if input data is bogus

    i.e. no features or samples
    """
    pass


class FailedToTrainError(LearnerError):
    """Learner exception thrown if training failed"""
    pass


class FailedToPredictError(LearnerError):
    """Learner exception if it fails to predictions.

    Usually happens if it was trained on degenerate data but without any
    complaints.
    """
    pass



class Learner(Node):
    """Common trainable processing object.

    A `Learner` is a `Node` that can (maybe has to) be trained on a dataset,
    before it can perform its function.
    """

    training_time = ConditionalAttribute(enabled=True,
        doc="Time (in seconds) it took to train the learner")


    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
          All arguments are passed to the baseclass.
        """
        Node.__init__(self, **kwargs)


    def train(self, ds):
        """
        The default implementation calls ``_pretrain()``, ``_train()``, and
        finally ``_posttrain()``.

        Parameters
        ----------
        ds: Dataset
          Training dataset.

        Returns
        -------
        None
        """
        got_ds = is_datasetlike(ds)

        # TODO remove first condition if all Learners get only datasets
        if got_ds and (ds.nfeatures == 0 or len(ds) == 0):
            raise DegenerateInputError(
                    "Cannot train classifier on degenerate data %s" % ds)
        if __debug__:
            debug("LRN", "Training learner %(lrn)s on dataset %(dataset)s",
                  msgargs={'lrn':self, 'dataset': ds})

        self._pretrain(ds)

        # remember the time when started training
        t0 = time.time()

        if got_ds:
            # things might have happened during pretraining
            if ds.nfeatures > 0:
                result = self._train(ds)
            else:
                warning("Trying to train on dataset with no features present")
                if __debug__:
                    debug("LRN",
                          "No features present for training, no actual training " \
                          "is called")
                result = None
        else:
            # in this case we claim to have no idea and simply try to train
            result = self._train(ds)

        # store timing
        self.ca.training_time = time.time() - t0

        # and post-proc
        result = self._posttrain(ds)


    def untrain(self):
        """Reverts changes in the state of this node caused by previous training
        """
        # TODO evaluate whether this should also reset the nodes collections, or
        # whether that should be done by a more general reset() method
        pass


    def _pretrain(self, ds):
        """Preparations prior training.

        By default, does nothing.

        Parameters
        ----------
        ds: Dataset
          Original training dataset.

        Returns
        -------
        None
        """
        pass


    def _train(self, ds):
        raise NotImplementedError


    def _posttrain(self, ds):
        """Finalizing the training.

        By default, does nothing.

        Parameters
        ----------
        ds: Dataset
          Original training dataset.

        Returns
        -------
        None
        """
        pass
