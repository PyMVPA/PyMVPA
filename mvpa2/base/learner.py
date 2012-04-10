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
from mvpa2.base.node import Node
from mvpa2.base.state import ConditionalAttribute
from mvpa2.base.types import is_datasetlike
from mvpa2.base.dochelpers import _repr_attrs

if __debug__:
    from mvpa2.base import debug


class LearnerError(Exception):
    """Base class for exceptions thrown by the Learners
    """
    pass


class DegenerateInputError(LearnerError):
    """Learner exception thrown if input data is not bogus

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


    def __init__(self, auto_train=False, force_train=False, **kwargs):
        """
        Parameters
        ----------
        auto_train : bool
          Flag whether the learner will automatically train itself on the input
          dataset when called untrained.
        force_train : bool
          Flag whether the learner will enforce training on the input dataset
          upon every call.
        **kwargs
          All arguments are passed to the baseclass.
        """
        Node.__init__(self, **kwargs)
        self.__is_trained = False
        self.__auto_train = auto_train
        self.__force_train = force_train


    def __repr__(self, prefixes=[]):
        return super(Learner, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['auto_train', 'force_train'], default=False))

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
                    "Cannot train learner on degenerate data %s" % ds)
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

        # finally flag as trained
        self._set_trained()


    def untrain(self):
        """Reverts changes in the state of this node caused by previous training
        """
        # flag the learner as untrained
        # important to do that before calling the implementation in the derived
        # class, as it might decide that an object remains trained
        self._set_trained(False)
        # call subclass untrain first to allow it to access current attributes
        self._untrain()
        # TODO evaluate whether this should also reset the nodes collections, or
        # whether that should be done by a more general reset() method
        self.reset()


    def _untrain(self):
        # nothing by default
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
        # nothing by default
        pass


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


    def _set_trained(self, status=True):
        """Set the Learner's training status

        Derived use this to set the Learner's status to trained (True) or
        untrained (False).
        """
        self.__is_trained = status


    def __call__(self, ds):
        # overwrite __call__ to perform a rigorous check whether the learner was
        # trained before use and auto-train
        if self.is_trained:
            # already trained
            if self.force_train:
                if __debug__:
                    debug('LRN', "Forcing training of %s on %s",
                          (self, ds))
                # but retraining is enforced
                self.train(ds)
            elif __debug__:
                debug('LRN', "Skipping training of already trained %s on %s",
                      (self, ds))
        else:
            # not trained
            if self.auto_train:
                # auto training requested
                if __debug__:
                    debug('LRN', "Auto-training %s on %s",
                          (self, ds))
                self.train(ds)
            else:
                # we always have to have trained before using a learner
                raise RuntimeError("%s needs to be trained before it can be "
                                   "used and auto training is disabled."
                                   % str(self))
        return super(Learner, self).__call__(ds)


    is_trained = property(fget=lambda x:x.__is_trained,
                          doc="Whether the Learner is currently trained.")
    auto_train = property(fget=lambda x:x.__auto_train,
                          doc="Whether the Learner performs automatic training"
                              "when called untrained.")
    force_train = property(fget=lambda x:x.__force_train,
                          doc="Whether the Learner enforces training upon every"
                              "called.")
