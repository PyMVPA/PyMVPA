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

from mvpa.base.node import Node

class Learner(Node):
    """Common trainable processing object.

    A `Learner` is a `Node` that can (maybe has to) be trained on a dataset,
    before it can perform its function.
    """
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
        self._pretrain(ds)
        result = self._train(ds)
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
