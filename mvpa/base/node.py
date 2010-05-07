# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Implementation of a common processing object (node)."""

__docformat__ = 'restructuredtext'


from mvpa.base.dochelpers import _str, _repr
from mvpa.base.state import ClassWithCollections


class Node(ClassWithCollections):
    """Common processing object.

    A `Node` is an object the processes datasets. It can be called with a
    `Dataset` and returns another dataset with the results. In addition, a node
    can also be used as a generator. Upon calling ``generate()`` with a datasets
    it yields (potentially) multiple result datasets.

    Node have a notion of ``space``. The meaning of this space may vary heavily
    across sub-classes. In general, this is a trigger that tells the node to
    compute and store information about the input data that is "interesting" in
    the context of the corresponding processing in the output dataset.
    """
    def __init__(self, space=None, **kwargs):
        """
        Parameters
        ----------
        space: str
          Name of the 'processing space'. The actual meaning of this argument
          heavily depends on the sub-class implementation. In general, this is
          a trigger that tells the node to compute and store information about
          the input data that is "interesting" in the context of the
          corresponding processing in the output dataset.
        """
        ClassWithCollections.__init__(self, **kwargs)
        self.set_space(space)


    def __call__(self, ds):
        """
        The default implementation calls ``_precall()``, ``_call()``, and
        finally returns the output of ``_postcall()``.

        Parameters
        ----------
        ds: Dataset
          Input dataset.

        Returns
        -------
        Dataset
        """
        self._precall(ds)
        result = self._call(ds)
        result = self._postcall(ds, result)
        return result


    def _precall(self, ds):
        """Preprocessing of data

        By default, does nothing.

        Parameters
        ----------
        ds: Dataset
          Original input dataset.

        Returns
        -------
        Dataset
        """
        return ds


    def _call(self, ds):
        raise NotImplementedError


    def _postcall(self, ds, result):
        """Postprocessing of results.

        By default, does nothing.

        Parameters
        ----------
        ds: Dataset
          Original input dataset.
        result: Dataset
          Preliminary result dataset (as produced by ``_call()``).

        Returns
        -------
        Dataset
        """
        return result


    def generate(self, ds):
        """Yield processing results.

        This methods causes the node to behave like a generator. By default it
        simply yields a single result of its processing -- identical to the
        output of calling the node with a dataset. Subclasses might implement
        generators that yield multiple results.

        Parameters
        ----------
        ds: Dataset
          Input dataset

        Returns
        -------
        generator
          the generator yields the result of the processing.
        """
        yield self(ds)


    def get_space(self):
        """Query the processing space name of this node."""
        return self.__space


    def set_space(self, name):
        """Set the processing space name of this node."""
        self.__space = name


    def __str__(self):
        return _str(self)
