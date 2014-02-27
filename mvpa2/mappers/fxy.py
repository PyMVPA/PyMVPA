# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Evaluate functions on pairs of datasets"""

__docformat__ = 'restructuredtext'

from mvpa2.base.dochelpers import _str, _repr_attrs
from mvpa2.datasets import Dataset
from mvpa2.mappers.base import Mapper
from mvpa2.base.dochelpers import borrowdoc

class FxyMapper(Mapper):
    """Mapper to execute a callable with two datasets as arguments.

    The first dataset is passed to the mapper during training, the second
    dataset is passed to forward/call(). This mapper is useful to, for example,
    compare two datasets regarding particular aspects, merge them, or perform
    other operations that require the presence of two datasets.
    """

    def __init__(self, fx, train_as_1st=True, **kwargs):
        """
        Parameters
        ----------
        fx : callable
          Functor that is called with the two datasets upon forward-mapping.
        train_as_1st : bool
          If True, the training dataset is passed to the target callable as
          the first argument and the other dataset as the second argument.
          If False, it is vice versa.

        Examples
        --------
        >>> from mvpa2.mappers.fxy import FxyMapper
        >>> from mvpa2.datasets import Dataset
        >>> callable = lambda x,y: len(x) > len(y)
        >>> ds1 = Dataset(range(5))
        >>> ds2 = Dataset(range(3))
        >>> fxy = FxyMapper(callable)
        >>> fxy.train(ds1)
        >>> fxy(ds2).item()
        True
        >>> fxy = FxyMapper(callable, train_as_1st=False)
        >>> fxy.train(ds1)
        >>> fxy(ds2).item()
        False
        """
        Mapper.__init__(self, **kwargs)
        self._fx = fx
        self._train_as_1st = train_as_1st
        self._ds_train = None

    @borrowdoc(Mapper)
    def __repr__(self, prefixes=[]):
        return super(FxyMapper, self).__repr__(
                prefixes=prefixes + _repr_attrs(self, ['fx']))

    def __str__(self):
        return _str(self, fx=self._fx.__name__)

    def _train(self, ds):
        self._ds_train = ds

    def _untrain(self):
        self._ds_train = None

    @borrowdoc(Mapper)
    def _forward_dataset(self, ds):
        # apply function
        if self._train_as_1st:
            out = self._fx(self._ds_train, ds)
        else:
            out = self._fx(ds, self._ds_train)
        # wrap output in a dataset if necessary
        if not isinstance(out, Dataset):
            try:
                out = Dataset(out)
            except ValueError:
                # not a sequence?
                out = Dataset([out])
        return out

    fx = property(fget=lambda self:self.__fx)

