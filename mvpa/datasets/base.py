# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset container"""

__docformat__ = 'restructuredtext'

import numpy as N
import copy

from mvpa.base.collections import SampleAttributesCollection, \
        FeatureAttributesCollection, DatasetAttributesCollection, \
        SampleAttribute, FeatureAttribute, DatasetAttribute
from mvpa.base.dataset import Dataset as BaseDataset
from mvpa.base.dataset import _expand_attribute
from mvpa.misc.support import idhash as idhash_
from mvpa.mappers.base import ChainMapper, FeatureSubsetMapper
from mvpa.mappers.flatten import FlattenMapper

if __debug__:
    from mvpa.base import debug


class Dataset(BaseDataset):
    @property
    def idhash(self):
        """To verify if dataset is in the same state as when smth else was done

        Like if classifier was trained on the same dataset as in question
        """

        res = 'self@%s samples@%s' % (idhash_(self), idhash_(self.samples))

        for col in (self.a, self.sa, self.fa):
            # We cannot count on the order the values in the dict will show up
            # with `self._data.value()` and since idhash will be order-dependent
            # we have to make it deterministic
            keys = col.keys()
            keys.sort()
            for k in keys:
                res += ' %s@%s' % (k, idhash_(col[k].value))
        return res


    @classmethod
    def from_basic(cls, samples, labels=None, chunks=None, mapper=None):
        """Create a Dataset from samples and elementary attributes.

        Parameters
        ----------
        samples : ndarray
          The two-dimensional samples matrix.
        labels : ndarray
        chunks : ndarray
        mapper : Mapper instance
          A (potentially trained) mapper instance that is used to forward-map
          the samples upon construction of the dataset. The mapper must
          have a simple feature space (samples x features) as output. Use
          chained mappers to achieve that, if necessary.

        Returns
        -------
        instance : Dataset

        Notes
        -----
        blah blah

        it needs to be a little longer to be able to pick it up

        See Also
        --------
        blah blah

        Examples
        --------
        blah blah
        """
        # we depend on arrays
        samples = N.asanyarray(samples)

        # put mapper as a dataset attribute in the general attributes collection
        # need to do that first, since we want to know the final
        # #samples/#features
        a_items = {}
        if not mapper is None:
            # forward-map the samples
            samples = mapper.forward(samples)
            # and store the mapper
            a_items['mapper'] = mapper

       # compile the necessary samples attributes collection
        sa_items = {}

        if not labels is None:
            sa_items['labels'] = _expand_attribute(labels,
                                                   samples.shape[0],
                                                  'labels')

        if not chunks is None:
            # unlike previous implementation, we do not do magic to do chunks
            # if there are none, there are none
            sa_items['chunks'] = _expand_attribute(chunks,
                                                   samples.shape[0],
                                                   'chunks')

        # common checks should go into __init__
        ds = cls(samples, sa=sa_items, a=a_items)

        return ds



    @classmethod
    def from_masked(cls, samples, labels=None, chunks=None, mask=None):
        """
        """
        # need to have arrays
        samples = N.asanyarray(samples)

        # use full mask if none is provided
        if mask is None:
            mask = N.ones(samples.shape[1:], dtype='bool')

        fm = FlattenMapper(shape=mask.shape)
        submapper = FeatureSubsetMapper(mask=fm.forward(mask))
        mapper = ChainMapper([fm, submapper])
        return cls.from_basic(samples, labels=labels, chunks=chunks,
                              mapper=mapper)


    # shortcut properties
    S = property(fget=lambda self:self.samples)
    labels = property(fget=lambda self:self.sa.labels,
                      fset=lambda self, v:self.sa.__setattr__('labels', v))
    uniquelabels = property(fget=lambda self:self.sa['labels'].unique)

    L = labels
    UL = property(fget=lambda self:self.sa['labels'].unique)
    chunks = property(fget=lambda self:self.sa.chunks,
                      fset=lambda self, v:self.sa.__setattr__('chunks', v))
    uniquechunks = property(fget=lambda self:self.sa['chunks'].unique)
    C = chunks
    UC = property(fget=lambda self:self.sa['chunks'].unique)
    mapper = property(fget=lambda self:self.a.mapper)
    O = property(fget=lambda self:self.a.mapper.reverse(self.samples))


# convenience alias
dataset = Dataset.from_basic
