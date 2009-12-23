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
from mvpa.mappers.base import ChainMapper, FeatureSliceMapper
from mvpa.mappers.flatten import FlattenMapper

if __debug__:
    from mvpa.base import debug


class Dataset(BaseDataset):
    __doc__ = BaseDataset.__doc__

    def get_mapped(self, mapper):
        """Feed this dataset through a mapper (forward).

        Parameters
        ----------
        mapper : Mapper

        Returns
        -------
        Dataset
          The forward-mapped dataset.
        """
        mds = mapper.forward(self)
        mds._append_mapper(mapper)
        return mds


    def _append_mapper(self, mapper):
        if not 'mapper' in self.a:
            self.a['mapper'] = mapper
            return

        pmapper = self.a.mapper
        # otherwise we have a mapper already, but is it a chain?
        if not isinstance(pmapper, ChainMapper):
            self.a.mapper = ChainMapper([pmapper])

        # is a chain mapper
        # merge slicer?
        lastmapper = self.a.mapper[-1]
        if isinstance(lastmapper, FeatureSliceMapper) \
           and lastmapper.is_mergable(mapper):
            lastmapper += mapper
        else:
            self.a.mapper.append(mapper)


    def __getitem__(self, args):
        # uniformize for checks below; it is not a tuple if just single slicing
        # spec is passed
        if not isinstance(args, tuple):
            args = (args,)

        # if we get an slicing array for feature selection and it is *not* 1D
        # try feeding it through the mapper (if there is any)
        if len(args) > 1 and isinstance(args[1], N.ndarray) \
           and len(args[1].shape) > 1 \
           and self.a.has_key('mapper'):
            args = list(args)
            args[1] = self.a.mapper.forward1(args[1])
            args = tuple(args)

        # let the base do the work
        ds = super(Dataset, self).__getitem__(args)

        # and adjusting the mapper (if any)
        if len(args) > 1 and 'mapper' in ds.a:
            # create matching mapper
            # the mapper is just appended to the dataset. It could also be
            # actually used to perform the slicing and prevent duplication of
            # functionality between the Dataset.__getitem__ and the mapper.
            # However, __getitem__ is sometimes more efficient, since it can
            # slice samples and feature axis at the same time. Moreover, the
            # mvpa.base.dataset.Dataset has no clue about mappers and should
            # be fully functional without them.
            subsetmapper = FeatureSliceMapper(args[1],
                                              dshape=self.samples.shape[1:])
            ds._append_mapper(subsetmapper)

        return ds



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
        # for all non-ndarray samples you need to go with the constructor
        samples = N.asanyarray(samples)

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
        ds = cls(samples, sa=sa_items)
        if not mapper is None:
            ds = ds.get_mapped(mapper)
        return ds


    @classmethod
    def from_masked(cls, samples, labels=None, chunks=None, mask=None,
                    space=None):
        """
        """
        # need to have arrays
        samples = N.asanyarray(samples)

        # use full mask if none is provided
        if mask is None:
            mask = N.ones(samples.shape[1:], dtype='bool')

        fm = FlattenMapper(shape=mask.shape, inspace=space)
        flatmask = fm.forward1(mask)
        submapper = FeatureSliceMapper(flatmask, dshape=flatmask.shape)
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
