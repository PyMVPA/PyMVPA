# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA's common Dataset container."""

__docformat__ = 'restructuredtext'

import numpy as N
import copy

from mvpa.base.collections import SampleAttributesCollection, \
        FeatureAttributesCollection, DatasetAttributesCollection, \
        SampleAttribute, FeatureAttribute, DatasetAttribute
from mvpa.base.dataset import AttrDataset
from mvpa.base.dataset import _expand_attribute
from mvpa.misc.support import idhash as idhash_
from mvpa.mappers.base import ChainMapper, FeatureSliceMapper
from mvpa.mappers.flatten import mask_mapper, FlattenMapper

if __debug__:
    from mvpa.base import debug


class Dataset(AttrDataset):
    __doc__ = AttrDataset.__doc__

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


    def item(self):
        """Provide the first element of samples array.

        Notes
        -----
        Introduced to provide compatibility with `numpy.asscalar`.
        See `numpy.ndarray.item` for more information.
        """
        return self.samples.item()


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
    def from_wizard(cls, samples, labels=None, chunks=None, mask=None,
                    mapper=None, space=None):
        """Convenience method to create dataset.

        Datasets can be created from N-dimensional samples. Data arrays with
        more than two dimensions are going to be flattened, while preserving
        the first axis (separating the samples) and concatenating all other as
        the second axis. Optionally, it is possible to specific labels and
        chunk attributes for all samples, and masking of the input data (only
        selecting elements corresponding to non-zero mask elements

        Parameters
        ----------
        samples : ndarray
          N-dimensional samples array. The first axis separates individual
          samples.
        labels : scalar or ndarray, optional
          Labels for all samples. If a scalar is provided its values is assigned
          as label to all samples.
        chunks : scalar or ndarray, optional
          Chunks definition for all samples. If a scalar is provided its values
          is assigned as chunk of all samples.
        mask : ndarray, optional
          The shape of the array has to correspond to the shape of a single
          sample (shape(samples)[1:] == shape(mask)). Its non-zero elements
          are used to mask the input data.
        space : str, optional
          If provided it is assigned to the mapper instance that performs the
          initial flattening of the data.
        mapper : Mapper instance, optional
          A (potentially trained) mapper instance that is used to forward-map
          the already flattened and masked samples upon construction of the
          dataset. The mapper must have a simple feature space (samples x
          features) as output. Use a `ChainMapper` to achieve that, if
          necessary.

        Returns
        -------
        instance : Dataset
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
        # apply mask through mapper
        if mask is None:
            if len(samples.shape) > 2:
                # if we have multi-dim data
                fm = FlattenMapper(shape=samples.shape[1:], inspace=space)
                ds = ds.get_mapped(fm)
        else:
            mm = mask_mapper(mask, inspace=space)
            ds = ds.get_mapped(mm)

        # apply generic mapper
        if not mapper is None:
            ds = ds.get_mapped(mapper)
        return ds


    @classmethod
    def from_channeltimeseries(cls, samples, labels=None, chunks=None,
                               t0=None, dt=None, channelids=None):
        """Create a dataset from segmented, per-channel timeseries.

        Channels are assumes to contain multiple, equally spaced acquisition
        timepoints. The dataset will contain additional feature attributes
        associating each feature with a specific `channel` and `timepoint`.

        Parameters
        ----------
        samples : ndarray
          Three-dimensional array: (samples x channels x timepoints).
        t0 : float
          Reference time of the first timepoint. Can be used to preserve
          information about the onset of some stimulation. Preferably in
          seconds.
        dt : float
          Temporal distance between two timepoints. Preferably in seconds.
        channelids : list
          List of channel names.
        labels, chunks
          See `Dataset.from_wizard` for documentation about these arguments.
        """
        # check samples
        if len(samples.shape) != 3:
            raise ValueError(
                "Input data should be (samples x channels x timepoints. Got: %s"
                % samples.shape)

        if not t0 is None and not dt is None:
            timepoints = N.arange(t0, t0 + samples.shape[2] * dt, dt)
            # broadcast over all channels
            timepoints = N.vstack([timepoints] * samples.shape[1])
        else:
            timepoints = None

        if not channelids is None:
            if len(channelids) != samples.shape[1]:
                raise ValueError(
                    "Number of channel ids does not match channels in the "
                    "sample data. Expected %i, but got %i"
                    % (samples.shape[1], len(channelids)))
            # broadcast over all timepoints
            channelids = N.dstack([channelids] * samples.shape[2])[0]

        ds = cls.from_wizard(samples, labels=labels, chunks=chunks)

        # add additional attributes
        if not timepoints is None:
            ds.fa['timepoints'] = ds.a.mapper.forward1(timepoints)
        if not channelids is None:
            ds.fa['channels'] = ds.a.mapper.forward1(channelids)

        return ds


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
dataset_wizard = Dataset.from_wizard

