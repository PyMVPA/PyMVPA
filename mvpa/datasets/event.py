#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Event-based dataset type"""

__docformat__ = 'restructuredtext'


import numpy as N

from mvpa.mappers.array import DenseArrayMapper
from mvpa.mappers.boxcar import BoxcarMapper
from mvpa.mappers.mask import MaskMapper
from mvpa.datasets.base import Dataset
from mvpa.datasets.mapped import MappedDataset
from mvpa.mappers.base import ChainMapper, CombinedMapper
from mvpa.base import warning

class EventDataset(MappedDataset):
    """Event-based dataset

    This dataset type can be used to segment 'raw' data input into meaningful
    boxcar-shaped samples, by simply defining a list of events
    (see :class:`~mvpa.misc.support.Event`).

    Additionally, it can be used to add arbitrary information (as features)
    to each event-sample (extracted from the event list itself). An
    appropriate mapper is automatically constructed, that merges original
    samples and additional features into a common feature space and also
    separates them again during reverse-mapping. Otherwise, this dataset type
    is a regular dataset (in contrast to `MetaDataset`).

    The properties of an :class:`~mvpa.misc.support.Event` supported/required
    by this class are:

    `onset`
      An integer indicating the startpoint of an event as the sample
      index in the input data.

    `duration`
      How many input data samples following the onset sample should be
      considered for an event. The embedded
      :class:`~mvpa.mappers.boxcar.BoxcarMapper` will use the maximum boxlength
      (i.e., `duration`) of all defined events to create a regular-shaped data
      array.

    `label`
      The corresponding label of that event (numeric or literal).

    `chunk`
      An optional chunk id.

    `features`
      A list with an arbitrary number of features values (floats), that will
      be added to the feature vector of the corresponding sample.
    """
    def __init__(self, samples=None, events=None, mask=None, bcshape=None,
                 **kwargs):
        """
        :Parameters:
          events: sequence of `Event` instances
            Both an events `onset` and `duration` are assumed to be provided
            as #samples. The boxlength will be determined by the maximum
            duration of all events.
          mask: boolean array
            Only features corresponding to non-zero mask elements will be
            considered for the final dataset. The mask shape has to match the
            shape of the generated boxcar-samples. If no mask is provided, a
            full mask will be constructed automatically.
          bcshape: tuple
            Shape of the boxcar samples generated by the embedded boxcar mapper.
            If not provided this is determined automatically. However, this
            required an extra mapping step.
          **kwargs
            All additional arguments are passed to the base class.
        """
        # check if we are in copy constructor mode
        if events is None:
            MappedDataset.__init__(self, samples=samples, **kwargs)
            return

        #
        # otherwise we really want to freshly prepare a dataset
        #

        # loop over events and extract all meaningful information to charge
        # a boxcar mapper
        startpoints = [e['onset'] for e in events]
        durations = [e['duration'] for e in events]

        # we need a regular array, so all events must have a common
        # boxlength
        boxlength = max(durations)
        if __debug__:
            if not max(durations) == min(durations):
                warning('Boxcar mapper will use maximum boxlength (%i) of all '
                        'provided Events.'% boxlength)

        # now look for stuff we need for the dataset itself
        labels = [e['label'] for e in events]
        # chunks are optional
        chunks = [e['chunk'] for e in events if e.has_key('chunk')]
        if not len(chunks):
            chunks = None

        # optional stuff
        # extract additional features for each event
        extrafeatures = [e['features'] 
                            for e in events if e.has_key('features')]

        # sanity check for extra features
        if len(extrafeatures):
            if len(extrafeatures) == len(startpoints):
                try:
                    # will fail if varying number of features per event
                    extrafeatures = N.asanyarray(extrafeatures)
                except ValueError:
                    raise ValueError, \
                          'Unequal number of extra features per event'
            else:
                raise ValueError, \
                      'Each event has to provide to same number of extra ' \
                      'features.'
        else:
            extrafeatures = None

        # now build the mapper
        # we know the properties of the boxcar mapper, so now use it
        # to determine its output size unless it is already provided
        bcmapper = BoxcarMapper(startpoints, boxlength)

        # determine array mapper input shape, as a fail-safe procedure
        # in case no mask provided, and to check the mask sanity if we have one
        if bcshape is None:
            # map the data and look at the shape of the first sample
            # to determine the properties of the array mapper
            bcshape = bcmapper(samples)[0].shape

        # now we can build the array mapper
        amapper = DenseArrayMapper(mask=mask, shape=bcshape)

        # now compose the full mapper for the main samples
        mapper = ChainMapper([bcmapper, amapper])

        # if we have extra features, we need to combine them with the rest
        if not extrafeatures is None:
            # first half for main samples, second half simple mask mapper
            # for unstructured additional features
            mapper = CombinedMapper(
                        (mapper,
                         MaskMapper(mask=N.ones(extrafeatures.shape[1]))))

            # add extra features to the samples
            samples = (samples, extrafeatures)

        # finally init baseclass
        MappedDataset.__init__(self,
                               samples=samples,
                               labels=labels,
                               chunks=chunks,
                               mapper=mapper,
                               **kwargs)
