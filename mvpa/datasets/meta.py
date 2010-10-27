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

import random
from mvpa.datasets.mapped import MappedDataset



if __debug__:
    from mvpa.base import debug, warning


class MetaDataset(object):
    """Dataset container

    The class is useful to combine several Datasets with different origin and
    type and bind them together. Such a combined dataset can then by used to
    e.g. pass it to a classifier.

    MetaDataset does not permanently duplicate data stored in the dataset it
    contains. The combined samples matrix is build on demand and samples
    attribute access is redirected to the first dataset in the container.

    Currently operations other than samples or feature selection are not fully
    supported, e.g. passing a MetaDataset to detrend() will initially result in
    a detrended MetaDataset, but the combined and detrended samples matrix will
    be lost after the next call to selectSamples() or selectFeatures(), which
    freshly pulls samples from all datasets in the container.  """

    # This class is intentionally _not_ implemented as a subclass of Dataset.
    # IMHO Dataset contains to much logic unecessary logic.
    # XXX implement MappedMetaDataset along with a MetaMapper that simply calls
    # the mappers in the datasets in the container; or maybe just add flag to
    # MetaDataset to behave like a MappedDataset
    def __init__(self, datasets):
        """Initialize dataset instance

        :Parameters:
          datasets : list
        """
        # XXX Maybe add checks that all datasets have identical samples
        #     attributes
        self.__datasets = datasets

        # contains the combine samples matrix for caching
        self.__samples = None


    def rebuildSamples(self):
        """Update the combined samples matrix from all underlying datasets.
        """
        # note, that hstack will do a copy of _all_ data
        self.__samples = N.hstack([ds.samples for ds in self.__datasets])


    def __getattr__(self, name):
        """Implemented to redirect access to underlying datasets.
        """
        if name == 'samples':
            # do something to combine (and cache) samples arrays
            if self.__samples is None:
                self.rebuildSamples()
            return self.__samples

        else:
            # redirect all other to first dataset
            # ??? maybe limit to some specific supported ones
            return self.__datasets[0].__getattribute__(name)


    def selectFeatures(self, ids, sort=True):
        """Do feature selection on all underlying datasets at once.
        """
        # determine which features belong to what dataset
        # and call its selectFeatures() accordingly
        ids = N.asanyarray(ids)
        result = []
        fsum = 0
        for ds in self.__datasets:
            # bool which meta feature ids belongs to this dataset
            selector = N.logical_and(ids < fsum + ds.nfeatures, ids >= fsum)
            # make feature ids relative to this dataset
            selected = ids[selector] - fsum
            # do feature selection on underlying dataset
            # XXX not sure if we should keep empty datasets? (probably)
            result.append(ds.selectFeatures(selected))
            fsum += ds.nfeatures

        return MetaDataset(result)


    def applyMapper(self, *args, **kwargs):
        """Apply a mapper on all underlying datasets.
        """
        return MetaDataset([ds.applyMapper(*args, **kwargs) \
                    for ds in self.__datasets])


    def selectSamples(self, *args, **kwargs):
        """Select samples from all underlying datasets at once.
        """
        return MetaDataset([ds.selectSamples(*args, **kwargs) \
                    for ds in self.__datasets])


    def permuteLabels(self, *args, **kwargs):
        """Toggle label permutation.
        """
        # permute on first
        self.__datasets[0].permuteLabels(*args, **kwargs)

        # and apply to all others
        for ds in self.__datasets[1:]:
            ds.samples[:] = self.__datasets[0].samples


    def getRandomSamples(self, nperlabel):
        """Return a MetaDataset with a random subset of samples.
        """
        # if interger is given take this value for all classes
        if isinstance(nperlabel, int):
            nperlabel = [ nperlabel for i in self.__datasets[0].uniquelabels ]

        sample = []
        # for each available class
        for i, r in enumerate(self.__datasets[0].uniquelabels):
            # get the list of pattern ids for this class
            sample += \
                random.sample((self.__datasets[0].labels == r).nonzero()[0],
                              nperlabel[i] )

        return MetaDataset([ds.selectSamples(sample) \
                    for ds in self.__datasets])


    def getNSamples( self ):
        """Currently available number of samples.
        """
        return self.__datasets[0].nsamples


    def getNFeatures( self ):
        """Number of features per sample.
        """
        return N.sum([ds.nfeatures for ds in self.__datasets])


    def setSamplesDType(self, dtype):
        """Set the data type of the samples array.
        """
        # reset samples
        self.__samples = None

        for ds in self.__datasets:
            if ds.samples.dtype != dtype:
                ds.samples = ds.samples.astype(dtype)


    def mapReverse(self, val):
        """Perform reverse mapping

        :Return:
          List of results per each used mapper and the corresponding part of
          the provided `val`.
        """
        # assure array and transpose for easy slicing
        # i.e. transpose of 1D does nothing, but of 2D puts features
        # along first dimension
        val = N.asanyarray(val).T

        # do we have multiple or just one
        mflag = len(val.shape) > 1

        result = []
        fsum = 0
        for ds in self.__datasets:
            # calculate upper border
            fsum_new = fsum + ds.nfeatures

            # now map back if mapper is present, otherwise just store
            # need to pass transposed!!
            if isinstance(ds, MappedDataset):
                result.append(ds.mapReverse(val[fsum:fsum_new].T))
            else:
                result.append(val[fsum:fsum_new].T)

            fsum = fsum_new

        return result


    # read-only class properties
    nsamples        = property(fget=getNSamples)
    nfeatures       = property(fget=getNFeatures)
    datasets        = property(fget=lambda self: self.__datasets)
