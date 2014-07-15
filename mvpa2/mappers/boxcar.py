# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Transform consecutive samples into individual multi-dimensional samples"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.mappers.base import Mapper
from mvpa2.clfs.base import accepts_dataset_as_samples
from mvpa2.base.dochelpers import _str

if __debug__:
    from mvpa2.base import debug

class BoxcarMapper(Mapper):
    """Mapper to combine multiple samples into a single sample.

    Notes
    -----

    This mapper is somewhat unconventional since it doesn't preserve number
    of samples (ie the size of 0-th dimension).
    """
    # TODO: extend with the possibility to provide real onset vectors and a
    #       samples attribute that is used to determine the actual sample that
    #       is matching a particular onset. The difference between target onset
    #       and sample could be stored as an additional sample attribute. Some
    #       utility functionality (outside BoxcarMapper) could be used to merge
    #       arbitrary sample attributes into the samples matrix (with
    #       appropriate mapper adjustment, e.g. CombinedMapper).
    def __init__(self, startpoints, boxlength, offset=0, **kwargs):
        """
        Parameters
        ----------
        startpoints : sequence
          Index values along the first axis of 'data'.
        boxlength : int
          The number of elements after 'startpoint' along the first axis of
          'data' to be considered for the boxcar.
        offset : int
          The offset between the provided starting point and the actual start
          of the boxcar.
        """
        Mapper.__init__(self, **kwargs)
        self._outshape = None

        startpoints = np.asanyarray(startpoints)
        if np.issubdtype(startpoints.dtype, 'i'):
            self.startpoints = startpoints
        else:
            if __debug__:
                debug('MAP', "Boxcar: obtained startpoints are not of int type."
                      " Rounding and changing dtype")
            self.startpoints = np.asanyarray(np.round(startpoints), dtype='i')

        # Sanity checks
        if boxlength < 1:
            raise ValueError, "Boxlength lower than 1 makes no sense."
        if boxlength - int(boxlength) != 0:
            raise ValueError, "boxlength must be an integer value."

        self.boxlength = int(boxlength)
        self.offset = offset
        self.__selectors = None

        # build a list of list where each sublist contains the indexes of to be
        # averaged data elements
        self.__selectors = [ slice(i + offset, i + offset + boxlength) \
                             for i in startpoints ]


    def __reduce__(self):
        # python < 2.6 cannot copy slices, we will use the constructor the get
        # them back and additionally reapply the stae of the object (except for
        # the bad bad slices)
        state = self.__dict__.copy()
        badguy = '_%s__selectors' % self.__class__.__name__
        if badguy in state:
            del state[badguy]
        return (self.__class__,
                    (self.startpoints, self.boxlength, self.offset),
                    state)


    @accepts_dataset_as_samples
    def _train(self, data):
        startpoints = self.startpoints
        boxlength = self.boxlength
        if __debug__:
            offset = self.offset
            for sp in startpoints:
                if ( sp + offset + boxlength - 1 > len(data)-1 ) \
                   or ( sp + offset < 0 ):
                    raise ValueError('Illegal box (start: %i, offset: %i, '
                          'length: %i) with total input sample being %i.' \
                          % (sp, offset, boxlength, len(data)))
        self._outshape = (len(startpoints), boxlength) + data.shape[1:]


    def __repr__(self):
        s = super(BoxcarMapper, self).__repr__()
        return s.replace("(", "(boxlength=%d, offset=%d, startpoints=%s, " %
                         (self.boxlength, self.offset, str(self.startpoints)),
                         1)


    def __str__(self):
        return _str(self, bl=self.boxlength)


    def forward1(self, data):
        # if we have a single 'raw' sample (not a boxcar)
        # extend it to cover the full box -- useful if one
        # wants to forward map a mask in raw dataspace (e.g.
        # fMRI ROI or channel map) into an appropriate mask vector
        if not self._outshape:
            raise RuntimeError("BoxcarMapper needs to be trained before "
                               ".forward1() can be used.")
        # first axes need to match
        if not data.shape[0] == self._outshape[2]:
            raise ValueError("Data shape %s does not match sample shape %s."
                             % (data.shape[0], self._outshape[2]))

        return np.vstack([data[np.newaxis]] * self.boxlength)


    def _forward_data(self, data):
        """Project an ND matrix into N+1D matrix

        This method also handles the special of forward mapping a single 'raw'
        sample. Such a sample is extended (by concatenating clones of itself) to
        cover a full boxcar. This functionality is only availably after a full
        data array has been forward mapped once.

        Returns
        -------
        array: (#startpoint, ...)
        """
        # NOTE: _forward_dataset() relies on the assumption that the following
        # also works with 1D arrays and still yields sane results
        return np.vstack([data[box][np.newaxis] for box in self.__selectors])


    def _forward_dataset(self, dataset):
        msamp = self._forward_data(dataset.samples)
        # make a shallow copy of the dataset, but excluding all sample
        # and feature attributes, since they need to be transformed anyway
        mds = dataset.copy(deep=False, sa=[], fa=[])
        # assign the new samples and adjust the length check of the collections
        mds.samples = msamp
        mds.sa.set_length_check(len(mds))
        mds.fa.set_length_check(mds.nfeatures)
        # map old feature attributes -- which simply get broadcasted along the
        # boxcar
        for k in dataset.fa:
            mds.fa[k] = self.forward1(dataset.fa[k].value)
        # map old sample attributes -- which simply get stacked into one for all
        # boxcar elements/samples
        for k in dataset.sa:
            # using _forward_data() instead of forward(), since we know that
            # this implementation can actually deal with 1D-arrays
            mds.sa[k] = self._forward_data(dataset.sa[k].value)
        # create the box offset attribute if space name is given
        if self.get_space():
            mds.fa[self.get_space() + '_offsetidx'] = np.arange(self.boxlength,
                                                                dtype='int')
            mds.sa[self.get_space() + '_onsetidx'] = self.startpoints.copy()
        return mds


    def reverse1(self, data):
        if __debug__:
            if not data.shape == self._outshape[1:]:
                raise ValueError("BoxcarMapper has not been train to "
                                 "reverse-map %s-shaped data, but %s."
                                 % (data.shape, self._outshape[1:]))

        # reimplemented since it is really only that
        return data


    def _reverse_data(self, data):
        if len(data.shape) < 2:
            # this is not something that this mapper created -- let's broadcast
            # its elements and hope that it would work
            return np.repeat(data, self.boxlength)

        # stack them all together -- this will cause overlapping boxcars to
        # result in multiple identical samples
        if not data.shape[1] == self.boxlength:
            # stacking doesn't make sense, since we got something strange
            raise ValueError("%s cannot reverse-map, since the number of "
                             "elements along the second axis (%i) does not "
                             "match the boxcar-length (%i)."
                             % (self.__class__.__name__,
                                data.shape[1],
                                self.boxlength))

        return np.concatenate(data)


    def _reverse_dataset(self, dataset):
        msamp = self._reverse_data(dataset.samples)
        # make a shallow copy of the dataset, but excluding all sample
        # and feature attributes, since they need to be transformed anyway
        mds = dataset.copy(deep=False, sa=[], fa=[])
        mds.samples = msamp
        mds.sa.set_length_check(len(mds))
        mds.fa.set_length_check(mds.nfeatures)
        # map old feature attributes -- which simply is taken the first one
        # and kill the inspace attribute, since it 
        inspace = self.get_space()
        for k in dataset.fa:
            if inspace is None or k != (inspace + '_offsetidx'):
                mds.fa[k] = dataset.fa[k].value[0]
        # reverse-map old sample attributes
        for k in dataset.sa:
            if inspace is None or k != (inspace + '_onsetidx'):
                mds.sa[k] = self._reverse_data(dataset.sa[k].value)
        return mds
