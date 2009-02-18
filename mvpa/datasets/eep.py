# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset that gets its samples from an EEP binary file"""

__docformat__ = 'restructuredtext'


import numpy as N

from mvpa.datasets.channel import ChannelDataset
from mvpa.misc.io.eepbin import EEPBin
from mvpa.base.dochelpers import enhancedDocString
from mvpa.base import warning

class EEPDataset(ChannelDataset):
    """Dataset using a EEP binary file as source.

    EEP files are used by *eeprobe* a software for analysing even-related
    potentials (ERP), which was developed at the Max-Planck Institute for
    Cognitive Neuroscience in Leipzig, Germany.

      http://www.ant-neuro.com/products/eeprobe
    """
    def __init__(self, samples=None, **kwargs):
        """Initialize EEPDataset.

        :Parameters:
          samples: Filename (string) of a EEP binary file or an `EEPBin`
                   object
        """
        # dataset props defaults
        dt = t0 = channelids = None

        # default way to use the constructor: with filename
        if not samples is None:
            if isinstance(samples, str):
                # open the eep file
                try:
                    eb = EEPBin(samples)
                except RuntimeError, e:
                    warning("ERROR: EEPDatasets: Cannot open samples file %s" \
                            % samples) # should we make also error?
                    raise e
            elif isinstance(samples, EEPBin):
                # nothing special
                eb = samples
            else:
                raise ValueError, \
                      "EEPDataset constructor takes the filename of an " \
                      "EEP file or a EEPBin object as 'samples' argument."
            samples = eb.data
            dt = eb.dt
            channelids = eb.channels
            t0 = eb.t0

        # init dataset
        ChannelDataset.__init__(self,
                                samples=samples,
                                dt=dt, t0=t0, channelids=channelids,
                                **(kwargs))


    __doc__ = enhancedDocString('EEPDataset', locals(), ChannelDataset)
