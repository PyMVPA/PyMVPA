# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Support for the binary EEP file format for EEG data"""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.datasets import Dataset
from mvpa2.misc.io import DataReader

def eep_dataset(samples, targets=None, chunks=None):
    """Create a dataset using an EEP binary file as source.

    EEP files are used by *eeprobe* a software for analysing even-related
    potentials (ERP), which was developed at the Max-Planck Institute for
    Cognitive Neuroscience in Leipzig, Germany.

      http://www.ant-neuro.com/products/eeprobe

    Parameters
    ----------
    samples : str or EEPBin instance
      This is either a filename of an EEP file, or an EEPBin instance, providing
      the samples data in EEP format.
    targets, chunks : sequence or scalar or None
      Values are pass through to `Dataset.from_wizard()`. See its documentation
      for more information.

    Returns
    -------
    Dataset
      Besides is usual attributes (e.g. targets, chunks, and a mapper). The
      returned dataset also includes feature attributes associating each same
      with a channel (by id), and a specific timepoint -- based on information
      read from the EEP data.
    """
    if isinstance(samples, str):
        # open the eep file
        eb = EEPBin(samples)
    elif isinstance(samples, EEPBin):
        # nothing special
        eb = samples
    else:
        raise ValueError("eep_dataset takes the filename of an "
              "EEP file or a EEPBin object as 'samples' argument.")

    # init dataset
    ds = Dataset.from_channeltimeseries(
            eb.data, targets=targets, chunks=chunks, t0=eb.t0, dt=eb.dt,
            channelids=eb.channels)
    return ds



class EEPBin(DataReader):
    """Read-access to binary EEP files.

    EEP files are used by *eeprobe* a software for analysing even-related
    potentials (ERP), which was developed at the Max-Planck Institute for
    Cognitive Neuroscience in Leipzig, Germany.

      http://www.ant-neuro.com/products/eeprobe

    EEP files consist of a plain text header and a binary data block in a
    single file. The header starts with a line of the form

    ';%d %d %d %g %g' % (Nchannels, Nsamples, Ntrials, t0, dt)

    where Nchannels, Nsamples, Ntrials are the numbers of channels, samples
    per trial and trials respectively. t0 is the time of the first sample
    of a trial relative to the stimulus onset and dt is the sampling interval.

    The binary data block consists of single precision floats arranged in the
    following way::

        <trial1,channel1,sample1>,<trial1,channel1,sample2>,...
        <trial1,channel2,sample1>,<trial1,channel2,sample2>,...
        .
        <trial2,channel1,sample1>,<trial2,channel1,sample2>,...
        <trial2,channel2,sample1>,<trial2,channel2,sample2>,...
    """
    def __init__(self, source):
        """Read EEP file and store header and data.

        Parameters
        ----------
        source : str
          Filename.
        """
        # init base class
        DataReader.__init__(self)
        # temp storage of number of samples
        nsamples = None
        # non-critical header components stored in temp dict
        hdr = {}

        infile = open(source, "rb")

        # read file the end of header of EOF
        while True:
            # one line at a time
            try:
                line = infile.readline().decode('ascii')
            except UnicodeDecodeError:
                break

            # stop if EOH or EOF
            if not line or line.startswith(';EOH;'):
                break

            # no crap!
            line = line.strip()

            # all but first line as colon
            if not line.count(':'):
                # top header
                l = line.split()
                # extract critical information
                self._props['nchannels'] = int(l[0][1:])
                self._props['ntimepoints'] = int(l[1])
                self._props['t0'] = float(l[3])
                self._props['dt'] = float(l[4])
                nsamples = int(l[2])
            else:
                # simply store non-critical extras
                l = line.split(':')
                key = l[0].lstrip(';')
                value = ':'.join(l[1:])
                hdr[key] = value

        # post process channel name info -> list
        if 'channels' in hdr:
            self._props['channels'] = hdr['channels'].split()

        self._data = \
            np.reshape(np.fromfile(infile, dtype='f'), \
                (nsamples,
                 self._props['nchannels'],
                 self._props['ntimepoints']))

        # cleanup
        infile.close()


    nchannels = property(fget=lambda self: self._props['nchannels'],
                         doc="Number of channels")
    ntimepoints  = property(fget=lambda self: self._props['ntimepoints'],
                         doc="Number of data timepoints")
    nsamples   = property(fget=lambda self: self._data.shape[0],
                         doc="Number of trials/samples")
    t0        = property(fget=lambda self: self._props['t0'],
                         doc="Relative start time of sampling interval")
    dt        = property(fget=lambda self: self._props['dt'],
                         doc="Time difference between two adjacent samples")
    channels  = property(fget=lambda self: self._props['channels'],
                         doc="List of channel names")
