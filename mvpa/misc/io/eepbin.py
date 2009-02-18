# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
#   Derived from the EEP binary reader of the pybsig toolbox
#   (C) by Ingo Fruend
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reader for binary EEP files."""

__docformat__ = 'restructuredtext'

import numpy as N
from mvpa.misc.io import DataReader


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

        :Parameter:
          source : str
            Filename.
        """
        # init base class
        DataReader.__init__(self)
        # temp storage of number of samples
        nsamples = None
        # non-critical header components stored in temp dict
        hdr = {}

        infile = open(source, "r")

        # read file the end of header of EOF
        while True:
            # one line at a time
            line = infile.readline()

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
        if hdr.has_key('channels'):
            self._props['channels'] = hdr['channels'].split()

        self._data = \
            N.reshape(N.fromfile(infile, dtype='f'), \
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
