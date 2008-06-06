#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
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


class EEPBin(object):
    """Read-access to binary EEP files.

    EEP files are used by eeprobe_ a software for analysing even-related
    potentials (ERP), which was developed at the Max-Planck Institute for
    Cognitive Neuroscience in Leipzig, Germany.

    .. _eegprobe: http://http://www.ant-neuro.com/products/eeprobe

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

        Parameter
        ---------

          source : str
            Filename.
        """
        infile = open(source, "r")

        # non-critical header components stored in dict
        self.__hdr = {}

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
                self.__nchannels = int(l[0][1:])
                self.__nsamples = int(l[1])
                self.__ntrials = int(l[2])
                self.__t0 = float(l[3])
                self.__dt = float(l[4])
            else:
                # simply store non-critical extras
                l = line.split(':')
                key = l[0].lstrip(';')
                value = ':'.join(l[1:])
                self.__hdr[key] = value

        # post process channel name info -> list
        if self.__hdr.has_key('channels'):
            self.__hdr['channels'] = self.__hdr['channels'].split()

        self.__data = \
            N.reshape(N.fromfile(infile, dtype='f'), \
                (self.__ntrials,
                 self.__nchannels,
                 self.__nsamples))

        # cleanup
        infile.close()


    nchannels = property(fget=lambda self: self.__nchannels,
                         doc="Number of channels")
    nsamples  = property(fget=lambda self: self.__nsamples,
                         doc="Number of data samples")
    ntrials   = property(fget=lambda self: self.__ntrials,
                         doc="Number of trials")
    t0        = property(fget=lambda self: self.__t0,
                         doc="Relative start time of sampling interval")
    dt        = property(fget=lambda self: self.__dt,
                         doc="Time difference between two adjacent samples")
    channels  = property(fget=lambda self: self.__hdr['channels'],
                         doc="List of channel names")
    data      = property(fget=lambda self: self.__data,
                         doc="Data array (trials x channels x samples)")
