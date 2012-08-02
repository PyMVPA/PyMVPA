# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""IO helper for MEG datasets."""

__docformat__ = 'restructuredtext'

import sys
import numpy as np

from mvpa2.base import externals

class TuebingenMEG(object):
    """Reader for MEG data from line-based textfile format.

    This class reads segmented MEG data from a textfile, which is created by
    converting the proprietary binary output files of a MEG device in
    Tuebingen (Germany) with an unkown tool.

    The file format is line-based, i.e. all timepoints for all samples/trials
    are written in a single line. Each line is prefixed with an identifier
    (using a colon as the delimiter between identifier and data). Two lines
    have a special purpose. The first 'Sample Number' is a list of timepoint
    ids, similar to `range(ntimepoints)` for each sample/trial (all
    concatenated into one line. The second 'Time' contains the timing
    information for each timepoint (relative to stimulus onset), again for all
    trials concatenated into a single line.

    All other lines contain various information (channels) recorded during
    the experiment. The meaning of some channels is unknown. Known ones are:

      M*: MEG channels
      EEG*: EEG channels
      ADC*: Analog to digital converter output

    Dataset properties are available from various class attributes. The `data`
    member provides all data from all channels (except for 'Sample Number' and
    'Time') in a NumPy array (nsamples x nchannels x ntimepoints).

    The reader supports uncompressed as well as gzipped input files (or other
    file-like objects).
    """

    def __init__(self, source):
        """Reader MEG data from texfiles or file-like objects.

        Parameters
        ----------
        source : str or file-like
          Strings are assumed to be filenames (with `.gz` suffix
          compressed), while all other object types are treated as file-like
          objects.
        """
        self.ntimepoints = None
        self.timepoints = None
        self.nsamples = None
        self.channelids = []
        self.data = []
        self.samplingrate = None

        # open textfiles
        if isinstance(source, str):
            if source.endswith('.gz'):
                externals.exists('gzip', raise_=True)
                import gzip
                source = gzip.open(source, 'r')
                if sys.version >= '3':
                    # module still can not open text files
                    # in py3: Issue #13989 and #10791 
                    source = source.read().decode('ascii').splitlines()
            else:
                source = open(source, 'r')

        # read file
        for line in source:
            # split ID
            colon = line.find(':')

            # ignore lines without id
            if colon == -1:
                continue

            id = line[:colon]
            data = line[colon+1:].strip()
            if id == 'Sample Number':
                timepoints = np.fromstring(data, dtype=int, sep='\t')
                # one more as it starts with zero
                self.ntimepoints = int(timepoints.max()) + 1
                self.nsamples = int(len(timepoints) / self.ntimepoints)
            elif id == 'Time':
                self.timepoints = np.fromstring(data,
                                               dtype=float,
                                               count=self.ntimepoints,
                                               sep='\t')
                self.samplingrate = self.ntimepoints \
                    / (self.timepoints[-1] - self.timepoints[0])
            else:
                # load data
                self.data.append(
                    np.fromstring(data, dtype=float, sep='\t').reshape(
                        self.nsamples, self.ntimepoints))
                # store id
                self.channelids.append(id)

        # reshape data from (channels x samples x timepoints) to
        # (samples x chanels x timepoints)
        self.data = np.swapaxes(np.array(self.data), 0, 1)


    def __str__(self):
        """Give a short summary.
        """
        return '<TuebingenMEG: %i samples, %i timepoints, %i channels>' \
                  % (self.nsamples, self.ntimepoints, len(self.channelids))

