#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Simple preprocessing of event-related data using FSL EV3 design files.

Events are defined using FSL's EV3 format (onset, duration, intensity).
For each event a number of volumes is selected and the mean volume is
computed. The selection method is a boxcar with defined length and offset from
the event onset (--boxlength, --boxoffset). The computed event data samples
are written back to a NIfTI file using the header information of the source
timeseries.

Additonally a file listing the sample attributes (label, chunk) is created.
All samples are assumed to be from the same chunk, where the chunk id can be
set using the --chunk option. If the source NIfTI data file contains more than
one chunk, one can be selected using the --chunklimits option. This will have
the effect that linear detrending (--detrend) will perform a separate fit for
this chunk and will not simply remove a global trend.
"""

from mvpa.misc.fsl.base import FslEV3
from mvpa.suite import *
"""
# Command above substitutes the following list

import sys

import numpy as N

from scipy.signal import detrend

from nifti import NiftiImage

from mvpa.misc.support import transformWithBoxcar
from mvpa.misc.iohelpers import SampleAttributes, FslEV3
from mvpa.misc import verbose
from mvpa.misc.cmdline import parser, \
     optsCommon, optZScore, optTr, optsBox, optsChunk, optDetrend
"""
from nifti.utils import time2vol

def main():
    """ Wrapped into a function call for easy profiling later on
    """

    parser.usage = """\
    %s [options] <NIfTI data> <output prefix> <EV file 1> [ <EV file 2> ... ]
    """ \
    % sys.argv[0]

    parser.option_groups += [ optsCommon, optsBox, optsChunk]
    parser.option_list += [optTr, optDetrend]

    (options, args) = parser.parse_args()

    if not len(args) >= 3:
        parser.error("Insufficient arguments.")
        sys.exit(1)

    verbose(1, "Loading data")

    # data filename
    dfile = args[0]
    # output prefix
    oprefix = args[1]
    # list of EV files
    evfiles = args[2:]

    verbose(2, "Reading conditions from files")
    evs = [ FslEV3(evfile) for evfile in evfiles ]

    verbose(2, "Loading volume file %s" % dfile)
    nimg = NiftiImage(dfile)

    verbose(1, "Preprocess data")
    # force float32 to prevent unecessary upcasting of int to float64
    data = nimg.data.astype('float32')

    if options.detrend:
        if options.chunklimits == None:
            verbose(2, "Linear detrending (whole dataset)")
            data = detrend(data, axis=0)
        else:
            verbose(2, "Linear detrending (data chunk only)")
            limits = [int(i) for i in options.chunklimits.split(',')]

            if data[limits[0]:limits[1],:].shape[0] == 0:
                raise ValueError, 'Invalid chunklimit value [%s].' \
                                  % options.chunklimits
            # use limits to do a piecewise linear detrending (separate linear
            # fit of the interesting chunk
            data = detrend(data, axis=0, bp=limits)


    verbose(2, "Convert onsets into volume ids")
    # transform onset time into a volume id
    onset_vols = [ time2vol(ev.onsets, options.tr, 0.0,
                   decimals = 0 ).astype('int') for ev in evs ]

    verbose(2, "Compute EV samples")

    labels = []
    chunks = []
    samples = []

    print data.shape
    # for each condition -> label
    for label, onsets in enumerate(onset_vols):
        # mean of all volumes in a window after each onset vol
        samples.append(transformWithBoxcar(data,
                                           onsets,
                                           options.boxlength,
                                           offset=options.boxoffset,
                                           fx=N.mean))
        labels += [label] * len(onsets)
        chunks += [options.chunk] * len(onsets)

    # concatenate into a single array (assumes that each entry in samples
    # already is 4d which is true, because transformWithBoxcar does it like
    # that
    samples = N.concatenate(samples, axis=0)

    attrs = SampleAttributes({'labels': labels, 'chunks': chunks})

    verbose(1, "Store results")
    NiftiImage(samples, nimg.header).save(oprefix + '.nii.gz')
    attrs.tofile(oprefix + '.attrs.txt')


# if ran stand-alone
if __name__ == "__main__":
    main()
