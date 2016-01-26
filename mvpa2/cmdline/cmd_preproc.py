# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Preprocess a PyMVPA dataset.

This command can apply a number of preprocessing steps to a dataset. Currently
supported are

1. Polynomial de-trending

2. Spectral filtering

3. Feature-wise Z-scoring

All preprocessing steps are applied in the above order. If a different order is
required, preprocessing has to be split into two separate command calls.


POLYNOMIAL DE-TRENDING

This type of de-trending can be used to regress out arbitrary signals. In
addition to polynomials of any degree arbitrary timecourses stored as sample
attributes in a dataset can be used as confound regressors. This detrending
functionality is, in contrast to the implementation of spectral filtering,
also applicable to sparse-sampled data with potentially irregular inter-sample
intervals.


SPECTRAL FILTERING

Several option are provided that are used to construct a Butterworth low-,
high-, or band-pass filter. It is advised to inspect the filtered data
carefully as inappropriate filter settings can lead to unintented side-effect.
Only dataset with a fixed sampling rate are supported. The sampling rate
must be provided.


Examples:

Normalize all features in a dataset by Z-scoring

  $ pymvpa2 preproc --zscore -o ds_preprocessed -i dataset.hdf5

Perform Z-scoring and quadratic detrending of all features, but process all
samples sharing a unique value of the "chunks" sample attribute individually

  $ pymvpa2 preproc --chunks "chunks" --poly-detrend 2 --zscore -o ds_pp2 -i ds.hdf5

"""

# magic line for manpage summary
# man: -*- % apply preprocessing steps to a PyMVPA dataset

__docformat__ = 'restructuredtext'

import numpy as np
import argparse

from mvpa2.base import verbose, warning, error
from mvpa2.datasets import Dataset
from mvpa2.mappers.detrend import PolyDetrendMapper
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
        import parser_add_common_opt, ds2hdf5, \
               arg2ds, parser_add_optgroup_from_def, \
               single_required_hdf5output

parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

detrend_args = ('options for data detrending', [
    (('--poly-detrend',), (PolyDetrendMapper, 'polyord'), dict(metavar='DEG')),
    (('--detrend-chunks',), (PolyDetrendMapper, 'chunks_attr'),
     dict(metavar='CHUNKS_ATTR')),
    (('--detrend-coords',),
     dict(type=str, metavar='COORDS_ATTR',
         help="""name of a samples attribute that is added to the
              preprocessed dataset storing the coordinates of each sample in the
              space spanned by the polynomials. If an attribute of such name
              is already present in the dataset its values are interpreted
              as sample coordinates in the space spanned by the polynomials.
              This can be used to detrend datasets with irregular sample
              spacing.""")),
    (('--detrend-regrs',), (PolyDetrendMapper, 'opt_regs'),
     dict(nargs='+', metavar='ATTR', type=str))
])

normalize_args = ('options for data normalization', [
    (('--zscore',),
     dict(action='store_true',
          help="""perform feature normalization by Z-scoring.""")),
    (('--zscore-chunks',),
     dict(metavar='CHUNKS_ATTR',
          help="""name of a dataset sample attribute defining chunks of
               samples that shall be Z-scored independently. By default
               no chunk-wise normalization is done.""")),
    (('--zscore-params',),
     dict(metavar='PARAM', nargs=2, type=float,
          help="""define a fixed parameter set (mean, std) for Z-scoring,
               instead of computing from actual data.""")),
])

bandpassfilter_args = ('options for spectral filtering', [
    (('--filter-passband',),
     dict(metavar='FREQ', nargs='+', type=float,
          help="""critical frequencies of a Butterworth filter's pass band.
          Critical frequencies need to match the unit of the specified sampling
          rate (see: --sampling-rate). In case of a band pass filter low and
          high frequency cutoffs need to be specified (in this order). For
          low and high-pass filters is single cutoff frequency must be
          provided. The type of filter (low/high-pass) is determined from the
          relation to the stop band frequency (--filter-stopband).""")),
    (('--filter-stopband',),
     dict(metavar='FREQ', nargs='+', type=float,
          help="""Analog setting to --filter-passband for specifying the
          filter's stop band.""")),
    (('--sampling-rate',),
     dict(metavar='FREQ', type=float,
          help="""sampling rate of the dataset. All frequency specifications
          need to match the unit of the sampling rate.""")),
    (('--filter-passloss',),
     dict(metavar='dB', type=float, default=1.0,
          help="""maximum loss in the passband (dB). Default: 1 dB""")),
    (('--filter-stopattenuation',),
     dict(metavar='dB', type=float, default=30.0,
         help="""minimum attenuation in the stopband (dB). Default: 30 dB""")),
])

common_args = ('common options for all preprocessing', [
    (('--chunks',),
     dict(metavar='CHUNKS_ATTR',
          help="""shortcut option to enabled uniform chunkwise processing for
               all relevant preprocessing steps (see --zscore-chunks,
               --detrend-chunks). This global setting can be overwritten by
               additionally specifying the corresponding individual "chunk"
               options.""")),
    (('--strip-invariant-features',),
     dict(action='store_true',
          help="""After all pre-processing steps are done, strip all invariant
          features from the dataset.""")),

])

def setup_parser(parser):
    parser_add_common_opt(parser, 'multidata', required=True)
    # order of calls is relevant!
    for src in (common_args, detrend_args, bandpassfilter_args,
                normalize_args):
        parser_add_optgroup_from_def(parser, src)
    parser_add_optgroup_from_def(parser, single_required_hdf5output)


def run(args):
    if args.chunks is not None:
        # apply global "chunks" setting
        for cattr in ('detrend_chunks', 'zscore_chunks'):
            if getattr(args, cattr) is None:
                # only overwrite if individual option is not given
                args.__setattr__(cattr, args.chunks)
    ds = arg2ds(args.data)
    if args.poly_detrend is not None:
        if args.detrend_chunks is not None \
           and not args.detrend_chunks in ds.sa:
            raise ValueError(
                "--detrend-chunks attribute '%s' not found in dataset"
                % args.detrend_chunks)
        from mvpa2.mappers.detrend import poly_detrend
        verbose(1, "Detrend")
        poly_detrend(ds, polyord=args.poly_detrend,
                     chunks_attr=args.detrend_chunks,
                     opt_regs=args.detrend_regrs,
                     space=args.detrend_coords)
    if args.filter_passband is not None:
        from mvpa2.mappers.filters import iir_filter
        from scipy.signal import butter, buttord
        if args.sampling_rate is None or args.filter_stopband is None:
            raise ValueError(
                "spectral filtering requires specification of "
                "--filter-stopband and --sampling-rate")
        # determine filter type
        nyquist = args.sampling_rate / 2.0
        if len(args.filter_passband) > 1:
            btype = 'bandpass'
            if not len(args.filter_passband) == len(args.filter_stopband):
                raise ValueError("passband and stopband specifications have to "
                        "match in size")
            wp = [v / nyquist for v in args.filter_passband]
            ws = [v / nyquist for v in args.filter_stopband]
        elif args.filter_passband[0] < args.filter_stopband[0]:
            btype = 'lowpass'
            wp = args.filter_passband[0] / nyquist
            ws = args.filter_stopband[0] / nyquist
        elif args.filter_passband[0] > args.filter_stopband[0]:
            btype = 'highpass'
            wp = args.filter_passband[0] / nyquist
            ws = args.filter_stopband[0] / nyquist
        else:
            raise ValueError("invalid specification of Butterworth filter")
        # create filter
        verbose(1, "Spectral filtering (%s)" % (btype,))
        try:
            ord, wn = buttord(wp, ws,
                              args.filter_passloss,
                              args.filter_stopattenuation,
                              analog=False)
            b, a = butter(ord, wn, btype=btype)
        except OverflowError:
            raise ValueError("cannot contruct Butterworth filter for the given "
                             "specification")
        ds = iir_filter(ds, b, a)

    if args.zscore:
        from mvpa2.mappers.zscore import zscore
        verbose(1, "Z-score")
        zscore(ds, chunks_attr=args.zscore_chunks,
               params=args.zscore_params)
        verbose(3, "Dataset summary %s" % (ds.summary()))
    # invariants?
    if args.strip_invariant_features is not None:
        from mvpa2.datasets.miscfx import remove_invariant_features
        ds = remove_invariant_features(ds)
    # and store
    ds2hdf5(ds, args.output, compression=args.hdf5_compression)
    return ds
