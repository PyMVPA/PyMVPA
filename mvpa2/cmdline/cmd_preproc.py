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

2. Feature-wise Z-scoring

All preprocessing steps are applied in the above order. If a different order is
required, preprocessing has to be split into two separate command calls.

Examples:

Normalize all features in a dataset by Z-scoring

  $ pymvpa2 preproc --zscore -o ds_preprocessed dataset.hdf5

Perform Z-scoring and quadratic detrending of all features, but process all
samples sharing a unique value of the "chunks" sample attribute individually

  $ pymvpa2 preproc --chunks "chunks" --poly-detrend 2 --zscore -o ds_pp2 ds.hdf5

"""

# magic line for manpage summary
# man: -*- % apply preprocessing steps to a PyMVPA dataset

__docformat__ = 'restructuredtext'

import numpy as np
import argparse
from mvpa2.base.hdf5 import h5save, h5load
from mvpa2.base import verbose, warning, error
from mvpa2.datasets import Dataset
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
        import parser_add_common_args, parser_add_common_opt, args2datasets

parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

detrend_args = ('options for data detrending', [
    ('--poly-detrend',
     dict(type=int, metavar='DEG',
          help="""perform detrending by regressing out polynomials trends up
               to order DEG.""")),
    ('--detrend-chunks',
     dict(type=str, default=None, metavar='CHUNKS_ATTR',
          help="""name of a dataset sample attribute defining chunks of
               samples that shall be detrended independently. By default
               all no chunk-wise detrending is done.""")),
    ('--detrend-coords',
     dict(type=str, metavar='COORDS_ATTR',
         help="""name of a samples attribute that is added to the
              preprocessed dataset storing the coordinates of each sample in the
              space spanned by the polynomials. If an attribute of such name
              is already present in the dataset its values are interpreted
              as sample coordinates in the space spanned by the polynomials.
              This can be used to detrend datasets with irregular sample
              spacing.""")),
    ('--detrend-regrs',
     dict(type=str, nargs='+', metavar='ATTR',
          help="""names of sample attributes that shall serve as an additional
          regressor during detrending. This can be used to, for example, regress
          out motion-related confound in fMRI data."""))
])

normalize_args = ('options for data normalization', [
    ('--zscore',
     dict(action='store_true',
          help="""perform feature normalization by Z-scoring.""")),
    ('--zscore-chunks',
     dict(metavar='CHUNKS_ATTR',
          help="""name of a dataset sample attribute defining chunks of
               samples that shall be Z-scored independently. By default
               no chunk-wise normalization is done.""")),
    ('--zscore-params',
     dict(metavar='PARAM', nargs=2, type=float,
          help="""define a fixed parameter set (mean, std) for Z-scoring,
               instead of computing from actual data.""")),
])

common_args = ('common options for all preprocessing', [
    ('--chunks',
     dict(metavar='CHUNKS_ATTR',
          help="""shortcut option to enabled uniform chunkwise processing for
               all relevant preprocessing steps (see --zscore-chunks,
               --detrend-chunks). This global setting can be overwritten by
               additionally specifying the corresponding individual "chunk"
               options.""")),
])

def setup_parser(parser):
    parser_add_common_args(parser, pos=['data'])
    # order of calls is relevant!
    for src in (common_args, detrend_args, normalize_args):
        srcgrp = parser.add_argument_group(src[0])
        for opt in src[1]:
            srcgrp.add_argument(opt[0], **opt[1])
    outputgrp = parser.add_argument_group('output options')
    parser_add_common_opt(outputgrp, 'output_file', required=True)
    parser_add_common_args(outputgrp, opt=['hdf5compression'])


def run(args):
    if not args.chunks is None:
        # apply global "chunks" setting
        for cattr in ('detrend_chunks', 'zscore_chunks'):
            if getattr(args, cattr) is None:
                # only overwrite if individual option is not given
                args.__setattr__(cattr, args.chunks)
    ds = h5load(args.data)
    if not args.poly_detrend is None:
        if not args.detrend_chunks is None \
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
    if args.zscore:
        from mvpa2.mappers.zscore import zscore
        verbose(1, "Z-score")
        zscore(ds, chunks_attr=args.zscore_chunks,
               params=args.zscore_params)
        verbose(3, "Dataset summary %s" % (ds.summary()))
    # and store
    outfilename = args.output
    if not outfilename.endswith('.hdf5'):
        outfilename += '.hdf5'
    verbose(1, "Save dataset to '%s'" % outfilename)
    h5save(outfilename, ds, mkdir=True, compression=args.compression)

    return ds
