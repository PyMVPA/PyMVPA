# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Scatter plot PyMVPA datasets and/or brain volumes

"""

# magic line for manpage summary
# man: -*- % scatter plot PyMVPA datasets and/or brain volumes

__docformat__ = 'restructuredtext'

from mvpa2.base import verbose
from mvpa2.misc.cmdline import opts, parser

import sys, os
import pylab as pl
from optparse import OptionParser, Option

from mvpa2.misc.plot.scatter import plot_scatter_files

def setup_parser(parser):
    parser.add_argument(
        "-t", "--volume", metavar="volume",
        help="If 4D image given which volume to plot.  If 5D with rudimentary "
        "4th, it gets removed. Default -- all")
    parser.add_argument(
        "-m", "--mask-file", metavar="mask_file",
        help="Filename to use as a mask to decide which voxels to plot.")
    parser.add_argument(
        "--thresholds", metavar="thresholds",
        help="How to threshold the mask volume.  Single value specifies "
             "lower threshold. Two comma-separated values specify exclusion "
             "range: e.g. '-3,3' would include all abs values >=3.  '3,-3' "
             "would then include all abs values < 3")
    parser.add_argument(
        "-M", "--masked-opacity", type=float,
        metavar="masked_opacity", default=0.,
        help="Opacity at which plot masked-out points.  Default is 0, i.e."
             " when they are not plotted at all")
    parser.add_argument("-u", "--unique-points",
        default=False,
        help="Plot those points which are present only in 1 of the volumes "
             "and not in the other along corresponding axis")
    parser.add_argument("-l", "--limits",
        choices=['auto', 'same', 'per-axis'],
        default='auto',
        help="How to decide on limits for the axes. When 'auto' -- if data "
              "ranges overlap is more than 50%% of the union range, 'same' is "
              "considered.")
    parser.add_argument("-x", "--x-jitter", type=float,
        metavar="x_jitter", default=0.,
        help="Half-width of the uniform jitter to add to x-coords. Useful"
             " for quantized (thus overlapping) data")
    parser.add_argument("-y", "--y-jitter", type=float,
        metavar="y_jitter", default=0.,
        help="Half-width of the uniform jitter to add to y-coords. Useful"
             " for quantized (thus overlapping) data")
    parser.add_argument("-o", "--output-img",
                        help="Where to output png of the scatterplot. If selected,"
                             "no figure will be shown")
##
## #    parser.usage = "%s [OPTIONS] FILE1 FILE2 [FILE...] \n\n" % sys.argv[0] + __doc__
## #    parser.option_groups = [opts.common]

    return parser

def run(args):
    """
    TODO:

    - proper cmdline options
    - add specification/use of the mask
    - Michael: fa option which would specify color
    """

    if '_ip' in dir():
        # are we in interactive ipython... lets filter out sys.argv
        sys.argv = sys.argv[:1]

    print args
    #(opts, files) = parser.parse_args()

    #files = [
    #    '/data/famface/subjects/km00/results/mvpa110523-ev+2_3/gauss_fwhm_4.0/mni/svmsl3_gross_acc-chance_familiarity_noself_cvidentity.nii.gz',
    #    '/data/famface/subjects/em00/results/mvpa110523-ev+2_3/gauss_fwhm_4.0/mni/svmsl3_gross_acc-chance_familiarity_noself_cvidentity.nii.gz'
    ##    '/data/famface/subjects/kl00/results/mvpa110523-ev+2_3/gauss_fwhm_4.0/svmsl3_gross_acc-chance_familiarity_noself_cvidentity.nii.gz',
    ##    '/data/famface/subjects/kl00/results/mvpa110523-ev+2_3/gauss_fwhm_4.0/gnbsl3_acc-chance_familiarity_noself_cvidentity.nii.gz',
    #    #'/data/famface/subjects/kl00/results/mvpa110714-ev+2_3/gauss_fwhm_4.0/gnbsl3_familiarity_noself_cvidentity_acc-chance.nii.gz'
    #    ]

    ## files = [
    ##     '/data/famface/subjects/sz00/results/mvpa110716-ev+1_3/gauss_fwhm_4.0/mni/similarity:all_sl3_identity_noself_correlation:correlation.nii.gz',
    ##     '/data/famface/subjects/sz00/results/mvpa110716-ev+1_3/gauss_fwhm_4.0/mni/similarity:all5_sl3_identity_noself_correlation:spearmanr.nii.gz']

    files_ = [
        '/data/famface/results/mvpa111229+tempderivs2-ev+1_3-23.good/__best_betas4__sl3___good_bestbad3/t-tests/mni/_familiarity_noself_cvidentity_gross_acc-chance_z.nii.gz',
        '/data/famface/results/mvpa111229+tempderivs2-ev+1_3-23.good/__best_betas4__sl3___good_bestbad3/t-tests/mni/_familiarity_noself_cvidentity_gross_acc-chance_bestm.nii.gz',
        ]
    assert(len(files) >= 2)             # just 2 files
    if opts.thresholds:
        opts.thresholds = [float(x) for x in opts.thresholds.split(',')]
    plot_scatter_files(files,
                       mask_file=opts.mask_file,
                       masked_opacity=opts.masked_opacity,
                       volume=opts.volume,
                       thresholds=opts.thresholds,
                       limits=opts.limits,
                       x_jitter=opts.x_jitter,
                       y_jitter=opts.y_jitter,
                       uniq=opts.unique_points)
    pl.show()
