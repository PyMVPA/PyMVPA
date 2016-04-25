# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Generate a QC plot for (BOLD fMRI) motion estimates of multiple segments.

The generated figure consists of two subplots: one for translation and one for
rotation. The L2-norm for each motion type is plotted. Segment boundaries
are indicated with dashed vertical lines. The following statistics are
visualized

1. Range across subjects (min, max) with a light gray shaded area

2. 50% percentile around the mean with a medium gray shaded area

3. Standard error of the mean (SEM; +/-) with a dark gray shaded area

4. Median across subjects with a black line

5. Outlier subjects are represented as individual red lines

Outliers are defined as subjects that exceed the threshold anywhere within
a given segment. In that case the entire motion time series for that segment is
plotted as an outlier.

Example

pymvpa2 plotmotionqc \
    -s sub*/func/*run-1_bold_mc.txt \
    -s sub*/func/*run-2_bold_mc.txt \
    --savefig motion.png

"""

# magic line for manpage summary
# man: -*- % BOLD fMRI motion QC plot

__docformat__ = 'restructuredtext'

import argparse
import numpy as np

parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}


def setup_parser(parser):
    parser.add_argument(
        '-s', '--segment', metavar='FILE', type=np.loadtxt, nargs='+',
        action='append',
        help="""two or more text files with motion estimate time series.
        This option can be given multiple times (with multiple time series
        each to generate a multi-segment plot (e.g. for multiple run).""")
    parser.add_argument(
        '--estimate-order', metavar='LABEL', default='transrot',
        choices=('transrot', 'rottrans'),
        help="""column order of estimates in the files. `transrot` indicates
        translation first, followed by rotation. `rottrans` refers to the
        oposite order. [Default: 'transrot']""")
    parser.add_argument(
        '--rad2deg', action='store_true',
        help="""If specified, rotation estimates are assumed to be in radian
        and will be converted to degrees.""")
    parser.add_argument(
        '--outlier-minthresh', type=float, default=None,
        help="""absolute minimum threshold of outlier detection. Only value
        larger than this this threshold will ever be considered as an
        outlier. [Default: None]""")
    parser.add_argument(
        '--outlier-stdthresh', type=float, default=None,
        help="""minimum threshold in units of standard deviation
        for outlier detection. [Default: None]""")
    parser.add_argument(
        '--savefig', metavar='FILENAME', nargs=1,
        help="""file name to store the QC figure under. Without this option
        the figure is shown in an interactive viewer.""")
    return parser


def motionqc_plot(data, outlier_abs_minthresh=None, outlier_stdthresh=None, ylabel=None):
    import pylab as pl
    from mvpa2.misc.plot import timeseries_boxplot, concat_ts_boxplot_stats
    from mvpa2.misc.stats import compute_ts_boxplot_stats

    # segments x [subjects x timepoints x props]
    segment_sizes = [d.shape[1] for d in data]

    # get stats for all segments and concatenate them
    stats = concat_ts_boxplot_stats(
        [compute_ts_boxplot_stats(
            d,
            outlier_abs_minthresh=outlier_abs_minthresh,
            outlier_thresh=outlier_stdthresh,
            aggfx=np.linalg.norm,
            greedy_outlier=True)
            for d in data])

    outlier = None
    if outlier_stdthresh:
        outlier = [list(np.where(np.sum(np.logical_not(o.mask), axis=0))[0])
                   for o in stats[1]]

    # plot
    timeseries_boxplot(
        stats[0]['median'],
        mean=stats[0]['mean'],
        std=stats[0]['std'],
        n=stats[0]['n'],
        min=stats[0]['min'],
        max=stats[0]['max'],
        p25=stats[0]['p25'],
        p75=stats[0]['p75'],
        outlierd=stats[1],
        segment_sizes=segment_sizes)
    xp, xl = pl.xticks()
    pl.xticks(xp, ['' for i in xl])
    pl.xlim((0, len(stats[0]['n'])))
    if ylabel:
        pl.ylabel(ylabel)

    pl.xlabel('time')

    return outlier


def run(args):
    import pylab as pl
    from mvpa2.base import verbose

    # segments x [subjects x timepoints x properties]
    data = [np.array(s) for s in args.segment]

    # put in standard property order: first translation, then rotation
    if args.estimate_order == 'rottrans':
        data = [d[:, :, (3, 4, 5, 0, 1, 2)] for d in data]

    # convert rotations, now known to be last
    if args.rad2deg:
        for d in data:
            v = d[:, :, 3:]
            np.rad2deg(v, v)

    # and plot
    # figure setup
    fig = pl.figure(figsize=(12, 5))
    # translation
    ax = pl.subplot(211)
    outlier = motionqc_plot(
        [d[..., :3] for d in data],
        args.outlier_minthresh,
        args.outlier_stdthresh,
        "translation\nestimate L2-norm")
    if outlier:
        verbose(
            0,
            "Detected per-segment translation outlier input samples {0} (zero-based)".format(
                outlier))
    # rotation
    ax = pl.subplot(212)
    outlier = motionqc_plot(
        [d[..., 3:] for d in data],
        args.outlier_minthresh,
        args.outlier_stdthresh,
        "rotation\nestimate L2-norm")
    if outlier:
        verbose(
            0,
            "Detected per-segment rotation outlier input samples {0} (zero-based)".format(
                outlier))

    if args.savefig is None:
        pl.show()
    else:
        pl.savefig(args.savefig[0])
