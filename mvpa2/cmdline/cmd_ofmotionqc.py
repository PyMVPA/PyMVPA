# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Generate a QC plot for BOLD fMRI motion estimates from an OpenFMRI dataset.

The generated figure consists of two subplots: one for translation and one for
rotation. The L2-norm for each motion type is plotted. Runs boundaries
are indicated with dashed vertical lines. The following statistics are
visualized

1. Range across subjects (min, max) with a light gray shaded area

2. 50% percentile around the mean with a medium gray shaded area

3. Standard error of the mean (SEM; +/-) with a dark gray shaded area

4. Median across subjects with a black line

5. Outlier subjects are represented as individual red lines

Outliers are defined as subjects that exceed the threshold anywhere within
a given run. In that case the entire motion time series for that run is
plotted as an outlier. IDs for detected outliers are printed to the console.
Note that these IDs do not account for missing data or excluded subjects.
However, in case of no missing data and no excluded subjects they do
correspond to numerical subject IDs.

"""

# magic line for manpage summary
# man: -*- % BOLD fMRI motion QC plot for an OpenFMRI dataset

__docformat__ = 'restructuredtext'

import mvpa2
import argparse

parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

def setup_parser(parser):
    parser.add_argument(
            '--path', metavar='PATH', required=True,
            help="""path to the root directory of the OpenFMRI dataset""")
    parser.add_argument(
            '-t', '--task', metavar='ID', type=int, default=1,
            help="""ID of the task with BOLD fMRI data for the QC plot.
            Default: 1""")
    parser.add_argument(
            '--estimate-fname', metavar='NAME', default='bold_moest.txt',
            help="""name of the files with the motion estimates in each BOLD
            task run folder. Default: bold_moest.txt""")
    parser.add_argument(
            '--relative', metavar='REF', type=int, nargs=2,
            help="""recode all estimates as the difference to the estimate for
            volume X in run Y (--relative Y X). Zero-based indices!""")
    parser.add_argument(
            '--exclude-subjs', metavar='ID', type=int, nargs='+',
            help="""list of space-separated suject IDs to exclude from the QC
            analysis""")
    parser.add_argument(
            '--estimate-order', metavar='LABEL', default='transrot',
            choices=('transrot', 'rottrans'),
            help="""column order of estimates in the files. `transrot` indicates
            translation first, followed by rotation. `rottrans` refers to the
            oposite order. Default: 'transrot'""")
    parser.add_argument(
            '--outlier-minthresh', type=float, default=None,
            help="""absolute minimum threshold of outlier detection. Only value
            larger than this this threshold will ever be considered as an
            outlier. Default: None""")
    parser.add_argument(
            '--outlier-stdthresh', type=float, default=None,
            help="""minimum threshold in units of standard deviation
            for outlier detection. Default: None""")
    parser.add_argument(
            '--savefig', default=None,
            help="""file name to store the QC figure under. Default: None""")
    return parser


def run(args):
    import numpy as np
    import pylab as pl
    from mvpa2.datasets.sources.openfmri import OpenFMRIDataset
    from mvpa2.misc.plot import timeseries_boxplot, concat_ts_boxplot_stats
    from mvpa2.misc.stats import compute_ts_boxplot_stats
    from mvpa2.base import verbose

    of = OpenFMRIDataset(args.path)
    data = of.get_task_bold_attributes(
            args.task, args.estimate_fname, np.loadtxt,
            exclude_subjs=args.exclude_subjs)
    segment_sizes = [len(d[d.keys()[0]]) for d in data]

    if not args.relative is None:
        # recode per-subject estimates wrt their particular reference
        ref = {subj: d[args.relative[1]]
               for subj, d in data[args.relative[0]].iteritems()}
        for d in data:
            for subj in d:
                if subj in d:
                    d[subj] -= ref[subj]
                    print subj, d[subj].mean()
    # collapse all data into a per-run (subj x vol x estimate) array
    data = [np.array(d.values()) for d in data]

    # figure setup
    pl.figure(figsize=(12, 5))
    ax = pl.subplot(211)

    plt_props = {
        'translation': 'estimate L2-norm in mm',
        'rotation': 'estimate L2-norm in deg'
    }
    def bxplot(stats, label):
        stats = concat_ts_boxplot_stats(stats)
        verbose(0, "List of outlier time series follows (if any)")
        for i, run in enumerate([np.where(np.sum(np.logical_not(o.mask), axis=0))
                              for o in stats[1]]):
            sids = run[0]
            if len(sids):
                verbose(0, "%s r%.3i: %s"
                           % (label, i + 1, [s + 1 for s in sids]))
        timeseries_boxplot(stats[0]['median'],
                mean=stats[0]['mean'], std=stats[0]['std'], n=stats[0]['n'],
                min=stats[0]['min'], max=stats[0]['max'],
                p25=stats[0]['p25'], p75=stats[0]['p75'],
                outlierd=stats[1], segment_sizes=segment_sizes)
        pl.title(label)
        xp, xl = pl.xticks()
        pl.xticks(xp, ['' for i in xl])
        pl.xlim((0, len(stats[0]['n'])))
        pl.ylabel(plt_props[label])

    # first three columns
    run_stats = [compute_ts_boxplot_stats(
        d[...,:3],
        outlier_abs_minthresh=args.outlier_minthresh,
        outlier_thresh=args.outlier_stdthresh,
        aggfx=np.linalg.norm,
        greedy_outlier=True)
        for d in data]
    ax = pl.subplot(211)
    if args.estimate_order == 'transrot':
        bxplot(run_stats, 'translation')
    else:
        bxplot(run_stats, 'rotation')

    # last three columns
    run_stats = [compute_ts_boxplot_stats(
        d[...,3:],
        outlier_abs_minthresh=args.outlier_minthresh,
        outlier_thresh=args.outlier_stdthresh,
        aggfx=np.linalg.norm,
        greedy_outlier=True)
        for d in data]
    ax = pl.subplot(212)
    if args.estimate_order == 'rottrans':
        bxplot(run_stats, 'translation')
    else:
        bxplot(run_stats, 'rotation')

    pl.xlabel('time in fMRI volumes')

    if args.savefig is None:
        pl.show()
    else:
        pl.savefig(args.savefig)
