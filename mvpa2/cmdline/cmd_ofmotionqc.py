# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Generate a QC plot for BOLD fMRI motion estimates from an OpenFMRI dataset

"""

# magic line for manpage summary
# man: -*- % BOLD fMRI motion QC plot for an OpenFMRI dataset

import mvpa2

__docformat__ = 'restructuredtext'

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
            '--exclude-subjs', metavar='ID', nargs='+',
            help="""list of space-separated suject IDs to exclude from the QC
            analysis""")
    parser.add_argument(
            '--outlier-minthresh', type=float, default=None,
            help="""minimum absolute threshold for outlier detection.
            Default: None""")
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

    of = OpenFMRIDataset(args.path)
    data = of.get_task_bold_attributes(
            args.task, args.estimate_fname, np.loadtxt,
            exclude_subjs=args.exclude_subjs)
    segment_sizes = [len(d[0]) for d in data]

    # figure setup
    pl.figure(figsize=(12, 5))
    ax = pl.subplot(211)

    # translation
    run_stats = [compute_ts_boxplot_stats(
                    d[...,:3],
                    outlier_abs_minthresh=args.outlier_minthresh,
                    outlier_thresh=args.outlier_stdthresh,
                    aggfx=np.linalg.norm,
                    greedy_outlier=True)
                        for d in data]
    stats = concat_ts_boxplot_stats(run_stats)

    timeseries_boxplot(stats[0]['median'],
                mean=stats[0]['mean'], std=stats[0]['std'], n=stats[0]['n'],
                min=stats[0]['min'], max=stats[0]['max'],
                p25=stats[0]['p25'], p75=stats[0]['p75'],
                outlierd=stats[1], segment_sizes=segment_sizes)
    pl.title('translation')
    xp, xl = pl.xticks()
    pl.xticks(xp, ['' for i in xl])
    pl.xlim((0, len(stats[0]['n'])))
    #pl.ylim((0,7))
    pl.ylabel('estimate L2-norm in mm')

    # rotation
    ax = pl.subplot(212)
    run_stats = [compute_ts_boxplot_stats(
                    d[...,3:],
                    outlier_abs_minthresh=args.outlier_minthresh,
                    outlier_thresh=args.outlier_stdthresh,
                    aggfx=np.linalg.norm,
                    greedy_outlier=True)
                        for d in data]
    stats = concat_ts_boxplot_stats(run_stats)

    timeseries_boxplot(stats[0]['median'],
                mean=stats[0]['mean'], std=stats[0]['std'], n=stats[0]['n'],
                min=stats[0]['min'], max=stats[0]['max'],
                p25=stats[0]['p25'], p75=stats[0]['p75'],
                outlierd=stats[1], segment_sizes=segment_sizes)

    pl.xlim((0, len(stats[0]['n'])))
    #pl.ylim((0,5))
    pl.title('rotation')
    pl.ylabel('estimate L2-norm in deg')
    pl.xlabel('time in fMRI volumes')

    if args.savefig is None:
        pl.show()
    else:
        pl.savefig(args.safefig)
