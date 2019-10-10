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

import pylab as pl
from mvpa2.cmdline.helpers \
    import parser_add_common_opt, ds2hdf5, hdf2ds
from mvpa2.misc.plot.scatter import plot_scatter_files

scatterplot_opts_grp = ('options for scatterplot', [
    (('data',),
     dict(nargs='+',
          help='Nifti datasets to scatterplot')),
    (('--volume', '-t'), dict(
        type=int,
        help="If 4D image given which volume to plot.  If 5D with rudimentary "
             "4th, it gets removed. Default -- all")),
    (("-m", "--mask-file"),
        dict(help="Filename to use as a mask to decide which voxels to plot.")),
    (("--thresholds", ),
        dict(help="How to threshold the mask volume.  Single value specifies "
             "lower threshold. Two comma-separated values specify exclusion "
             "range: e.g. '-3,3' would include all abs values >=3.  '3,-3' "
             "would then include all abs values < 3")),
    (("-M", "--masked-opacity"),
        dict(type=float, default=0.,
             help="Opacity at which plot masked-out points.  Default is 0, i.e."
                  " when they are not plotted at all")),
    (("-u", "--unique-points"),
        dict(default=False,
            help="Plot those points which are present only in 1 of the volumes "
                 "and not in the other along corresponding axis")),
    (("-l", "--limits"),
        dict(choices=['auto', 'same', 'per-axis'], default='auto',
             help="How to decide on limits for the axes. When 'auto' -- if data "
             "ranges overlap is more than 50%% of the union range, 'same' is "
             "considered.")),
    (("-x", "--x-jitter"),
        dict(type=float, default=0.,
             help="Half-width of the uniform jitter to add to x-coords. Useful"
                  " for quantized (thus overlapping) data")),
    (("-y", "--y-jitter"),
        dict(type=float, default=0.,
             help="Half-width of the uniform jitter to add to y-coords. Useful"
                  " for quantized (thus overlapping) data")),
    (("-s", "--stats"),
     dict(default=False,
          help="Whether to print additional stats on the data")),
    (("-o", "--output-img"),
        dict(help="Where to output png of the scatterplot.")),
])


def setup_parser(parser):
    from .helpers import parser_add_optgroup_from_def
    parser_add_optgroup_from_def(parser, scatterplot_opts_grp)


def run(args):
    plot_scatter_files(args.data,
                       mask_file=args.mask_file,
                       masked_opacity=args.masked_opacity,
                       volume=args.volume,
                       thresholds=args.thresholds,
                       limits=args.limits,
                       x_jitter=args.x_jitter,
                       y_jitter=args.y_jitter,
                       uniq=args.unique_points,
                       include_stats=args.stats)
    if args.output_img:
        pl.savefig(args.output_img)
    pl.show()
