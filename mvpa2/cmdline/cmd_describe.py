# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Describe a dataset's content

This command can produce a number of reports for datasets. Currently
supported are summary statistics in text format, as well as basic plots.
See the --report option for more information.

"""

# magic line for manpage summary
# man: -*- % describe a dataset's content

__docformat__ = 'restructuredtext'

import numpy as np
import sys
import argparse
from mvpa2.base import verbose, warning, error
from mvpa2.datasets import vstack
from mvpa2.base.dochelpers import _indent
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
    import arg2ds, parser_add_optgroup_from_def, parser_add_common_opt

parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

def arg2transform(args):
    args = args.split(':')
    if not len(args) == 2:
        raise ValueError("--numpy-xfm needs exactly two arguments")
    if not args[0] in ('samples', 'features'):
        raise ValueError("transformation axis must be 'samples' or 'features' (was: %s)"
                         % args[0])
    axis = args[0]
    if not hasattr(np, args[1]):
        raise ValueError("the NumPy package does not have a '%s' function" % args[1])
    fx = getattr(np, args[1])
    return fx, axis

def _limit_lines(string, maxlines):
    lines = string.split('\n')
    truncated = '\n'.join(lines[:maxlines])
    if len(lines) > maxlines:
        truncated += ' ...'
    return truncated

def _describe_samples(samp, style):
    if style == 'terse':
        return "%s@%s\n" % (samp.shape, samp.dtype)
    else:
        return 'IMPLEMENT ME\n'

def _describe_array_attr(attr, style):
    if len(attr.value.shape) == 1:
        shape = attr.value.shape[0]
    else:
        shape = attr.value.shape
    if style == 'terse':
        return '%s %s@%s' % (attr.name, shape, attr.value.dtype)
    else:
        return 'IMPLEMENT ME\n'

def _describe_attr(attr, style):
    if style == 'terse':
        return '%s %s' % (attr.name,
                          _limit_lines(str(attr.value), 1))
    else:
        return 'IMPLEMENT ME\n'

def txt_content_summary_terse(ds, args):
    print ds.summary(targets_attr=args.target_attr)
    info = '\n\nDetails on dataset attributes:\n'
    for cdesc, col, describer in \
            (('sample', ds.sa, _describe_array_attr),
             ('feature', ds.fa, _describe_array_attr),
             ('dataset', ds.a, _describe_attr)):
        info += ' %s attributes:\n' % cdesc
        for attr in sorted(col.values(),
                           cmp=lambda x, y: cmp(x.name, y.name)):
            info += '  %s\n' % describer(attr, 'terse')
    print info

def sample_histogram(ds, args):
    import pylab as pl
    pl.figure()
    pl.hist(np.ravel(ds.samples), bins=args.histogram_bins)
    if args.xlim is not None:
        pl.xlim(*args.xlim)
    if args.ylim is not None:
        pl.ylim(*args.ylim)
    for opt, fx in ((args.x_marker, pl.axvline),
                    (args.y_marker, pl.axhline)):
        if opt is not None:
            for val in opt:
                fx(val, linestyle='--')
    if args.figure_title is not None:
        pl.title(args.figure_title)
    pl.show()

info_fx = {
        'txtsummary' : txt_content_summary_terse,
        'sample_histogram' : sample_histogram,
}

xfm_grp = ('options for transforming dataset content before plotting', [
    (('--numpy-xfm',), dict(type=arg2transform, metavar='SPEC',
        help="""apply a Numpy function along a given axis of the samples before
        generating the dataset info summary. For example, 'samples:std' will
        apply the 'std' function along the samples axis, i.e. compute a vector
        of standard deviations for all features in a dataset""")),
])

output_grp = ('options for plot formatting', [
    (('--figure-title',), dict(type=str,
        help="""title for a plot""")),
    (('--histogram-bins',), dict(type=int, default=20,
        metavar='VALUE',
        help="""number of bin for histograms""")),
    (('--xlim',), dict(type=float, nargs=2,
        help="""minimum and maximum value of the x-axis extent in a figure""")),
    (('--ylim',), dict(type=float, nargs=2,
        help="""minimum and maximum value of the y-axis extent in a figure""")),
    (('--x-marker',), dict(type=float, nargs='+',
        help="""list of x-value to draw markers on in a figure""")),
    (('--y-marker',), dict(type=float, nargs='+',
        help="""list of y-value to draw markers on in a figure""")),
])


ds_descr_grp = ('options for dataset description', [
    (('--target-attr',), dict(default='targets', metavar='NAME',
        help="""name of a samples attributes defining 'target'. This
        information is used to define groups of samples when
        generating information on the within and between category
        data structure in a dataset.""")),
])

def setup_parser(parser):
    parser_add_common_opt(parser, 'multidata', required=True)
    parser.add_argument('-r', '--report',
            **dict(type=str, choices=info_fx.keys(),
                 default='txtsummary',
                 help="""choose a type of report. Default: terse summary in
                 text format."""))
    parser_add_optgroup_from_def(parser, xfm_grp)
    parser_add_optgroup_from_def(parser, output_grp)
    parser_add_optgroup_from_def(parser, ds_descr_grp)


def run(args):
    ds = arg2ds(args.data)
    verbose(3, 'Concatenation yielded %i samples with %i features' % ds.shape)
    if args.numpy_xfm is not None:
        from mvpa2.mappers.fx import FxMapper
        fx, axis = args.numpy_xfm
        mapper = FxMapper(axis, fx)
        ds = ds.get_mapped(mapper)
    info_fx[args.report](ds, args)
