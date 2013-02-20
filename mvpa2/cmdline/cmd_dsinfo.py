# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Describe a dataset's content

This command generates a comprehensive description of a dataset's content in
text format and writes it to STDOUT.

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
    import parser_add_common_args, hdf2ds, parser_add_optgroup_from_def

parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

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
    if not args.xlim is None:
        pl.xlim(*args.xlim)
    if not args.ylim is None:
        pl.ylim(*args.ylim)
    for opt, fx in ((args.x_marker, pl.axvline),
                    (args.y_marker, pl.axhline)):
        if not opt is None:
            for val in opt:
                fx(val, linestyle='--')
    if not args.figure_title is None:
        pl.title(args.figure_title)
    pl.show()

info_fx = {
        'content_summary' : txt_content_summary_terse,
        'sample_histogram' : sample_histogram,
}

output_grp = ('options for output formating', [
    (('--style',), dict(type=str, choices=info_fx.keys(),
        default='content_summary', metavar='MODE',
        help="""info type""")),
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
    parser_add_common_args(parser, pos=['multidata'])
    parser_add_optgroup_from_def(parser, output_grp)
    parser_add_optgroup_from_def(parser, ds_descr_grp)


def run(args):
    dss = hdf2ds(args.data)
    verbose(3, 'Loaded %i dataset(s)' % len(dss))
    ds = vstack(dss)
    verbose(3, 'Concatenation yielded %i samples with %i features' % ds.shape)
    info_fx[args.style](ds, args)
