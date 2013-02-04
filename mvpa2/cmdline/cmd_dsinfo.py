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

output_grp = ('options for output formating', [
    (('--style',), dict(type=str, choices=('terse', 'full'), default='terse',
        metavar='MODE',
        help="""output format style: only 'terse' is implemented for now"""
        )),
])



def setup_parser(parser):
    parser_add_common_args(parser, pos=['multidata'])
    parser_add_optgroup_from_def(parser, output_grp)


def run(args):
    dss = hdf2ds(args.data)
    verbose(3, 'Loaded %i dataset(s)' % len(dss))
    ds = vstack(dss)
    verbose(3, 'Concatenation yielded %i samples with %i features' % ds.shape)
    info = 'samples: '
    info += _describe_samples(ds.samples, args.style)
    for cdesc, col, describer in \
            (('sample', ds.sa, _describe_array_attr),
             ('feature', ds.fa, _describe_array_attr),
             ('dataset', ds.a, _describe_attr)):
        info += '%s attributes:\n' % cdesc
        for attr in sorted(col.values(),
                           cmp=lambda x, y: cmp(x.name, y.name)):
            info += '  %s\n' % describer(attr, args.style)
    print info
