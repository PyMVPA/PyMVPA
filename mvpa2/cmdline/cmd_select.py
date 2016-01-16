# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Select a subset of samples and/or features from a dataset

A subset of samples and/or feature can be selected by their indices (see
--samples/features-by-index) or via simple expressions evaluating attribute
value (see --samples/features-by-attr). It is possible to specify options
for selecting samples and features simultaneously. It is also possible to strip
arbitrary attributes from the output dataset (see --strip-...).

SELECTION BY INDEX

All --...-by-index options accept a sequence of integer indices. Alternatively
it is possible to specify regular sequences of indices using a START:STOP:STEP
notation (zero-based). For example, ':5' selects the first five elements, '2:4'
selects the third and fourth element, and ':20:2' selects all even numbered
elements from the first 20.

SELECTION BY ATTRIBUTE

All --...by-attr options support a simple expression language that allows for
creating filters/masks from attribute values. Such selection expressions are
made up of ATTRIBUTE OPERATOR VALUE triplets that can be combined via 'and' or
'or' keywords. For example:

... --samples-by-attr subj eq 5 and run lt 5 or run gt 10

selects all samples where attribute 'subj' equals 5 and the run attribute is
either less than 5 or greater than 10. 'and' and 'or' operations are done in
strictly serial order (no nested conditions).

Supported operators are:

eq (equal)
ne (not equal)
ge (greater or equal)
le (less or equal)
gt (greater than)
lt (less than)

"""

# magic line for manpage summary
# man: -*- % select a subset of samples and/or features from a dataset

__docformat__ = 'restructuredtext'

import numpy as np
import sys
import argparse
from mvpa2.base import verbose, warning, error
from mvpa2.datasets import Dataset, vstack
from mvpa2.mappers.fx import FxMapper
from mvpa2.datasets.eventrelated import eventrelated_dataset, find_events
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
    import parser_add_common_opt, \
           ds2hdf5, hdf2ds

# map operator symbols to method names
attr_operators = {
    'eq': '__eq__',
    'ne': '__ne__',
    'ge': '__ge__',
    'le': '__le__',
    'gt': '__gt__',
    'lt': '__lt__',
}

filter_joiner = {
    'and': np.logical_and,
    'or': np.logical_or,
}

def _eval_attr_expr(expr, col):
    #read from the front
    rev_expr = expr[::-1]
    # current filter -- take all
    actfilter = np.ones(len(col[col.keys()[0]]), dtype='bool')
    joiner = None
    while len(rev_expr):
        ex = rev_expr.pop()
        if not ex in ('and', 'or'):
            # eval triplet (attr, operator, value)
            attr = ex
            try:
                attr = col[attr].value
            except KeyError:
                raise ValueError("unknown attribute '%s' in expression [%s]. Valid attributes: %s"
                                 % (attr, ' '.join(expr), col.keys())) 
            op = rev_expr.pop()
            try:
                op = attr_operators[op]
            except KeyError:
                raise ValueError("unknown operator '%s' in expression [%s]. Valid operators: %s"
                                 % (op, ' '.join(expr), attr_operators.keys())) 
            val = rev_expr.pop()
            # convert value into attr dtype
            try:
                val = attr.dtype.type(val)
            except ValueError:
                raise ValueError("can't convert '%s' into attribute data type '%s' in expression [%s]"
                                 % (val, attr.dtype, ' '.join(expr))) 
            # evaluate expression
            newfilter = getattr(attr, op)(val)
            # merge with existing filter
            if joiner is None:
                # replace
                actfilter = newfilter
            else:
                # join
                actfilter = joiner(actfilter, newfilter)
        else:
            joiner = filter_joiner[ex]
    return actfilter

parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

samples_grp = ('options for selecting samples', [
    (('--samples-by-index',), dict(type=str, nargs='+', metavar='IDX',
        help="""select a subset of samples by index. See section 'SELECTION BY
        INDEX' for more details."""
        )),
    (('--samples-by-attr',), dict(type=str, nargs='+', metavar='EXPR',
        help="""select a subset of samples by attribute evaluation. See section
        'SELECTION BY ATTRIBUTE' for more details."""
        )),
])

features_grp = ('options for selecting features', [
    (('--features-by-index',), dict(type=str, nargs='+', metavar='IDX',
        help="""select a subset of features by index. See section 'SELECTION BY
        INDEX' for more details."""
        )),
    (('--features-by-attr',), dict(type=str, nargs='+', metavar='EXPR',
        help="""select a subset of features by attribute evaluation. See section
        'SELECTION BY ATTRIBUTE' for more details."""
        )),
])


strip_grp = ('options for removing attributes', [
    (('--strip-sa',), dict(type=str, nargs='+', metavar='ATTR',
        help="""strip one or more samples attributes given by their name from
        a dataset."""
        )),
    (('--strip-fa',), dict(type=str, nargs='+', metavar='ATTR',
        help="""strip one or more feature attributes given by their name from
        a dataset."""
        )),
    (('--strip-da',), dict(type=str, nargs='+', metavar='ATTR',
        help="""strip one or more dataset attributes given by their name from
        a dataset."""
        )),
])

def setup_parser(parser):
    from .helpers import parser_add_optgroup_from_def, \
        parser_add_common_attr_opts, single_required_hdf5output
    parser_add_common_opt(parser, 'multidata', required=True)
    parser_add_optgroup_from_def(parser, samples_grp, exclusive=True)
    parser_add_optgroup_from_def(parser, features_grp, exclusive=True)
    parser_add_optgroup_from_def(parser, strip_grp)
    parser_add_optgroup_from_def(parser, single_required_hdf5output)

def run(args):
    dss = hdf2ds(args.data)
    verbose(3, 'Loaded %i dataset(s)' % len(dss))
    ds = vstack(dss)
    verbose(3, 'Concatenation yielded %i samples with %i features' % ds.shape)
    # slicing
    sliceme = {'samples': slice(None), 'features': slice(None)}
    # indices
    for opt, col, which in ((args.samples_by_index, ds.sa, 'samples'),
                     (args.features_by_index, ds.fa, 'features')):
        if opt is None:
            continue
        if len(opt) == 1 and opt[0].count(':'):
            # slice spec
            arg = opt[0].split(':')
            spec = []
            for a in arg:
                if not len(a):
                    spec.append(None)
                else:
                    spec.append(int(a))
            sliceme[which] = slice(*spec)
        else:
            # actual indices
            sliceme[which] = [int(o) for o in opt]
    # attribute evaluation
    for opt, col, which in ((args.samples_by_attr, ds.sa, 'samples'),
                     (args.features_by_attr, ds.fa, 'features')):
        if opt is None:
            continue
        sliceme[which] = _eval_attr_expr(opt, col)

    # apply selection
    ds = ds.__getitem__((sliceme['samples'], sliceme['features']))
    verbose(1, 'Selected %i samples with %i features' % ds.shape)

    # strip attributes
    for attrarg, col, descr in ((args.strip_sa, ds.sa, 'sample '),
                                (args.strip_fa, ds.fa, 'feature '),
                                (args.strip_da, ds.a, '')):
        if attrarg is not None:
            for attr in attrarg:
                try:
                    del col[attr]
                except KeyError:
                    warning("dataset has no %sattribute '%s' to remove"
                            % (descr, attr))
    # and store
    ds2hdf5(ds, args.output, compression=args.hdf5_compression)
    return ds
