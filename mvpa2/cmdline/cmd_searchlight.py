# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Travelling ROI analysis

"""

# magic line for manpage summary
# man: -*- % travelling ROI analysis

__docformat__ = 'restructuredtext'

import numpy as np
import sys
import os
import argparse
from mvpa2.base import verbose, warning, error
from mvpa2.datasets import vstack
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
    import parser_add_common_args, parser_add_common_opt, ds2hdf5, hdf2ds, \
           get_crossvalidation_instance, crossvalidation_opts_grp, \
           arg2neighbor, script2obj

parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

searchlight_opts_grp = ('options for searchlight setup', [
    (('--payload',), dict(required=True,
        help="""switch to select a particular analysis type to be run in a
        searchlight fashion on a dataset. Depending on the choice the
        corresponding analysis setup options are evaluated. 'cv' computes
        a cross-validation analysis. Alternatively, the argument to this option
        can also be a script filename in which a custom measure is built that
        is then ran as a searchlight.""")),
    (('--neighbors',), dict(type=arg2neighbor, metavar='SPEC', action='append',
        required=True,
        help="""define the size and shape of an ROI with respect to a
        center/seed location. If a single integer number is given, it is
        interpreted as the radius (in number of grid elements) around a seed
        location. By default grid coordinates for features are taken from
        a 'voxel_indices' feature attribute in the input dataset. If coordinates
        shall be taken from a different attribute, the radius value can be
        prefixed with the attrubute name, i.e. 'altcoords:2'. For ROI shapes
        other than spheres (with potentially additional parameters), the shape
        name can be specified as well, i.e. 'voxel_indices:HollowSphere:3:2'.
        All neighborhood objects from the mvpa2.misc.neighborhood module are
        supported. For custom ROI shapes it is also possible to pass a script
        filename, or an attribute name plus script filename combination, i.e.
        'voxel_indices:myownshape.py' (advanced). It is possible to specify
        this option multiple times to define multi-space ROI shapes for, e.g.,
        spatio-temporal searchlights.""")),
    (('--nproc',), dict(type=int, default=1,
        help="""Use the specific number or worker processes for computing.""")),
    (('--multiproc-backend',), dict(choices=('native', 'hdf5'),
        default='native',
        help="""Specifies the way results are provided back from a processing
        block in case of --nproc > 1. 'native' is pickling/unpickling of
        results, while 'hdf5' uses HDF5 based file storage. 'hdf5' might be more
        time and memory efficient in some cases.""")),
    (('--aggregate-fx',), dict(type=script2obj,
        help="""use a custom result aggregation function for the searchlight
             """)),
])

searchlight_constraints_opts_grp = ('options for constraining the searchlight', [
    (('--scatter-rois',), dict(type=arg2neighbor, metavar='SPEC',
        help="""scatter ROI locations across the available space. The arguments
        supported by this option are identical to those of --neighbors. ROI
        locations are randomly picked from all possible locations with the
        constraint that the center coordinates of any ROI is NOT within
        the neighborhood (as defined by this option's argument) of a second
        ROI. Increasing the size of the neighborhood therefore increases the
        scarceness of the sampling.""")),
    (('--roi-attr',), dict(metavar='ATTR',
        help="""name of a feature attribute whose non-zero values define
        possible ROI seeds/centers. Conflicts""")),
])

def setup_parser(parser):
    from .helpers import parser_add_optgroup_from_def, \
        parser_add_common_attr_opts, single_required_hdf5output, ca_opts_grp
    parser_add_common_args(parser, pos=['multidata'])
    parser_add_optgroup_from_def(parser, searchlight_opts_grp)
    parser_add_optgroup_from_def(parser, ca_opts_grp)
    parser_add_optgroup_from_def(parser, searchlight_constraints_opts_grp,
                                 exclusive=True)
    parser_add_optgroup_from_def(parser, crossvalidation_opts_grp,
                                 prefix='--cv-')
    parser_add_optgroup_from_def(parser, single_required_hdf5output)

def run(args):
    if os.path.isfile(args.payload) and args.payload.endswith('.py'):
        measure = script2obj(args.payload)
    elif args.payload == 'cv':
        if args.cv_learner is None or args.cv_partitioner is None:
            raise ValueError('cross-validation payload requires --learner and --partitioner')
        # get CV instance
        measure = get_crossvalidation_instance(
                    args.cv_learner, args.cv_partitioner, args.cv_errorfx,
                    args.cv_sampling_repetitions, args.cv_learner_space,
                    args.cv_balance_training, args.cv_permutations,
                    args.cv_avg_datafold_results, args.cv_prob_tail)
    else:
        raise RuntimeError("this should not happen")
    dss = hdf2ds(args.data)
    verbose(3, 'Loaded %i dataset(s)' % len(dss))
    ds = vstack(dss)
    verbose(3, 'Concatenation yielded %i samples with %i features' % ds.shape)
    # setup neighborhood
    # XXX add big switch to allow for setting up surface-based neighborhoods
    from mvpa2.misc.neighborhood import IndexQueryEngine
    qe = IndexQueryEngine(**dict(args.neighbors))
    # scatter_neighborhoods
    if not args.scatter_rois is None:
        from mvpa2.misc.neighborhood import scatter_neighborhoods
        attr, nb = args.scatter_rois
        coords = ds.fa[attr].value
        seed_coords, roi_ids = scatter_neighborhoods(nb, coords)
        verbose(3, 'Attempting %i ROI analyses' % len(roi_ids))
    elif not args.roi_attr is None:
        verbose(3, 'Attempting ROI analyses in %s' % args.roi_attr)
        roi_ids = args.roi_attr
    else:
        verbose(3, 'Attempting %i ROI analyses' % ds.nfeatures)
        roi_ids = None
    from mvpa2.measures.searchlight import Searchlight

    sl = Searchlight(measure,
                     queryengine=qe,
                     roi_ids=roi_ids,
                     nproc=args.nproc,
                     results_backend=args.multiproc_backend,
                     results_fx=args.aggregate_fx,
                     enable_ca=args.enable_ca,
                     disable_ca=args.disable_ca)
    # XXX support me too!
    #                 add_center_fa
    #                 results_fx
    #                 tmp_prefix
    #                 nblocks
    #                 null_dist
    # run 
    res = sl(ds)
    # XXX create more output
    # and store
    ds2hdf5(res, args.output, compression=args.hdf5_compression)
    return res
