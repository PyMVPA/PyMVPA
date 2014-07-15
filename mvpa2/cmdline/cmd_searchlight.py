# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Traveling ROI analysis

"""

# magic line for manpage summary
# man: -*- % traveling ROI analysis

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
    import parser_add_common_opt, ds2hdf5, hdf2ds, \
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
        prefixed with the attribute name, i.e. 'altcoords:2'. For ROI shapes
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
    (('--roi-attr',), dict(metavar='ATTR/EXPR', nargs='+',
        help="""name of a feature attribute whose non-zero values define
        possible ROI seeds/centers. Alternatively, this can also be an
        expression like: parcellation_roi eq 16 (see the 'select' command
        on information what expressions are supported).""")),
])


# XXX this should eventually move into the main code base, once
# sufficiently generalized
def _fill_in_scattered_results(sl, dataset, roi_ids, results):
    """this requires the searchlight conditional attribute 'roi_feature_ids'
    to be enabled"""
    import numpy as np
    from mvpa2.datasets import Dataset

    resmap = None
    probmap = None
    for resblock in results:
        for res in resblock:
            if resmap is None:
                # prepare the result container
                resmap = np.zeros((len(res), dataset.nfeatures),
                                  dtype=res.samples.dtype)
                if 'null_prob' in res.fa:
                    # initialize the prob map also with zeroes, as p=0 can never
                    # happen as an empirical result
                    probmap = np.zeros((dataset.nfeatures,) + res.fa.null_prob.shape[1:],
                                      dtype=res.samples.dtype)
                observ_counter = np.zeros(dataset.nfeatures, dtype=int)
            #project the result onto all features -- love broadcasting!
            resmap[:, res.a.roi_feature_ids] += res.samples
            if not probmap is None:
                probmap[res.a.roi_feature_ids] += res.fa.null_prob
            # increment observation counter for all relevant features
            observ_counter[res.a.roi_feature_ids] += 1
    # when all results have been added up average them according to the number
    # of observations
    observ_mask = observ_counter > 0
    resmap[:, observ_mask] /= observ_counter[observ_mask]
    result_ds = Dataset(resmap,
                        fa={'observations': observ_counter})
    if not probmap is None:
        # transpose to make broadcasting work -- creates a view, so in-place
        # modification still does the job
        probmap.T[:,observ_mask] /= observ_counter[observ_mask]
        result_ds.fa['null_prob'] = probmap.squeeze()
    if 'mapper' in dataset.a:
        import copy
        result_ds.a['mapper'] = copy.copy(dataset.a.mapper)
    return result_ds


def setup_parser(parser):
    from .helpers import parser_add_optgroup_from_def, \
        parser_add_common_attr_opts, single_required_hdf5output, ca_opts_grp
    parser_add_common_opt(parser, 'multidata', required=True)
    parser_add_optgroup_from_def(parser, searchlight_opts_grp)
    parser_add_optgroup_from_def(parser, ca_opts_grp)
    parser_add_optgroup_from_def(parser, searchlight_constraints_opts_grp)
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
    # determine ROIs
    rids = None     # all by default
    aggregate_fx = args.aggregate_fx
    if args.roi_attr is not None:
        # first figure out which roi features should be processed
        if len(args.roi_attr) == 1 and args.roi_attr[0] in ds.fa.keys():
            # name of an attribute -> pull non-zeroes
            rids = ds.fa[args.roi_attr[0]].value.nonzero()[0]
        else:
            # an expression?
            from .cmd_select import _eval_attr_expr
            rids = _eval_attr_expr(args.roi_attr, ds.fa).nonzero()[0]

    seed_ids = None
    if args.scatter_rois is not None:
        # scatter_neighborhoods among available ids if was requested
        from mvpa2.misc.neighborhood import scatter_neighborhoods
        attr, nb = args.scatter_rois
        coords = ds.fa[attr].value
        if rids is not None:
            # select only those which were chosen by ROI
            coords = coords[rids]
        _, seed_ids = scatter_neighborhoods(nb, coords)
        if aggregate_fx is None:
            # no custom one given -> use default "fill in" function
            aggregate_fx = _fill_in_scattered_results
            if args.enable_ca is None:
                args.enable_ca = ['roi_feature_ids']
            elif 'roi_feature_ids' not in args.enable_ca:
                args.enable_ca += ['roi_feature_ids']

    if seed_ids is None:
        roi_ids = rids
    else:
        if rids is not None:
            # we had to sub-select by scatterring among available rids
            # so we would need to get original ids
            roi_ids = rids[seed_ids]
        else:
            # scattering happened on entire feature-set
            roi_ids = seed_ids

    verbose(3, 'Attempting %i ROI analyses'
               % ((roi_ids is None) and ds.nfeatures or len(roi_ids)))

    from mvpa2.measures.searchlight import Searchlight

    sl = Searchlight(measure,
                     queryengine=qe,
                     roi_ids=roi_ids,
                     nproc=args.nproc,
                     results_backend=args.multiproc_backend,
                     results_fx=aggregate_fx,
                     enable_ca=args.enable_ca,
                     disable_ca=args.disable_ca)
    # XXX support me too!
    #                 add_center_fa
    #                 tmp_prefix
    #                 nblocks
    #                 null_dist
    # run
    res = sl(ds)
    if (seed_ids is not None) and ('mapper' in res.a):
        # strip the last mapper link in the chain, which would be the seed ID selection
        res.a['mapper'] = res.a.mapper[:-1]
    # XXX create more output
    # and store
    ds2hdf5(res, args.output, compression=args.hdf5_compression)
    return res
