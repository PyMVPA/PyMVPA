# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
Functionality for surface-based voxel selection

Created on Feb 13, 2012

WiP.

References
----------
NN Oosterhof, T Wiestler, PE Downing, J Diedrichsen (2011). A comparison of volume-based
and surface-based multi-voxel pattern analysis. Neuroimage, 56(2), pp. 593-600

'Surfing' toolbox: http://surfing.sourceforge.net
(and the associated documentation)
'''

__docformat__ = 'restructuredtext'


import time
import collections
import operator
import datetime
import math
import os

import numpy as np

from mvpa2.base import warning, externals

from mvpa2.misc.surfing import volgeom, volsurf, volume_mask_dict
from mvpa2.support.nibabel import surf
from mvpa2.base.progress import ProgressBar, seconds2prettystring

if externals.exists('h5py'):
    from mvpa2.base.hdf5 import h5save, h5load


# TODO: see if we use these contants, or let it be up to the user
# possibly also rename them
LINEAR_VOXEL_INDICES = "linear_voxel_indices"
CENTER_DISTANCES = "center_distances"
GREY_MATTER_POSITION = "grey_matter_position"

from mvpa2.base import debug
if __debug__:
    if not "SVS" in debug.registered:
        debug.register("SVS", "Surface-based voxel selection "
                       " (a.k.a. 'surfing')")

class VoxelSelector(object):
    '''Voxel selection for surface-based searchlights'''

    def __init__(self, radius, distance_surf, n2v, distance_metric='dijkstra',
                            outside_node_margin=None):
        '''
        Voxel selection using cortical surfaces.

        Parameters
        ----------
        radius: int or float
            Searchlight radius. If the type is int, then this set the number of
            voxels in each searchlight (with variable size of the disc across
            searchlights). If the type is float, then this sets the disc radius in
            metric distance (with variable number of voxels across searchlights).
            In the latter case, the distance unit is usually in milimeters
            (which is the unit used for FreeSurfer surfaces).
            If radius is zero then only the center node itself is considered.
        distance_surf: surf.Surface
            A surface to be used for distance measurement. Usually this is the
            intermediate distance constructed by taking the node-wise average of
            the pial and white surface.
        n2v: dict
            Mapping from center nodes to surrounding voxels (and their distances).
            Usually this is the output from volsurf.node2voxels.
        distance_metric: str
            Distance measure used to define distances between nodes on the surface.
            Currently supports 'dijkstra' and 'euclidean'
        outside_node_margin: float or True or None (default)
            By default nodes outside the volume are skipped; using this
            parameter allows for a marign. If this value is a float (possibly
            np.inf), then all nodes within outside_node_margin Dijkstra
            distance from any node within the volume are still assigned
            associated voxels. If outside_node_margin is True, then a node is
            always assigned voxels regardless of its position in the volume.
        '''
        tp = type(radius)
        if tp is int: # fixed number of voxels
            self._fixedradius = False
            # use a (very arbitary) way to estimate how big the radius should
            # be initally to select the required number of voxels
            initradius_mm = .001 + 1.5 * float(radius) ** .5
        elif tp is float: # fixed metric radius
            self._fixedradius = True
            initradius_mm = radius
        else:
            raise TypeError("Illegal type for radius: expected int or float")
        self._targetradius = radius # radius to achieve (float or int)
        self._initradius_mm = initradius_mm # initial radius in mm
        self._optimizer = _RadiusOptimizer(initradius_mm)
        self._distance_metric = distance_metric # }
        self._surf = distance_surf                     # } save input
        self._n2v = n2v                       # }
        self._outside_node_margin = outside_node_margin

    def _select_approx(self, voxprops, count=None):
        '''
        Select approximately a certain number of voxels.

        Parameters
        ----------
        voxprops: dict
            Various voxel properties, where voxprops[key] is a list with
            voxprops[key][i] property key for voxel i.
            Should at least have a key 'distances'.
            Each of voxprops[key] should be list-like of equal length.
        count: int (default: None)
            How many voxels should be selected, approximately

        Returns
        -------
        v2d_sel: dict
            Similar mapping as the input, with approximately 'count' voxels
            with the smallest distance in 'voxprops'
            If 'count is None' then 'v2d_sel is None'
            If voxprops has fewer than 'count' elemens then 'v2d_sel' is None

        Notes
        -----
        Distances are only computed along the surface; the relative position of
        a voxel within the gray matter is ignored. Therefore, multiple voxels
        can have the same distance from a center node. See node2voxels.
        '''

        if count is None:
            return voxprops

        distkey = CENTER_DISTANCES
        if not distkey in voxprops:
            raise KeyError("No distance key %s in it - cannot select voxels" %
                           distkey)
        allds = voxprops[distkey]

        #assert sorted(allds)==allds #that's what voxprops should give us

        n = len(allds)

        if n < count or n == 0:
            return None

        # here, a 'chunk' is a set of voxels at the same distance. voxels are
        # selected in chunks with increasing distance. either all voxels in a
        # chunk are selected or none.
        curchunk = []
        prevd = allds[0]
        chunkcount = 1
        for i in xrange(n):
            d = allds[i] # distance
            if i > 0 and prevd != d:
                if i >= count: # we're done, use the chunk we have now
                    break
                curchunk = [i] # start a new chunk
                chunkcount += 1
            else:
                curchunk.append(i)

            prevd = d

        # see if the last chunk should be added or not to be as close as
        # possible to count
        firstpos = curchunk[0]
        lastpos = curchunk[-1]

        # difference in distance between desired count and positions
        delta = (count - firstpos) - (lastpos - count)
        if delta > 0:
            # lastpos is closer to count
            cutpos = lastpos + 1
        elif delta < 0:
            # firstpos is closer to count
            cutpos = firstpos
        else:
            # it's a tie, choose quasi-randomly based on chunkcount
            cutpos = firstpos if chunkcount % 2 == 0 else (lastpos + 1)

        for k in voxprops.keys():
            voxprops[k] = voxprops[k][:cutpos]

        return voxprops


    def disc_voxel_attributes(self, src):
        '''
        Voxel selection for single center node

        Parameters
        ----------
        src: int
            Index of center node to be used as searchlight center

        Returns
        -------
        voxprops: dict
            Various voxel properties, where voxprops[key] is a list with
            voxprops[key][i] property key for voxel i.
            Has at least a key 'distances'.
            Each of voxprops[key] should be list-like of equal length.
        '''
        optimizer = self._optimizer
        dist_surf = self._surf
        n2v = self._n2v
        outside_node_margin = self._outside_node_margin

        def node_in_vol(nd):
            return nd in n2v and not n2v[nd] is None

        if not node_in_vol(src) and not outside_node_margin is True:
            skip = True
            if not outside_node_margin is None:
                if math.isinf(outside_node_margin):
                    if __debug__:
                        debug("SVS", "")
                        debug("SVS", "node %s is outside - considering all other "
                                     "nodes that may be inside" % (src,))
                    for nd in n2v:
                        if node_in_vol(nd):
                            skip = False
                            break
                else:
                    node_distances = dist_surf.circlearound_n2d(src,
                                                radius=outside_node_margin,
                                                metric=self._distance_metric)

                    if __debug__:
                        debug("SVS", "")
                        debug("SVS", "node %s is outside - considering %d distances"
                                    " to other nodes that may be inside." % ((src,), len(node_distances)))
                    for nd, d in node_distances.iteritems():
                        if nd in n2v and not n2v[nd] is None and d <= outside_node_margin:
                            if __debug__:
                                debug("SVS", "node #%s is distance %s <= %s from #%d "
                                      " and kept" %
                                      ((src,), d, outside_node_margin, nd))
                            skip = False
                            break

            if skip:
                # no voxels associated with this node, skip
                if __debug__:
                    debug("SVS", "Skipping node %s (no voxels associated)" %
                                        (src,), cr=True)

                return []

        radius_mm = optimizer.get_start()
        radius = self._targetradius

        maxiter = 100
        for counter in xrange(maxiter):
            if radius_mm == 0:
                # only the node itself.
                # this should work except for very strange surfaces where
                # multiple nodes occupy exactly the same spatial location
                around_n2d = {src:0.}
            else:
                around_n2d = dist_surf.circlearound_n2d(src, radius_mm,
                                               self._distance_metric)

            allvxdist = self.nodes2voxel_attributes(around_n2d, n2v)

            if not allvxdist:
                voxel_attributes = []

            if self._fixedradius:
                # select all voxels
                voxel_attributes = self._select_approx(allvxdist, count=None)
            else:
                # select only certain number
                voxel_attributes = self._select_approx(allvxdist, count=radius)

            if voxel_attributes is None:
                # coult not find enough voxels, stay in loop and try again
                # with bigger radius
                radius_mm = optimizer.get_next()
            else:
                break

        if counter + 1 >= maxiter:
            raise ValueError("Failure to increase radius to get %d voxels for "
                             " node #%d" % (radius, src))

        if voxel_attributes and len(voxel_attributes[CENTER_DISTANCES]):
            # found at least one voxel; update our optimizer
            maxradius = voxel_attributes[CENTER_DISTANCES][-1]
            optimizer.set_final(maxradius)

        return voxel_attributes

    def disc_voxel_indices_and_attributes(self, src):
        ''' For now this is a wrapper
        TODO integrate with calling function'''
        attrs = self.disc_voxel_attributes(src)

        if not attrs:
            return None, None

        idxs = attrs.pop(LINEAR_VOXEL_INDICES)
        return idxs, attrs


    def nodes2voxel_attributes(self, n2d, n2v, distancesummary=min):
        '''
        Computes voxel distances

        Parameters
        ----------
        n2d: dict
            A mapping from node indices to distances (to a center node)
            Usually this is the output from surf.circlearound_n2d and thus
            only contains voldata for voxels surrounding a single center node
        n2v: dict
            A mapping from nodes to surrounding voxel indices and distances.
            n2v[i]=v2d is a dict mapping node i to a dict v2d, which in turn
            maps voxel indices to distances to the center node (i.e. v2d[j]=d
            means that the distance from voxel with linear index j to the
            node with index i is d
        distancesummary: function
            This is by default the min function. It is used to summarize
            cases where a single voxels has multiple distances (and nodes)
            associated with it. By default we take the minimum distance, and
            the node that gives rise to this distance, as a representative
            for the distance.

        Returns
        -------
        voxelprops: dict
            Mapping from keys to lists that contain voxel properties.
            Each list should have the same length
            It has at least a key sparse_volmasks._VOXIDXSLABEL which maps to
            the linear voxel indices. It may also have 'distances' (distance from
            center node along the cortex)  and 'gmpositions' (relative position in
            the gray matter)

        '''

        # mapping from voxel indices to all distances
        v2dps = collections.defaultdict(set)

        # get node indices and associated (distance, grey matter positions)
        for nd, d in n2d.iteritems():
            if nd in n2v:
                vps = n2v[nd] # all voxels associated with this node
                if not vps is None:
                    for vx, pos in vps.items():
                        v2dps[vx].add((d, pos)) # associate voxel with tuple of distance and relative position


        # converts a tuple (vx, set([(d0,p0),(d1,p1),...]) to a triple (vx,pM,dM)
        # where dM is the minimum across all d*
        def unpack_dp(vx, dp, distancesummary=distancesummary):
            d, p = distancesummary(dp) # implicit sort by first elemnts first, i.e. distance
            return vx, d, p


        # make triples of (voxel index, distance to center node, relative position in grey matter)
        vdp = [unpack_dp(vx, dp) for vx, dp in v2dps.iteritems()]

        # sort triples by distance to center node
        vdp.sort(key=operator.itemgetter(1))

        if not vdp:
            vdp_tup = ([], [], []) # empty
        else:
            vdp_tup = zip(*vdp) # unzip triples into three lists

        vdp_tps = (np.int32, np.float32, np.float32)
        vdp_labels = (LINEAR_VOXEL_INDICES, CENTER_DISTANCES, GREY_MATTER_POSITION)

        voxel_attributes = dict()
        for i in xrange(3):
            voxel_attributes[vdp_labels[i]] = np.asarray(vdp_tup[i], dtype=vdp_tps[i])

        return voxel_attributes

def voxel_selection(vol_surf_mapping, radius, source_surf=None, source_surf_nodes=None,
                    distance_metric='dijkstra',
                    eta_step=10, nproc=None,
                    outside_node_margin=None,
                    results_backend=None, tmp_prefix='tmpvoxsel'):

    """
    Voxel selection for multiple center nodes on the surface

    Parameters
    ----------
    vol_surf_mapping: volsurf.VolSurfMapping
        Contains gray and white matter surface, and volume geometry
    radius: int or float
        Size of searchlight. If an integer, then it indicates the number of
        voxels. If a float, then it indicates the radius of the disc
    source_surf: surf.Surface or None
        Surface used to compute distance between nodes. If omitted, it is
        the average of the gray and white surfaces.
    source_surf_nodes: list of int or numpy array or None
        Indices of nodes in source_surf that serve as searchlight center.
        By default every node serves as a searchlight center.
    distance_metric: str
        Distance metric between nodes. 'euclidean' or 'dijksta' (default)
    eta_step: int
        Report progress every eta_step (default: 10).
    nproc: int or None
        Number of parallel threads. None means as many threads as the
        system supports. The pprocess is required for parallel threads; if
        it cannot be used, then a single thread is used.
    outside_node_margin: float or True or None (default)
        By default nodes outside the volume are skipped; using this
        parameter allows for a marign. If this value is a float (possibly
        np.inf), then all nodes within outside_node_margin Dijkstra
        distance from any node within the volume are still assigned
        associated voxels. If outside_node_margin is True, then a node is
        always assigned voxels regardless of its position in the volume.
    results_backend : 'native' or 'hdf5' or None (default).
        Specifies the way results are provided back from a processing block
        in case of nproc > 1. 'native' is pickling/unpickling of results by
        pprocess, while 'hdf5' would use h5save/h5load functionality.
        'hdf5' might be more time and memory efficient in some cases.
        If None, then 'hdf5' if used if available, else 'native'.
    tmp_prefix : str, optional
        If specified -- serves as a prefix for temporary files storage
        if results_backend == 'hdf5'.  Thus can specify the directory to use
        (trailing file path separator is not added automagically).

    Returns
    -------
    sel: volume_mask_dict.VolumeMaskDictionary
        Voxel selection results, that associates, which each node, the indices
        of the surrounding voxels.
    """

    # construct the intermediate surface, which is used
    # to measure distances
    intermediate_surf = (vol_surf_mapping.pial_surface * .5) + \
                        (vol_surf_mapping.white_surface * .5)

    if source_surf is None:
        source_surf = intermediate_surf
    else:
        source_surf = surf.from_any(source_surf)

    if _debug():
        debug('SVS', "Generated high-res intermediate surface: "
              "%d nodes, %d faces" %
              (intermediate_surf.nvertices, intermediate_surf.nfaces))
        debug('SVS', "Mapping source to high-res surface:"
              " %d nodes, %d faces" %
              (source_surf.nvertices, source_surf.nfaces))


    if distance_metric[0].lower() == 'e' and outside_node_margin:
        # euclidean distance: identity mapping
        # this is *slow*
        n = source_surf.nvertices
        xyz = source_surf.vertices
        src2intermediate = dict((i, tuple(xyz[i])) for i in xrange(n))
    else:
        # find a mapping from nodes in source_surf to those in
        # intermediate surface
        src2intermediate = source_surf.map_to_high_resolution_surf(\
                                                        intermediate_surf)

    # if no sources are given, then visit all ndoes
    if source_surf_nodes is None:
        source_surf_nodes = np.arange(source_surf.nvertices)

    n = len(source_surf_nodes)

    if _debug():
        debug('SVS',
              "Performing surface-based voxel selection"
              " for %d centers" % n)

    # visit in random order, for for better ETA estimate
    visitorder = list(np.random.permutation(len(source_surf_nodes)))

    # construct mapping from nodes to enclosing voxels
    n2v = vol_surf_mapping.get_node2voxels_mapping()

    if __debug__:
        debug('SVS', "Generated mapping from nodes"
              " to intersecting voxels")

    # build voxel selector
    voxel_selector = VoxelSelector(radius, intermediate_surf, n2v,
                                   distance_metric,
                                   outside_node_margin=outside_node_margin)

    if _debug():
        debug('SVS', "Instantiated voxel selector (radius %r)" % radius)


    # structure to keep output data. Initialize with None, then
    # make a sparse_attributes instance when we know what the attributes are
    node2volume_attributes = None

    attribute_mapper = voxel_selector.disc_voxel_indices_and_attributes

    srcs_order = [source_surf_nodes[node] for node in visitorder]
    src_trg_nodes = [(src, src2intermediate[src]) for src in srcs_order]

    if nproc is not None and nproc > 1 and not externals.exists('pprocess'):
        raise RuntimeError("The 'pprocess' module is required for "
                           "multiprocess searchlights. Please either "
                           "install python-pprocess, or reduce `nproc` "
                           "to 1 (got nproc=%i) or set to default None"
                           % nproc)

    if nproc is None:
        if externals.exists('pprocess'):
            try:
                import pprocess
                nproc = pprocess.get_number_of_cores() or 1
                if _debug() :
                    debug("SVS", 'Using pprocess with %d cores' % nproc)
            except:
                if _debug():
                    debug("SVS", 'pprocess not available')

        if nproc is None:
            # importing pprocess failed - so use a single core
            nproc = 1
            debug("SVS", 'Using %d cores - pprocess not available' % nproc)

    # get the the voxel selection parameters
    parameter_dict = vol_surf_mapping.get_parameter_dict()
    parameter_dict.update(dict(radius=radius,
                               outside_node_margin=outside_node_margin,
                               distance_metric=distance_metric),
                               source_nvertices=source_surf.nvertices)


    init_output = lambda: volume_mask_dict.VolumeMaskDictionary(
                                    vol_surf_mapping.volgeom,
                                    intermediate_surf,
                                    meta=parameter_dict)

    if nproc > 1:
        if results_backend == 'hdf5':
            externals.exists('h5py', raise_=True)
        elif results_backend is None:
            if externals.exists('h5py') and externals.versions['hdf5'] >= '1.8.7':
                results_backend = 'hdf5'
            else:
                results_backend = 'native'
        if _debug():
            debug('SVS', "Using '%s' backend" % (results_backend,))

        if not results_backend in ('native', 'hdf5'):
            raise ValueError('Illegal results backend %r' % results_backend)

        import pprocess
        n_srcs = len(src_trg_nodes)
        blocks = np.array_split(np.arange(n_srcs), nproc)

        results = pprocess.Map(limit=nproc)
        reducer = results.manage(pprocess.MakeParallel(_reduce_mapper))

        if __debug__:
            debug('SVS', "Starting %d child processes", (len(blocks),))

        for i, block in enumerate(blocks):
            empty_dict = init_output()

            src_trg = []
            for idx in block:
                src_trg.append(src_trg_nodes[idx])

            if _debug():
                debug('SVS', "  starting block %d/%d: %d centers" %
                            (i + 1, nproc, len(src_trg)), cr=True)

            reducer(empty_dict, attribute_mapper, src_trg,
                    eta_step=eta_step, proc_id='%d' % (i + 1,),
                    results_backend=results_backend, tmp_prefix=tmp_prefix)
        if _debug():
            debug('SVS', '')
            debug('SVS', 'Started all %d child processes' % (len(blocks)))
            tstart = time.time()

        node2volume_attributes = None
        for i, result in enumerate(results):
            if result is None:
                continue

            if results_backend == 'hdf5':
                result_fn = result
                result = h5load(result_fn)
                os.remove(result_fn)

            if node2volume_attributes is None:
                # first time we have actual results.
                # Use as a starting point
                node2volume_attributes = result
                if _debug():
                    debug('SVS', '')
                    debug('SVS', "Merging results from %d child "
                                 "processes using '%s' backend" %
                                 (len(blocks), results_backend))
            else:
                # merge new with current data
                node2volume_attributes.merge(result)
            if _debug():
                debug('SVS', "  merged result block %d/%d" % (i + 1, nproc),
                                cr=True)

        if _debug():
            telapsed = time.time() - tstart
            debug('SVS', "")
            debug('SVS', 'Merged results from %d child processed - '
                         'took %s' %
                         (len(blocks), seconds2prettystring(telapsed)))

    else:
        empty_dict = init_output()
        node2volume_attributes = _reduce_mapper(empty_dict,
                                                attribute_mapper,
                                                src_trg_nodes,
                                                eta_step=eta_step)
        debug('SVS', "")

    if _debug():
        if node2volume_attributes is None:
            msgs = ["Voxel selection completed: none of %d nodes have "
                    "voxels associated" % len(visitorder)]
        else:
            nvox_selected = np.sum(node2volume_attributes.get_mask() != 0)
            vg = vol_surf_mapping.volgeom

            msgs = ["Voxel selection completed: %d / %d nodes have "
                    "voxels associated" %
                    (len(node2volume_attributes.keys()), len(visitorder)),
                    "Selected %d / %d  voxels (%.0f%%) in the mask at least once" %
                    (nvox_selected, vg.nvoxels_mask,
                     100. * nvox_selected / vg.nvoxels_mask)]

        for msg in msgs:
            debug("SVS", msg)


    if node2volume_attributes is None:
        warning('No voxels associated with any of %d nodes' %
                        len(visitorder))
    return node2volume_attributes

def _reduce_mapper(node2volume_attributes, attribute_mapper,
                   src_trg_indices, eta_step=1, proc_id=None,
                   results_backend='native', tmp_prefix='tmpvoxsel'):
    '''applies voxel selection to a list of src_trg_indices
    results are added to node2volume_attributes.
    '''

    if not src_trg_indices:
        return None

    if not results_backend in ('native', 'hdf5'):
        raise ValueError('Illegal results backend %r' % results_backend)


    def _pat(index, xs=src_trg_indices, f=max):
        try:
            if not xs:
                y = 1
            else:
                y = f(x[index] for x in xs)
            if y < 1:
                y = 1
            p = '%%%dd' % math.ceil(math.log10(y))
        except:
            p = '%s'
        return p

    progresspat = '(node %s -> %s)' % (_pat(0), _pat(1))

    # start the clock
    bar = ProgressBar()
    n = len(src_trg_indices)

    for i, (src, trg) in enumerate(src_trg_indices):
        idxs, misc_attrs = attribute_mapper(trg)

        if idxs is not None:
            node2volume_attributes.add(int(src), idxs, misc_attrs)

        if _debug() and eta_step and (i % eta_step == 0 or i == n - 1):
            msg = bar(float(i + 1) / n, progresspat % (src, trg))
            if not proc_id is None:
                msg += ' (#%s)' % proc_id
            debug('SVS', msg, cr=True)

    if results_backend == 'hdf5':
        tmp_postfix = ('__tmp__%d_%s.h5py' %
                                 (hash(time.time()), proc_id))
        tmp_fn = tmp_prefix + tmp_postfix
        h5save(tmp_fn, node2volume_attributes)
        return tmp_fn
    else:
        return node2volume_attributes

def _debug():
    return __debug__ and 'SVS' in debug.active



def run_voxel_selection(radius, volume, white_surf, pial_surf,
                         source_surf=None, source_surf_nodes=None,
                         volume_mask=None, distance_metric='dijkstra',
                         start_mm=0, stop_mm=0, start_fr=0., stop_fr=1.,
                         nsteps=10, eta_step=1, nproc=None,
                         outside_node_margin=None,
                         results_backend=None, tmp_prefix='tmpvoxsel',
                         node_voxel_mapping='maximal'):

    """
    Voxel selection wrapper for multiple center nodes on the surface

    Parameters
    ----------
    radius: int or float
        Size of searchlight. If an integer, then it indicates the number of
        voxels. If a float, then it indicates the radius of the disc
    volume: Dataset or NiftiImage or volgeom.Volgeom
        Volume in which voxels are selected.
    white_surf: str of surf.Surface
        Surface of white-matter to grey-matter boundary, or filename
        of file containing such a surface.
    pial_surf: str of surf.Surface
        Surface of grey-matter to pial-matter boundary, or filename
        of file containing such a surface.
    source_surf: surf.Surface or None
        Surface used to compute distance between nodes. If omitted, it is
        the average of the gray and white surfaces.
    source_surf_nodes: list of int or numpy array or None
        Indices of nodes in source_surf that serve as searchlight center.
        By default every node serves as a searchlight center.
    volume_mask: None (default) or False or int
        Mask from volume to apply from voxel selection results. By default
        no mask is applied. If volume_mask is an integer k, then the k-th
        volume from volume is used to mask the data. If volume is a Dataset
        and has a property volume.fa.voxel_indices, then these indices
        are used to mask the data, unless volume_mask is False or an integer.
    distance_metric: str
        Distance metric between nodes. 'euclidean' or 'dijksta' (default)
    start_fr: float (default: 0)
            Relative start position of line in gray matter, 0.=white
            surface, 1.=pial surface
    stop_fr: float (default: 1)
        Relative stop position of line (as in see start)
    start_mm: float (default: 0)
        Absolute start position offset (as in start_fr)
    stop_mm: float (default: 0)
        Absolute start position offset (as in start_fr)
    nsteps: int (default: 10)
        Number of steps from white to pial surface
    eta_step: int (default: 1)
        After how many searchlights an estimate should be printed of the
        remaining time until completion of all searchlights
    nproc: int or None
        Number of parallel threads. None means as many threads as the
        system supports. The pprocess is required for parallel threads; if
        it cannot be used, then a single thread is used.
    outside_node_margin: float or None (default)
        By default nodes outside the volume are skipped; using this
        parameter allows for a marign. If this value is a float (possibly
        np.inf), then all nodes within outside_node_margin Dijkstra
        distance from any node within the volume are still assigned
        associated voxels. If outside_node_margin is True, then a node is
        always assigned voxels regardless of its position in the volume.
    results_backend : 'native' or 'hdf5' or None (default).
        Specifies the way results are provided back from a processing block
        in case of nproc > 1. 'native' is pickling/unpickling of results by
        pprocess, while 'hdf5' would use h5save/h5load functionality.
        'hdf5' might be more time and memory efficient in some cases.
        If None, then 'hdf5' if used if available, else 'native'.
    tmp_prefix : str, optional
        If specified -- serves as a prefix for temporary files storage
        if results_backend == 'hdf5'.  Thus can specify the directory to use
        (trailing file path separator is not added automagically).
    node_voxel_mapping: 'minimal' or 'maximal' or 'minimal_lowres'
        If 'minimal' then each voxel is associated with at most one node.
        If 'maximal' it is associated with as many nodes that contain the
        voxel (default: 'maximal').
        If 'minimal_lowres' then each voxel is associated with at most one
        node, and each node that is mapped onto has a corresponding node
        (at the same spatial location) in source_surf.


    Returns
    -------
    sel: volume_mask_dict.VolumeMaskDictionary
        Voxel selection results, that associates, which each node, the indices
        of the surrounding voxels.
    """

    vg = volgeom.from_any(volume, volume_mask)

    mapper_dict = dict(maximal=volsurf.VolSurfMaximalMapping,
                       minimal=volsurf.VolSurfMinimalMapping,
                       minimal_lowres=volsurf.VolSurfMinimalLowresMapping)

    mapper = mapper_dict[node_voxel_mapping]

    vsm = mapper(vg, white=white_surf, pial=pial_surf,
                 intermediate=source_surf, nsteps=nsteps, start_fr=start_fr,
                 stop_fr=stop_fr, start_mm=start_mm, stop_mm=stop_mm)

    sel = voxel_selection(vol_surf_mapping=vsm, radius=radius,
                          source_surf=source_surf,
                          source_surf_nodes=source_surf_nodes,
                          distance_metric=distance_metric,
                          eta_step=eta_step, nproc=nproc,
                          outside_node_margin=outside_node_margin,
                          results_backend=results_backend,
                          tmp_prefix=tmp_prefix)

    return sel

class _RadiusOptimizer():
    '''
    Internal class to optimize the initial radius used for voxel selection.

    In the case of selecting a fixed number of voxels in each searchlight, the
    required radius will vary across searchlights. The general strategy is to take
    some initial radius, find the nodes that are within that radius, select the
    corresponding voxels, and see if enough voxels are selected. If not, the radius is
    increased and these steps repeated.

    A larger initial radius means a decrease in the probability that not enough voxels are
    selected, but an increase in time to compute distances and select voxels.

    The challenge therefore to find the optimal initial radius so that overall computational
    time is minimized.

    The present implementation is very stupid and just increases the radius every time
    by a factor of 1.5.

    NNO: as of August 2012 it seems that voxel selection is actually quite fast,
    so maybe this function is good as is
    '''
    def __init__(self, initradius):
        '''new instance, with certain initial radius'''
        self._initradius = initradius
        self._initmult = 1.5

    def get_start(self):
        '''get an (initial) radius for a new searchlight.'''
        self._curradius = self._initradius
        self._count = 0
        return self._curradius

    def get_next(self):
        '''get a new (better=larger) radius for the current searchlight'''
        self._count += 1
        self._curradius *= self._initmult
        return self._curradius

    def set_final(self, finalradius):
        '''to tell what the final radius was that satisfied the number of required voxels'''
        pass

    def __repr__(self):
        return 'radius is %f, %d steps' % (self._curradius, self._count)

def from_any(s):
    '''
    Loads or returns voxel selection results

    Parameters
    ----------
    s: basestring or volume_mask_dict.VolumeMaskDictionary
        if a string it is assumed to be a file name and loaded using h5load. If
        a volume_mask_dict.VolumeMaskDictionary then it is returned.

    Returns
    -------
    r: volume_mask_dict.VolumeMaskDictionary
    '''

    # this is just a convenience function
    return volume_mask_dict.from_any(s)

