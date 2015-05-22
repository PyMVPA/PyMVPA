# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA atlases"""

import unittest, re
import numpy as np

from mvpa2.testing import *

skip_if_no_external('nibabel')
skip_if_no_external('lxml')

from mvpa2.base import externals
from mvpa2.atlases import *

from mvpa2 import pymvpa_dataroot

"""Basic tests for support of atlases such as the ones
shipped with FSL
"""

def test_transformations():
    """TODO"""
    raise SkipTest, "Please test application of transformations"

@sweepargs(name=KNOWN_ATLASES.keys())
def test_atlases(name):
    """Basic testing of atlases"""

    #filename = KNOWN_ATLASES[name] % {'name': name}
    try:
        atlas = Atlas(name=name)
    except IOError, e:
        # so we just don't have it
        raise SkipTest('Skipped atlas %s due to %s' % (name, e))
    #print isinstance(atlas.atlas, objectify.ObjectifiedElement)
    #print atlas.header.images.imagefile.get('offset')
    #print atlas.label_voxel( (0, -7, 20) )
    #print atlas[ 0, 0, 0 ]
    coord = (-63, -12, 22)

    # Atlas must have at least 1 level and that one must
    # have some labels
    ok_(len(atlas.levels[0].labels) > 0)

    for res in [ atlas(coord),
                 atlas.label_point(coord) ]:
        ok_(res.get('coord_queried', None) == coord,
                        '%s: Comparison failed. Got %s and %s'
                        % (name, res.get('coord_queried', None), coord))
        ok_('labels' in res)
        # all atlases so far are based on voxels
        ok_('voxel_queried' in res)

    # test explicit level specification via slice, although bogus here
    # XXX levels in queries should be deprecated -- too much of
    # performance hit
    res0 = atlas(coord, range(atlas.nlevels))
    ok_(res0 == res)

    #print atlas[ 0, -7, 20, [1,2,3] ]
    #print atlas[ (0, -7, 20), 1:2 ]
    #print atlas[ (0, -7, 20) ]
    #print atlas[ (0, -7, 20), : ]
    #   print atlas.get_labels(0)


def test_fsl_hox_queries():
    skip_if_no_external('atlas_fsl')

    tshape = (182, 218, 182)        # target shape of fsl atlas chosen by default
    atl = Atlas(name='HarvardOxford-Cortical')
    atl.levels[0].find('Frontal Pole')
    assert_equal(len(atl.find(re.compile('Fusiform'), unique=False)),
                 4)

    m = atl.get_map(1)
    assert_equal(m.shape, tshape)
    ok_(np.max(m)==100)
    ok_(np.min(m)==0)

    ms = atl.get_maps('Fusiform')
    assert_equal(len(ms), 4)
    for l, m in ms.iteritems():
        assert_equal(m.shape, tshape)

    ms = atl.get_maps('ZaZaZa')
    ok_(not len(ms))

    assert_raises(ValueError, atl.get_map, 'Fusiform')
    ok_(len(atl.find('Fusiform', unique=False)) == 4)
    ff_map = atl.get_map('Fusiform', strategy='max')
    assert_equal(ff_map.shape, tshape)

    # atlas has very unfortunate shape -- the same under .T ... heh heh
    # Lets validate either map is in correct orientation
    ok_(ff_map[119, 91, 52] > 60)
    ok_(ff_map[52, 91, 119] == 0)

    # Lets validate some coordinates queries
    r_gi = atl(-48, -75, 19)
    r_point = atl.label_point((-48, -75, 19))
    r_voxel = atl.label_voxel((138, 51, 91))

    # by default, __getitem__ queries coordinates in voxels
    ok_(r_voxel == atl[(138, 51, 91)] == atl[138, 51, 91])
    # by default -- opens at highest-available resolution,
    # i.e. 1mm since a while
    ok_(atl.resolution == 1.)

    # by default, __call__ queries coordinates in space
    ok_(r_point == atl(-48, -75, 19) == atl((-48, -75, 19)))

    ok_(r_point['labels'] == r_voxel['labels'] ==
         [[{'index': 21, 'prob': 64,
            'label': 'Lateral Occipital Cortex, superior division'},
           {'index': 22, 'prob': 22,
            'label': 'Lateral Occipital Cortex, inferior division'}]])
    ok_(r_point['voxel_atlas'] == r_point['voxel_queried'] ==
        list(r_voxel['voxel_queried']) == [138, 51, 91])
    # TODO: unify list/tuple in above -- r_point has lists

    # Test loading of custom atlas
    # for now just on the original file
    atl2 = Atlas(name='HarvardOxford-Cortical',
                 image_file=atl._image_file)

    # we should get exactly the same maps from both in this dummy case
    ok_((atl.get_map('Frontal Pole') == atl2.get_map('Frontal Pole')).all())


    # Lets falsify and feed some crammy file as the atlas
    atl2 = Atlas(name='HarvardOxford-Cortical',
                 image_file=pathjoin(pymvpa_dataroot, 'example4d.nii.gz'))

    # we should get not even comparable maps now ;)
    ok_(atl.get_map('Frontal Pole').shape != atl2.get_map('Frontal Pole').shape)

    # Lets check unique resolution for the atlas
    maps = atl.get_maps('Fusiform')
    maps_max = atl.get_maps('Fusiform', overlaps='max')

    mk = maps.keys()
    ok_(set(mk) == set(maps_max.keys()))

    maps_ab = np.array([maps[k]!=0 for k in mk])
    maps_max_ab = np.array([maps_max[k]!=0 for k in mk])

    # We should have difference
    ok_(np.any(maps_ab != maps_max_ab))
    maps_max_ab_sum = np.sum(maps_max_ab, axis=0)
    ok_(np.all(0<=maps_max_ab_sum))
    ok_(np.all(maps_max_ab_sum<=1))
    ok_(np.any(np.sum(maps_ab, axis=0)>1))

    # we should still cover the same set of voxels
    assert_array_equal(np.max(maps_ab, axis=0), np.max(maps_max_ab, axis=0))

# Basic testing of Talairach atlases in its original space
def test_pymvpa_talairach():
    skip_if_no_external('atlas_pymvpa')

    atl = Atlas(name='talairach')
    atld = Atlas(name='talairach-dist',
                 reference_level='Closest Gray',
                 distance=10)

    p = [-22, -40, 8]
    pl  = atl.label_point(p)
    pld = atld.label_point(p)

    ok_(np.any(pl['voxel_queried'] != pld['voxel_queried']))
    ok_(pld['distance'] >0)
    # Common labels
    for l in pl, pld:
        assert_equal(l['labels'][0]['label'].text, 'Left Cerebrum')
        assert_equal(l['labels'][1]['label'].text, 'Sub-lobar')

    # different ones
    assert_equal(pl['labels'][3]['label'].text, 'Cerebro-Spinal Fluid')
    assert_equal(pld['labels'][3]['label'].text, 'Gray Matter')

    assert_equal(pl['labels'][4]['label'].text, 'None')
    assert_equal(pld['labels'][4]['label'].text, 'Caudate Tail')
