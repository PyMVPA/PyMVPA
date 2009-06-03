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
import numpy as N

from mvpa.base import externals, warning

if externals.exists('nifti', raiseException=True):
    from mvpa.atlases import *
else:
    raise RuntimeError, "Don't run me if no nifti is present"

import os
from mvpa import pymvpa_dataroot

class AtlasesTests(unittest.TestCase):
    """Basic tests for support of atlases such as the ones
    shipped with FSL
    """
    def testTransformations(self):
        """TODO"""
        pass

    def testAtlases(self):
        """Basic testing of atlases"""

        tested = 0
        for name in KNOWN_ATLASES.keys():
            #filename = KNOWN_ATLASES[name] % {'name': name}
            try:
                atlas = Atlas(name=name)
                tested += 1
            except IOError:
                # so we just don't have it
                continue
            #print isinstance(atlas.atlas, objectify.ObjectifiedElement)
            #print atlas.header.images.imagefile.get('offset')
            #print atlas.labelVoxel( (0, -7, 20) )
            #print atlas[ 0, 0, 0 ]
            coord = (-63, -12, 22)

            # Atlas must have at least 1 level and that one must
            # have some labels
            self.failUnless(len(atlas.levels_dict[0].labels) > 0)

            for res in [ atlas[coord],
                         atlas.labelPoint(coord) ]:
                self.failUnless(res.get('coord_queried', None) == coord,
                                '%s: Comparison failed. Got %s and %s'
                                % (name, res.get('coord_queried', None), coord))
                self.failUnless('labels' in res)
                # all atlases so far are based on voxels
                self.failUnless('voxel_queried' in res)

            # test explicit level specification via slice, although bogus here
            # XXX levels in queries should be deprecated -- too much of
            # performance hit
            res0 = atlas[coord, range(atlas.Nlevels)]
            self.failUnless(res0 == res)

            #print atlas[ 0, -7, 20, [1,2,3] ]
            #print atlas[ (0, -7, 20), 1:2 ]
            #print atlas[ (0, -7, 20) ]
            #print atlas[ (0, -7, 20), : ]
            #   print atlas.getLabels(0)
        if not tested:
            warning("No atlases were found -- thus no testing was done")

    def testFind(self):
        if not externals.exists('atlas_fsl'): return
        tshape = (182, 218, 182)        # target shape of fsl atlas chosen by default
        atl = Atlas(name='HarvardOxford-Cortical')
        atl.levels_dict[0].find('Frontal Pole')
        self.failUnlessEqual(
            len(atl.find(re.compile('Fusiform'), unique=False)),
            4)

        m = atl.getMap(1)
        self.failUnlessEqual(m.shape, tshape)
        self.failUnless(N.max(m)==100)
        self.failUnless(N.min(m)==0)

        ms = atl.getMaps('Fusiform')
        self.failUnlessEqual(len(ms), 4)
        self.failUnlessEqual(ms[0].shape, tshape)

        ms = atl.getMaps('ZaZaZa')
        self.failUnless(not len(ms))

        self.failUnlessRaises(ValueError, atl.getMap, 'Fusiform')
        self.failUnless(len(atl.find('Fusiform', unique=False)) == 4)
        self.failUnlessEqual(atl.getMap('Fusiform', strategy='max').shape,
                             tshape)

        # Test loading of custom atlas
        # for now just on the original file
        atl2 = Atlas(name='HarvardOxford-Cortical',
                     image_file=atl._image_file)

        # we should get exactly the same maps from both in this dummy case
        self.failUnless((atl.getMap('Frontal Pole') ==
                         atl2.getMap('Frontal Pole')).all())


        # Lets falsify and feed some crammy file as the atlas
        atl2 = Atlas(name='HarvardOxford-Cortical',
                     image_file=os.path.join(pymvpa_dataroot,
                                             'example4d.nii.gz'))

        # we should get not even comparable maps now ;)
        self.failUnless(atl.getMap('Frontal Pole').shape
                        != atl2.getMap('Frontal Pole').shape)


def suite():
    return unittest.makeSuite(AtlasesTests)


if __name__ == '__main__':
    import runner

