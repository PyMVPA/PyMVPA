# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA atlases"""

import unittest

from mvpa.base import externals, warning
if externals.exists('nifti', raiseException=True):
    from mvpa.atlases import *
else:
    raise RuntimeError, "Don't run me if no nifti is present"

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

def suite():
    return unittest.makeSuite(AtlasesTests)


if __name__ == '__main__':
    import runner

