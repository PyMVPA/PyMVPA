# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Collection of the known atlases"""

import os

from mvpa2.base import warning
from mvpa2.atlases.base import *
from mvpa2.atlases.fsl import *

__all__ = [ "KNOWN_ATLAS_FAMILIES", "KNOWN_ATLASES", "Atlas"]

KNOWN_ATLAS_FAMILIES = {
    'pymvpa': (["talairach", "talairach-dist"],
               r"/usr/share/rumba/atlases/data/%(name)s_atlas.xml"),
    'fsl': (["HarvardOxford-Cortical", "HarvardOxford-Subcortical",
             "JHU-tracts", "Juelich", "MNI", "Thalamus"],
            r"/usr/share/fsl/data/atlases/%(name)s.xml")
    # XXX make use of FSLDIR
    }

# map to go from the name to the path
KNOWN_ATLASES = dict(reduce(lambda x,y:x+[(yy,y[1]) for yy in y[0]],
                             KNOWN_ATLAS_FAMILIES.values(), []))


def Atlas(filename=None, name=None, *args, **kwargs):
    """A convinience factory for the atlases
    """
    if filename is None:
        if name is None:
            raise ValueError, \
                  "Please provide either path or name of the atlas to be used"
        atlaspath = KNOWN_ATLASES[name]
        filename = atlaspath % ( {'name': name} )
        if not os.path.exists(filename):
            raise IOError, \
                  "File %s for atlas %s was not found" % (filename, name)
    else:
        if name is not None:
            raise ValueError, "Provide only filename or name"

    try:
        # Just to guestimate what atlas that is
        tempAtlas = XMLBasedAtlas(filename=filename, load_maps=False) #, *args, **kwargs)
        version = tempAtlas.version
        atlas_source = None
        for cls in [PyMVPAAtlas, FSLAtlas]:
            if cls._check_version(version):
                atlas_source = cls.source
                break
        if atlas_source is None:
            if __debug__: debug('ATL_', "Unknown atlas " + filename)
            return tempAtlas

        atlasTypes = {
            'PyMVPA': {"Label" : LabelsAtlas,
                       "Reference": ReferencesAtlas},
            'FSL': {"Label" : FSLLabelsAtlas,
                    "Probabalistic": FSLProbabilisticAtlas,
                    "Probabilistic": FSLProbabilisticAtlas,
                    }
            }[atlas_source]
        atlasType = tempAtlas.header.type.text
        if atlasTypes.has_key(atlasType):
            if __debug__: debug('ATL_', "Creating %s Atlas" % atlasType)
            return atlasTypes[atlasType](filename=filename, *args, **kwargs)
            #return ReferencesAtlas(filename)
        else:
            warning("Unknown %s type '%s' of atlas in %s." " Known are %s" %
                    (atlas_source, atlasType, filename,
                     atlasTypes.keys()), 2)
            return tempAtlas
    except XMLAtlasException, e:
        print "File %s is not a valid XML based atlas due to %s" \
              % (filename, `e`)
        raise e


if __name__ == '__main__':
    from mvpa2.base import verbose
    verbose.level = 10
    for name in [
        #'data/talairach_atlas.xml',
        '/usr/share/fsl/data/atlases/HarvardOxford-Cortical.xml',
        '/usr/share/fsl/data/atlases/HarvardOxford-Subcortical.xml'
        ]:
        atlas = Atlas(name)
        #print isinstance(atlas.atlas, objectify.ObjectifiedElement)
        #print atlas.header.images.imagefile.get('offset')
        #print atlas.label_voxel( (0, -7, 20) )
        #print atlas[ 0, 0, 0 ]
        print atlas[ -63, -12, 22 ]
        #print atlas[ 0, -7, 20, [1,2,3] ]
        #print atlas[ (0, -7, 20), 1:2 ]
        #print atlas[ (0, -7, 20) ]
        #print atlas[ (0, -7, 20), : ]
        #   print atlas.get_labels(0)
