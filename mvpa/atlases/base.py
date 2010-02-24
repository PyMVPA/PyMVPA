# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Base classes for Anatomy atlases support

TODOs:
======

 * major optimization. Now code is sloppy and slow -- plenty of checks etc

Module Organization
===================
mvpa.atlases.base module contains support for various atlases

:group Base: BaseAtlas XMLBasedAtlas Label Level LabelsLevel
:group Talairach: PyMVPAAtlas LabelsAtlas ReferencesAtlas
:group Exceptions: XMLAtlasException

"""

from mvpa.base import externals

if externals.exists('lxml', raiseException=True):
    from lxml import etree, objectify

from mvpa.base.dochelpers import enhanced_doc_string

import os, re
import numpy as np
from numpy.linalg import norm

from mvpa.atlases.transformation import SpaceTransformation, Linear
from mvpa.misc.support import reuse_absolute_path

if externals.exists('nifti', raiseException=True):
    from nifti import NiftiImage

from mvpa.base import warning
if __debug__:
    from mvpa.base import debug


##REF: Name was automagically refactored
def check_range(coord, range):
    """
    Check if coordinates are within range (0,0,0) - (range)
    Return True on success
    """
    # TODO: optimize
    if len(coord) != len(range):
        raise ValueError("Provided coordinate %r and given range %r" % \
                         (coord, range) + \
                         " have different dimensionality"
                         )
    for c, r in zip(coord, range):
        if c < 0 or c >= r:
            return False
    return True


class BaseAtlas(object):
    """Base class for the atlases.
    """

    def __init__ (self):
        """
        Create an atlas object based on the... XXX
        """
        self.__name = "blank"           # XXX use or remove


class XMLAtlasException(Exception):
    """Exception to be thrown if smth goes wrong dealing with XML based atlas
    """
    pass

class XMLBasedAtlas(BaseAtlas):

    def __init__(self, filename=None,
                 resolution=None, image_file=None,
                 query_voxel=False,
                 coordT=None, levels=None):
        """
        Parameters
        ----------
        filename : str
          Filename for the xml definition of the atlas
        resolution : None or float
          Some atlases link to multiple images at different
          resolutions. if None -- best resolution is selected
          using 0th dimension resolution
        image_file : None or str
          If None, overrides filename for the used imagefile, so
          it could load a custom (re-registered) atlas maps
        query_voxel : bool, optional
          By default [x,y,z] assumes coordinates in space, but if
          query_voxel is True, they are assumed to be voxel coordinates
        coordT
          Optional transformation to apply first
        levels : None or slice or list of int
          What levels by default to operate on
        """
        BaseAtlas.__init__(self)
        self.__version = None
        self.__type = None              # XXX use or remove
        self._image_file = None
        self.__atlas = None
        self._filename = filename
        self._resolution = resolution
        self._force_image_file = image_file
        self.query_voxel = query_voxel
        self.levels = levels

        if filename:
            self.load_atlas(filename)

        # common sanity checks
        if not self._check_version(self.version):
            raise IOError("Version %s is not recognized to be native to class %s" % \
                          (self.version, self.__name__))

        if not set(['header', 'data']) == set([i.tag for i in self.getchildren()]):
            raise IOError("No header or data were defined in %s" % filename)

        header = self.header
        headerChildrenTags = XMLBasedAtlas._children_tags(header)
        if not ('images' in headerChildrenTags) or \
           not ('imagefile' in XMLBasedAtlas._children_tags(header.images)):
            raise XMLAtlasException("Atlas requires image/imagefile header fields")

        # Load and post-process images
        self._image = None
        self._load_images()
        if self._image is not None:
            self._extent = np.abs(np.asanyarray(self._image.extent[0:3]))
            self._voxdim = np.asanyarray(self._image.voxdim)
            self.relativeToOrigin = True
        # Assign transformation to get into voxel coordinates,
        # spaceT will be set accordingly
        self.set_coord_t(coordT)
        self._load_data()


    ##REF: Name was automagically refactored
    def _check_range(self, c):
        """ check and adjust the voxel coordinates"""
        # check range
        # list(c) for consistent appearance... some times c might be ndarray
        if __debug__: debug('ATL__', "Querying for voxel %s" % `list(c)`)
        if not check_range(c, self.extent):
            msg = "Coordinates %s are not within the extent %s." \
                  "Reset to (0,0,0)" % ( `c`, `self.extent` )
            if __debug__: debug('ATL_', msg)
            # assume that voxel [0,0,0] is blank
            c = [0]*3;
        return c


    @staticmethod
    ##REF: Name was automagically refactored
    def _check_version(version):
        """To be overriden in the derived classes. By default anything is good"""
        return True


    ##REF: Name was automagically refactored
    def _load_images(self):
        """To be overriden in the derived classes. By default does nothing"""
        pass


    ##REF: Name was automagically refactored
    def _load_data(self):
        """To be overriden in the derived classes. By default does nothing"""
        pass


    ##REF: Name was automagically refactored
    def load_atlas(self, filename):
        if __debug__: debug('ATL_', "Loading atlas definition xml file " + filename)
        # Create objectify parser first
        parser = etree.XMLParser(remove_blank_text=True)
        lookup = objectify.ObjectifyElementClassLookup()
        parser.setElementClassLookup(lookup)
        try:
            self.__atlas = etree.parse(filename, parser).getroot()
        except IOError:
            raise XMLAtlasException("Failed to load XML file %s" % filename)

    @property
    def version(self):
        if not self.__atlas is None \
               and ("version" in self.__atlas.attrib.keys()):
            return self.__atlas.get("version")
        else:
            return None

    @staticmethod
    def _compare_lists(checkitems, neededitems):
        raise RuntimeError, "DEPRECATED _compare_lists"
        checkitems.sort()
        neededitems.sort()
        return (checkitems == neededitems)


    @staticmethod
    def _children_tags(root):
        return [i.tag for i in root.getchildren()]


    def __getattr__(self, attr):
        """
        Lazy way to provide access to the definitions in the atlas
        """
        if not self.__atlas is None:
            return getattr(self.__atlas, attr)
        else:
            raise XMLAtlasException("Atlas in " + self.__name__ + " was not read yet")


    ##REF: Name was automagically refactored
    def set_coord_t(self, coordT):
        """Set coordT transformation.

        spaceT needs to be adjusted since we glob those two
        transformations together
        """
        self._coordT = coordT           # lets store for debugging etc
        if self._image is not None:
            # Combine with the image's qform
            coordT = Linear(np.linalg.inv(self._image.qform),
                            previous=coordT)
        self._spaceT = SpaceTransformation(
            previous=coordT, to_real_space=False
            )


    ##REF: Name was automagically refactored
    def label_point(self, coord, levels=None):
        """Return labels for the given spatial point at specified levels

        Function takes care about first transforming the point into
        the voxel space

        Parameters
        ----------
        coord : tuple
          Coordinates of the point (xyz)
        levels : None or list of int
          At what levels to return the results
        """
        coord_ = np.asarray(coord)          # or we would alter what should be constant
        #if not isinstance(coord, np.numpy):
        #c = self.getVolumeCoordinate(coord)
        #c = self.spaceT.to_voxel_space(coord_)
        #if self.coordT:
        #   coord_t = self.coordT[coord_]
        #else:
        #   coord_t = coord_

        c = self.spaceT(coord_)

        result = self.label_voxel(c, levels)
        result['coord_queried'] = coord
        #result['coord_trans'] = coord_t
        result['voxel_atlas'] = c
        return result


    ##REF: Name was automagically refactored
    def levels_listing(self):
        lkeys = range(self.n_levels)
        return '\n'.join(['%d: ' % k + str(self._levels_dict[k])
                          for k in lkeys])


    ##REF: Name was automagically refactored
    def _get_levels(self, levels=None):
        """Helper to provide list of levels to operate on

        Depends on given `levels` as well as self.levels
        """
        if levels is None:
            levels = [ i for i in xrange(self.n_levels) ]
        elif (isinstance(levels, slice)):
            # levels are given as a range
            if levels.step: step = levels.step
            else: step = 1

            if levels.start: start = levels.start
            else: start = 0

            if levels.stop: stop = levels.stop
            else: stop = self.n_levels

            levels = [ i for i in xrange(start, stop, step) ]

        elif isinstance(levels, list) or isinstance(levels, tuple):
            # levels given as list
            levels = list(levels)

        elif isinstance(levels, int):
            levels = [ levels ]

        else:
            raise TypeError('Given levels "%s" are of unsupported type' % `levels`)

        # test given values
        levels_dict = self.levels_dict
        for level in levels:
            if not level in levels_dict:
                raise ValueError, \
                      "Levels %s is not known (out of range?). Known levels are:\n%s" \
                      % (level, self.levels_listing())

        return levels


    def __getitem__(self, index):
        """
        Accessing the elements via simple indexing. Examples:
        print atlas[ 0, -7, 20, [1,2,3] ]
        print atlas[ (0, -7, 20), 1:2 ]
        print atlas[ (0, -7, 20) ]
        print atlas[ (0, -7, 20), : ]
        """
        if len(index) in [2, 4]:
            levels_slice = index[-1]
        else:
            if self.levels is None:
                levels_slice = slice(None,None,None)
            else:
                levels_slice = self.levels

        levels = self._get_levels(levels=levels_slice)

        if len(index) in [3, 4]:
            # we got coordinates 1 by 1 + may be a level
            coord = index[0:3]

        elif len(index) in [1, 2]:
            coord = index[0]
            if isinstance(coord, list) or isinstance(coord, tuple):
                if len(coord) != 3:
                    raise TypeError("Given coordinates must be in 3D")
            else:
                raise TypeError("Given coordinates must be a list or a tuple")

        else:
            raise TypeError("Unknown shape of parameters `%s`" % `index`)

        if self.query_voxel:
            return self.label_voxel(coord, levels)
        else:
            return self.label_point(coord, levels)


    # REDO in some sane fashion so referenceatlas returns levels for the base
    ##REF: Name was automagically refactored
    def _get_levels_dict(self):
        return self._get_levels_dict_virtual()

    ##REF: Name was automagically refactored
    def _get_levels_dict_virtual(self):
        return self._levels_dict

    levels_dict = property(fget=_get_levels_dict)


    origin = property(fget=lambda self:self._origin)
    extent = property(fget=lambda self:self._extent)
    voxdim = property(fget=lambda self:self._voxdim)
    spaceT = property(fget=lambda self:self._spaceT)
    coordT = property(fget=lambda self:self._spaceT,
                      fset=set_coord_t)

class Label(object):
    """Represents a label. Just to bring all relevant information together
    """
    def __init__ (self, text, abbr=None, coord=(None, None, None),
                  count=0, index=0):
        """
        Parameters
        ----------
        text : str
          Fullname of the label
        abbr : str, optional
          Abbreviated name.
        coord : tuple of float, optional
          Coordinates.
        count : int, optional
          Count of those labels in the atlas
        """
        self.__text = text.strip()
        if abbr is not None:
            abbr = abbr.strip()
        self.__abbr = abbr
        self.__coord = coord
        self.__count = count
        self.__index = int(index)


    @property
    def index(self):
        return self.__index

    def __repr__(self):
        return "Label(%s%s, coord=(%s, %s, %s), count=%s, index=%s)" % \
               ((self.__text,
                (', abbr=%s' % repr(self.__abbr), '')[int(self.__abbr is None)])
                + tuple(self.__coord) + (self.__count, self.__index))

    def __str__(self):
        return self.__text

    @staticmethod
    ##REF: Name was automagically refactored
    def from_xml(Elabel):
        kwargs = {}
        if Elabel.attrib.has_key('x'):
            kwargs['coord'] = ( Elabel.attrib.get('x'),
                                Elabel.attrib.get('y'),
                                Elabel.attrib.get('z') )
        for l in ('count', 'abbr', 'index'):
            if Elabel.attrib.has_key(l):
                kwargs[l] = Elabel.attrib.get(l)
        return Label(Elabel.text.strip(), **kwargs)

    @property
    def count(self): return self.__count
    @property
    def coord(self): return self.__coord
    @property
    def text(self):  return self.__text
    @property
    def abbr(self):
        """Returns abbreviated version if such is available
        """
        if self.__abbr in [None, ""]:
            return self.__text
        else:
            return self.__abbr


class Level(object):
    """Represents a level. Just to bring all relevant information together
    """
    def __init__ (self, description):
        self.description = description
        self._type = "Base"

    def __repr__(self):
        return "%s Level: %s" % \
               (self.level_type, self.description)

    def __str__(self):
        return self.description

    @staticmethod
    ##REF: Name was automagically refactored
    def from_xml(Elevel, level_type=None):
        """Simple factory of levels
        """
        if level_type is None:
            if not Elevel.attrib.has_key("type"):
                raise XMLAtlasException("Level must have type specified. Level: " + `Elevel`)
            level_type = Elevel.get("type")

        levelTypes = { 'label':     LabelsLevel,
                       'reference': ReferencesLevel }

        if levelTypes.has_key(level_type):
            return levelTypes[level_type].from_xml(Elevel)
        else:
            raise XMLAtlasException("Unknown level type " + level_type)

    level_type = property(lambda self: self._type)


class LabelsLevel(Level):
    """Level of labels.

	XXX extend
    """
    def __init__ (self, description, index=None, labels=[]):
        Level.__init__(self, description)
        self.__index = index
        self.__labels = labels
        self._type = "Labels"

    def __repr__(self):
        return Level.__repr__(self) + " [%d] " % \
               (self.__index)

    @staticmethod
    ##REF: Name was automagically refactored
    def from_xml(Elevel, levelIndex=[0]):
        # XXX this is just for label type of level. For distance we need to ...
        # we need to assure the right indexing

        index = 0
        if Elevel.attrib.has_key("index"):
            index = int(Elevel.get("index"))

        maxindex = max([int(i.get('index')) \
                        for i in Elevel.label[:]])
        labels = [ None for i in xrange(maxindex+1) ]
        for label in Elevel.label[:]:
            labels[ int(label.get('index')) ] = Label.from_xml(label)

        levelIndex[0] = max(levelIndex[0], index) + 1 # assign next one

        return LabelsLevel(Elevel.get('description'),
                           index,
                           labels)

    @property
    def index(self): return self.__index

    @property
    def labels(self): return self.__labels

    def __getitem__(self, index):
        return self.__labels[index]

    def find(self, target, unique=True):
        """Return labels descr of which matches the string

        Parameters
        ----------
        target : str or re._pattern_type
          Substring in abbreviation to be searched for, or compiled
          regular expression to be searched or matched if anchored.
        unique : bool, optional
          If True, raise exception if none or more than 1 was found. Return
          just a single item if found (not list).
        """
        if isinstance(target, re._pattern_type):
            res = [l for l in self.__labels if target.search(l.abbr)]
        else:
            res = [l for l in self.__labels if target in l.abbr]

        if unique:
            if len(res) != 1:
                raise ValueError, "Got %d matches whenever just 1 was " \
                      "looked for (target was %s)." % (len(res), target)
            return res[0]
        else:
            return res


class ReferencesLevel(Level):
    """Level which carries reference points
    """
    def __init__ (self, description, indexes=[]):
        Level.__init__(self, description)
        self.__indexes = indexes
        self._type = "References"

    @staticmethod
    ##REF: Name was automagically refactored
    def from_xml(Elevel):
        # XXX should probably do the same for the others?
        requiredAttrs = ['x', 'y', 'z', 'type', 'description']
        if not set(requiredAttrs) == set(Elevel.attrib.keys()):
            raise XMLAtlasException("ReferencesLevel has to have " +
                                    "following attributes defined " +
                                    `requiredAttrs`)

        indexes = ( int(Elevel.get("x")), int(Elevel.get("y")),
                    int(Elevel.get("z")) )

        return ReferencesLevel(Elevel.get('description'),
                               indexes)

    @property
    def indexes(self): return self.__indexes


class PyMVPAAtlas(XMLBasedAtlas):
    """Base class for PyMVPA atlases, such as LabelsAtlas and ReferenceAtlas
    """

    source = 'PyMVPA'

    def __init__(self, *args, **kwargs):
        XMLBasedAtlas.__init__(self, *args, **kwargs)

        # sanity checks
        header = self.header
        headerChildrenTags = XMLBasedAtlas._children_tags(header)
        if not ('space' in headerChildrenTags) or \
           not ('space-flavor' in headerChildrenTags):
            raise XMLAtlasException("PyMVPA Atlas requires specification of" +
                                    " the space in which atlas resides")

        self.__space = header.space.text
        self.__spaceFlavor = header['space-flavor'].text


    __doc__ = enhanced_doc_string('PyMVPAAtlas', locals(), XMLBasedAtlas)


    ##REF: Name was automagically refactored
    def _load_images(self):
        # shortcut
        imagefile = self.header.images.imagefile
        #self.n_levels = len(self._levels_by_id)

        # Set offset if defined in XML file
        # XXX: should just take one from the qoffset... now that one is
        #       defined... this origin might be misleading actually
        self._origin = np.array( (0, 0, 0) )
        if imagefile.attrib.has_key('offset'):
            self._origin = np.array( [int(x) for x in
                                     imagefile.get('offset').split(',')] )

        # Load the image file which has labels
        if self._force_image_file is not None:
            imagefilename = self._force_image_file
        else:
            imagefilename = imagefile.text
        imagefilename = reuse_absolute_path(self._filename, imagefilename)

        try:
            self._image  = NiftiImage(imagefilename)
        except RuntimeError, e:
            raise RuntimeError, \
                  " Cannot open file %s due to %s" % (imagefilename, e)

        self._data = self._image.data

        # remove bogus dimensions on top of 4th
        if len(self._data.shape[0:-4]) > 0:
            bogus_dims = self._data.shape[0:-4]
            if max(bogus_dims)>1:
                raise RuntimeError, "Atlas %s has more than 4 of non-singular" \
                      "dimensions" % imagefilename
            new_shape = self._data.shape[-4:]
            self._data.reshape(new_shape)

        #if self._image.extent[3] != self.n_levels:
        #   raise XMLAtlasException("Atlas %s has %d levels defined whenever %s has %d volumes" % \
        #                           ( filename, self.n_levels, imagefilename, self._image.extent[3] ))


    ##REF: Name was automagically refactored
    def _load_data(self):
        # Load levels
        self._levels_dict = {}
        # preprocess labels for different levels
        self._Nlevels = 0
        index_incr = 0
        for index, child in enumerate(self.data.getchildren()):
            if child.tag == 'level':
                level = Level.from_xml(child)
                self._levels_dict[level.description] = level
                if hasattr(level, 'index'):
                    index = level.index
                else:
                    # to avoid collision if some levels do
                    # have indexes
                    while index_incr in self._levels_dict:
                        index_incr += 1
                    index, index_incr = index_incr, index_incr+1
                self._levels_dict[index] = level
            else:
                raise XMLAtlasException(
                    "Unknown child '%s' within data" % child.tag)
            self._Nlevels += 1


    ##REF: Name was automagically refactored
    def _get_n_levels_virtual(self):
        return self._Nlevels

    ##REF: Name was automagically refactored
    def _get_n_levels(self):
        return self._get_n_levels_virtual()

    @staticmethod
    ##REF: Name was automagically refactored
    def _check_version(version):
        # For compatibility lets support "RUMBA" atlases
        return version.startswith("pymvpa-") or version.startswith("rumba-")


    space = property(fget=lambda self:self.__space)
    space_flavor = property(fget=lambda self:self.__spaceFlavor)
    n_levels = property(fget=_get_n_levels)


class LabelsAtlas(PyMVPAAtlas):
    """
    Atlas which provides labels for the given coordinate
    """

    ##REF: Name was automagically refactored
    def label_voxel(self, c, levels=None):
        """
        Return labels for the given voxel at specified levels specified by index
        """
        levels = self._get_levels(levels=levels)

        result = {'voxel_queried' : c}

        # check range
        c = self._check_range(c)

        resultLevels = []
        for level in levels:
            if self._levels_dict.has_key(level):
                level_ = self._levels_dict[ level ]
            else:
                raise IndexError(
                    "Unknown index or description for level %d" % level)

            resultIndex =  int(self._data[ level_.index, \
                                            c[2], c[1], c[0] ])

            resultLevels += [ {'index': level_.index,
                               'id': level_.description,
                               'label' : level_[ resultIndex ]} ]

        result['labels'] = resultLevels
        return result

    __doc__ = enhanced_doc_string('LabelsAtlas', locals(), PyMVPAAtlas)



class ReferencesAtlas(PyMVPAAtlas):
    """
    Atlas which provides references to the other atlases.

    Example: the atlas which has references to the closest points
    (closest Gray, etc) in another atlas.
    """

    def __init__(self, distance=0, *args, **kwargs):
        """Initialize `ReferencesAtlas`
        """
        PyMVPAAtlas.__init__(self, *args, **kwargs)
        # sanity checks
        if not ('reference-atlas' in XMLBasedAtlas._children_tags(self.header)):
            raise XMLAtlasException(
                "ReferencesAtlas must refer to a some other atlas")

        referenceAtlasName = self.header["reference-atlas"].text

        # uff -- another evil import but we better use the factory method
        from mvpa.atlases.warehouse import Atlas
        self.__referenceAtlas = Atlas(filename=reuse_absolute_path(
            self._filename, referenceAtlasName))

        if self.__referenceAtlas.space != self.space or \
           self.__referenceAtlas.space_flavor != self.space_flavor:
            raise XMLAtlasException(
                "Reference and original atlases should be in the same space")

        self.__referenceLevel = None
        self.set_distance(distance)

    __doc__ = enhanced_doc_string('ReferencesAtlas', locals(), PyMVPAAtlas)

    # number of levels must be of the referenced atlas due to
    # handling of that in __getitem__
    #n_levels = property(fget=lambda self:self.__referenceAtlas.n_levels)
    ##REF: Name was automagically refactored
    def _get_n_levels_virtual(self):
        return self.__referenceAtlas.n_levels


    ##REF: Name was automagically refactored
    def set_reference_level(self, level):
        """
        Set the level which will be queried
        """
        if self._levels_dict.has_key(level):
            self.__referenceLevel = self._levels_dict[level]
        else:
            raise IndexError, \
                  "Unknown reference level %r. " % level + \
                  "Known are %r" % (self._levels_dict.keys(), )


    ##REF: Name was automagically refactored
    def label_voxel(self, c, levels = None):

        if self.__referenceLevel is None:
            warning("You did not provide what level to use "
                    "for reference. Assigning 0th level -- '%s'"
                    % (self._levels_dict[0],))
            self.set_reference_level(0)
            # return self.__referenceAtlas.label_voxel(c, levels)

        c = self._check_range(c)

        # obtain coordinates of the closest voxel
        cref = self._data[ self.__referenceLevel.indexes, c[2], c[1], c[0] ]
        dist = norm( (cref - c) * self.voxdim )
        if __debug__:
            debug('ATL__', "Closest referenced point for %r is "
                  "%r at distance %3.2f" % (c, cref, dist))
        if (self.distance - dist) >= 1e-3: # neglect everything smaller
            result = self.__referenceAtlas.label_voxel(cref, levels)
            result['voxel_referenced'] = c
            result['distance'] = dist
        else:
            result = self.__referenceAtlas.label_voxel(c, levels)
            if __debug__:
                debug('ATL__', "Closest referenced point is "
                      "further than desired distance %.2f" % self.distance)
            result['voxel_referenced'] = None
            result['distance'] = 0
        return result


    ##REF: Name was automagically refactored
    def levels_listing(self):
        return self.__referenceAtlas.levels_listing()

    ##REF: Name was automagically refactored
    def _get_levels_dict_virtual(self):
        return self.__referenceAtlas.levels_dict

    ##REF: Name was automagically refactored
    def set_distance(self, distance):
        """Set desired maximal distance for the reference
        """
        if distance < 0:
            raise ValueError("Distance should not be negative. "
                             " Thus '%f' is not a legal value" % distance)
        if __debug__:
            debug('ATL__',
                  "Setting maximal distance for queries to be %d" % distance)
        self.__distance = distance

    distance = property(fget=lambda self:self.__distance, fset=set_distance)

