#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: t -*- 
#ex: set sts=4 ts=4 sw=4 noet:
#------------------------- =+- Python script -+= -------------------------
"""
 @file      atlas.py
 @date      Wed Aug 15 14:38:32 2007
 @brief


  RUMBA project                    Psychology Department. Rutgers, Newark
  http://psychology.rutgers.edu/RUMBA       http://psychology.rutgers.edu
  e-mail: yoh@psychology.rutgers.edu

 DESCRIPTION (NOTES):


  TODOs:
	_origin can be dumped now (?) since we use qform
	_voxdim only used to compute distance in the case of reference atlas

 COPYRIGHT: Yaroslav Halchenko 2007

 LICENSE:

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the 
  Free Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,
  MA 02110-1301, USA.

 On Debian system see /usr/share/common-licenses/GPL for the full license.
"""
#-----------------\____________________________________/------------------

__author__ = 'Yaroslav Halhenko'
__revision__ = '$Revision: $'
__date__ = '$Date:  $'
__copyright__ = 'Copyright (c) 2007 Yaroslav Halchenko'
__license__ = 'GPL'


from lxml import etree, objectify
import os, re
import numpy as np
import numpy.linalg as la
from nifti import NiftiImage

from rumba.tools.transformation import SpaceTransformation, Linear
from rumba.tools.misc import reuseAbsolutePath

try:
	from rumba.tools.debug import printdebug, setVerbosity
except:
	print "No verbose output available"
	def setVerbosity(*args, **keywords):		pass
	def printdebug(*args, **keywords):          pass


KNOWN_ATLAS_FAMILIES = { 'rumba': (["talairach", "talairach-dist"],
									r"/usr/share/rumba/atlases/data/%(name)s_atlas.xml"),
						  'fsl': (["HarvardOxford-Cortical", "HarvardOxford-Subcortical",
								   "JHU-tracts", "Juelich", "MNI", "Thalamus"],
								  r"/usr/share/fsl/data/atlases/%(name)s.xml") # XXX make use of FSLDIR
						  }

# map to go from the name to the path
KNOWN_ATLASES = dict(reduce(lambda x,y:x+[(yy,y[1]) for yy in y[0]],
							 KNOWN_ATLAS_FAMILIES.values(), []))


def checkRange(coord, range):
	"""
	Check if coordinates are within range (0,0,0) - (range)
	Return True on success
	"""
	if len(coord) != len(range):
		raise ValueError("Provided coordinate %s and given range %s" % \
						 (`coord`, `range`) + \
						 " have different dimensionality"
						 )
	for c,r in zip(coord, range):
		if c<0 or c>=r:
			return False
	return True


class BaseAtlas:
	"""
	Base class for the atlases.
	"""

	def __init__ (self):
		"""
		Create an atlas object based on the... XXX
		"""
		self.__name = "blank"			# XXX use or remove


class XMLAtlasException(Exception):
	"""
	Exception to be thrown if smth goes wrong dealing with
	XML based atlas
	"""
	def __init__(self, msg=""):
		self.__msg = msg
	def __repr__(self):
		return self.__msg


class XMLBasedAtlas(BaseAtlas):

	def __init__(self, filename=None, resolution=None, query_voxel=False,
				 coordT=None, levels=None):
		"""
		:Parameters:
		  filename : string
		    Filename for the xml definition of the atlas
		  resolution : None or float
		    Some atlases link to multiple images at different
			resolutions. if None -- best resolution is selected
			using 0th dimension resolution
		  query_voxel : bool
		    By default [x,y,z] assumes coordinates in space, but if
			query_voxel is True, they are assumed to be voxel coordinates
		  coordT
		    Optional transformation to apply first
		  levels : None or slice or list of int
		    What levels by default to operate on
		"""
		BaseAtlas.__init__(self)
		self.__version = None
		self.__type = None				# XXX use or remove
		self._imagefile = None
		self.__atlas = None
		self._filename = filename
		self._resolution = resolution
		self.query_voxel = query_voxel
		self.levels = levels

		if filename:
			self.loadAtlas(filename)

		# common sanity checks
		if not self._checkVersion(self.version):
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
		self._loadImages()
		if self._image is not None:
			self._extent = np.abs(np.asanyarray(self._image.extent[0:3]))
			self._voxdim = np.asanyarray(self._image.voxdim)
			self.relativeToOrigin = True
		# Assign transformation to get into voxel coordinates,
		# spaceT will be set accordingly
		self.setCoordT(coordT)
		self._loadData()


	def _checkRange(self, c):
		""" check and adjust the voxel coordinates"""
		# check range
		# list(c) for consistent appearance... some times c might be ndarray
		printdebug("Querying for voxel %s" % `list(c)`, 5)
		if not checkRange(c, self.extent):
			msg = "Coordinates %s are not within the extent %s." \
				  "Reset to (0,0,0)" % ( `c`, `self.extent` )
			printdebug(msg, 2)
			# assume that voxel [0,0,0] is blank
			c = [0]*3;
		return c


	@staticmethod
	def _checkVersion(version):
		"""To be overriden in the derived classes. By default anything is good"""
		return True


	def _loadImages(self):
		"""To be overriden in the derived classes. By default does nothing"""
		pass


	def _loadData(self):
		"""To be overriden in the derived classes. By default does nothing"""
		pass


	def loadAtlas(self, filename):
		printdebug("Loading atlas definition xml file " + filename, 3)
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


	def setCoordT(self, coordT):
		"""Set coordT transformation.

		spaceT needs to be adjusted since we glob those two
		transformations together
		"""
		self._coordT = coordT			# lets store for debugging etc
		if self._image is not None:
			# Combine with the image's qform
			coordT = Linear(np.linalg.inv(self._image.qform),
							previous=coordT)
		self._spaceT = SpaceTransformation(
			previous=coordT, toRealSpace=False
			)


	def labelPoint(self, coord, levels=None):
		"""
		Return labels for the given spatial point at specified levels specified by index,
		so we first transform point into the voxel space
		"""
		coord_ = np.asarray(coord)			# or we would alter what should be constant
		#if not isinstance(coord, np.numpy):
		#c = self.getVolumeCoordinate(coord)
		#c = self.spaceT.toVoxelSpace(coord_)
		#if self.coordT:
		#	coord_t = self.coordT[coord_]
		#else:
		#	coord_t = coord_

		c = self.spaceT(coord_)

		result = self.labelVoxel(c, levels)
		result['coord_queried'] = coord
		#result['coord_trans'] = coord_t
		result['voxel_atlas'] = c
		return result


	def levelsListing(self):
		lkeys = range(self.Nlevels)
		return '\n'.join(['%d: ' % k + str(self._levels_dict[k])
						  for k in lkeys])


	def _getLevels(self, levels=None):
		"""Helper to provide list of levels to operate on

		Depends on given `levels` as well as self.levels
		"""

		if (isinstance(levels, slice)):
			# levels are given as a range
			if levels.step: step = levels.step
			else: step = 1

			if levels.start: start = levels.start
			else: start = 0

			if levels.stop: stop = levels.stop
			else: stop = self.Nlevels

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
					  % (level, self.levelsListing())

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

		levels = self._getLevels(levels=levels_slice)

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
			return self.labelVoxel(coord, levels)
		else:
			return self.labelPoint(coord, levels)


	# REDO in some sane fashion so referenceatlas returns levels for the base
	def _getLevelsDict(self):
		return self._getLevelsDict_virtual()

	def _getLevelsDict_virtual(self):
		return self._levels_dict

	levels_dict = property(fget=_getLevelsDict)


	origin = property(fget=lambda self:self._origin)
	extent = property(fget=lambda self:self._extent)
	voxdim = property(fget=lambda self:self._voxdim)
	spaceT = property(fget=lambda self:self._spaceT)
	coordT = property(fget=lambda self:self._spaceT,
					  fset=setCoordT)

class Label:
	"""
	Simple class to represent a label. Just to bring all relevant
	information together
	"""
	def __init__ (self, text, abbr=None, coord=(None, None,None), count=0, index=0):
		"""
		:Parameters:
		  text : basestring
		    fullname of the label
		  abbr : basestring
		    abbreviated name (optional)
		  coord : tuple of float
		    coordinates (optional)
		  count : int
		    count of those labels in the atlas (optional)

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
	def generateFromXML(Elabel):
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


class Level:
	"""
	Simple class to represent a level. Just to bring all relevant
	information together
	"""
	def __init__ (self, description):
		self.description = description
		self._type = "Base"

	def __repr__(self):
		return "%s Level: %s" % \
			   (self.levelType, self.description)

	def __str__(self):
		return self.description

	@staticmethod
	def generateFromXML(Elevel, levelType=None):
		"""
		Simple factory of levels
		"""
		if levelType is None:
			if not Elevel.attrib.has_key("type"):
				raise XMLAtlasException("Level must have type specified. Level: " + `Elevel`)
			levelType = Elevel.get("type")

		levelTypes = { 'label':     LabelsLevel,
					   'reference': ReferencesLevel }

		if levelTypes.has_key(levelType):
			return levelTypes[levelType].generateFromXML(Elevel)
		else:
			raise XMLAtlasException("Unknown level type " + levelType)

	levelType = property(lambda self: self._type)


class LabelsLevel(Level):
	"""
	Simple class to represent a level. Just to bring all relevant
	information together
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
	def generateFromXML(Elevel, levelIndex=[0]):
		# XXX this is just for label type of level. For distance we need to ...
		# we need to assure the right indexing

		index = 0
		if Elevel.attrib.has_key("index"):
			index = int(Elevel.get("index"))

		maxindex = max([int(i.get('index')) \
						for i in Elevel.label[:]])
		labels = [ None for i in xrange(maxindex+1) ]
		for label in Elevel.label[:]:
			labels[ int(label.get('index')) ] = Label.generateFromXML(label)

		levelIndex[0] = max(levelIndex[0], index) + 1 # assign next one

		return LabelsLevel(Elevel.get('description'),
						   index,
						   labels)

	@property
	def index(self): return self.__index

	def __getitem__(self, index):
		return self.__labels[index]


class ReferencesLevel(Level):
	"""
	Level which carries reference points
	"""
	def __init__ (self, description, indexes=[]):
		Level.__init__(self, description)
		self.__indexes = indexes
		self._type = "References"

	@staticmethod
	def generateFromXML(Elevel):
		# XXX should probably do the same for the others?
		requiredAttrs = ['x', 'y', 'z', 'type', 'description']
		if not set(requiredAttrs) == set(Elevel.attrib.keys()):
			raise XMLAtlasException("ReferencesLevel has to have " +
									"following attributes defined " +
									`requiredAttrs`)

		indexes = ( int(Elevel.get("x")), int(Elevel.get("y")), int(Elevel.get("z")) )

		return ReferencesLevel(Elevel.get('description'),
							   indexes)

	@property
	def indexes(self): return self.__indexes


class RumbaAtlas(XMLBasedAtlas):
	"""
	Base class for RUMBA atlases, such as LabelsAtlas and ReferenceAtlas
	"""
	source = 'RUMBA'
	def __init__(self, *args, **kwargs):
		XMLBasedAtlas.__init__(self, *args, **kwargs)

		# sanity checks
		header = self.header
		headerChildrenTags = XMLBasedAtlas._children_tags(header)
		if not ('space' in headerChildrenTags) or \
		   not ('space-flavor' in headerChildrenTags):
			raise XMLAtlasException("Rumba Atlas requires specification of" +
									" the space in which atlas resides")

		self.__space = header.space.text
		self.__spaceFlavor = header['space-flavor'].text


	def _loadImages(self):
		# shortcut
		imagefile = self.header.images.imagefile
		#self.Nlevels = len(self._levels_by_id)

		# Set offset if defined in XML file
		# XXX: should just take one from the qoffset... now that one is
		#       defined... this origin might be misleading actually
		self._origin = np.array( (0,0,0) )
		if imagefile.attrib.has_key('offset'):
			self._origin = np.array( map(int, imagefile.get('offset').split(',')) )

		# Load the image file which has labels
		imagefilename = reuseAbsolutePath(self._filename, imagefile.text)

		try:
			self._image  = NiftiImage(imagefilename)
		except RuntimeError, e:
			raise RuntimeError, " Cannot open file " + imagefilename

		self._data   = self._image.data

		# remove bogus dimensions on top of 4th
		if len(self._data.shape[0:-4]) > 0:
			bogus_dims = self._data.shape[0:-4]
			if max(bogus_dims)>1:
				raise RuntimeError, "Atlas " + imagefilename + " has more than 4 of non-singular dimensions"
			new_shape = self._data.shape[-4:]
			self._data.reshape(new_shape)

		#if self._image.extent[3] != self.Nlevels:
		#	raise XMLAtlasException("Atlas %s has %d levels defined whenever %s has %d volumes" % \
		#							( filename, self.Nlevels, imagefilename, self._image.extent[3] ))


	def _loadData(self):
		# Load levels
		self._levels_dict = {}
		# preprocess labels for different levels
		self.Nlevels = 0
		for index, child in enumerate(self.data.getchildren()):
			if child.tag == 'level':
				level = Level.generateFromXML(child)
				self._levels_dict[level.description] = level
				try:
					self._levels_dict[level.index] = level
				except:
					pass
			else:
				raise XMLAtlasException("Unknown child '%s' within data" % child.tag)
			self.Nlevels += 1


	@staticmethod
	def _checkVersion(version):
		return version.startswith("rumba-")


	space = property(fget=lambda self:self.__space)
	spaceFlavor = property(fget=lambda self:self.__spaceFlavor)



class LabelsAtlas(RumbaAtlas):
	"""
	Atlas which provides labels for the given coordinate
	"""

	def labelVoxel(self, c, levels=None):
		"""
		Return labels for the given voxel at specified levels specified by index
		"""
		levels = self._getLevels(levels=levels)

		result = {'voxel_queried' : c}

		# check range
		c = self._checkRange(c)

		resultLevels = []
		for level in levels:
			if self._levels_dict.has_key(level):
				level_ = self._levels_dict[ level ]
			else:
				raise IndexError("Unknown index or description for level %d" % level)

			resultIndex =  int(self._data[ level_.index, \
											c[2], c[1], c[0] ])

			resultLevels += [ {'index': level_.index,
							   'id': level_.description,
							   'label' : level_[ resultIndex ]} ]

		result['labels'] = resultLevels
		return result


class ReferencesAtlas(RumbaAtlas):
	"""
	Atlas which provides references to the other atlases. For instance the atlas which has
	references to the closest points (closest Gray, etc) in another atlas.
	"""
	def __init__(self, distance=0, *args, **kwargs):
		RumbaAtlas.__init__(self, *args, **kwargs)
		# sanity checks
		if not ('reference-atlas' in XMLBasedAtlas._children_tags(self.header)):
			raise XMLAtlasException("ReferencesAtlas must refer to a some other atlas")

		referenceAtlasName = self.header["reference-atlas"].text
		self.__referenceAtlas = Atlas(reuseAbsolutePath(self._filename, referenceAtlasName))

		if self.__referenceAtlas.space != self.space or \
		   self.__referenceAtlas.spaceFlavor != self.spaceFlavor:
			raise XMLAtlasException("Reference and original atlases should be in the same space")

		self.__referenceLevel = None
		self.setDistance(distance)

	# number of levels must be of the referenced atlas due to
	# handling of that in __getitem__
	Nlevels = property(fget=lambda self:self.__referenceAtlas.Nlevels)

	def setReferenceLevel(self, level):
		"""
		Set the level which will be queried
		"""
		if self._levels_dict.has_key(level):
			self.__referenceLevel = self._levels_dict[level]
		else:
			raise IndexError("Unknown reference level " + `level` +\
							 ". Known are " + `self._levels_dict.keys()`)


	def labelVoxel(self, c, levels = None):

		if self.__referenceLevel is None:
			printdebug("WARNING: You did not provide what level to use "
					   "for reference. No referencing is done", 0)
			return self.__referenceAtlas.labelVoxel(c, levels)

		c = self._checkRange(c)

		# obtain coordinates of the closest voxel
		cref = self._data[ self.__referenceLevel.indexes, c[2], c[1], c[0] ]
		dist = la.norm( (cref - c) * self.voxdim )
		printdebug("Closest referenced point for %s is %s at distance %3.2f"
				   % (`c`, `cref`, dist), 5)
		if (self.distance - dist) >= 1e-3: # neglect everything smaller
			result = self.__referenceAtlas.labelVoxel(cref, levels)
			result['voxel_referenced'] = c
			result['distance'] = dist
		else:
			result = self.__referenceAtlas.labelVoxel(c, levels)
			printdebug("Closest referenced point is further than desired "
					   "distance %.2f" % self.distance, 5)
			result['voxel_referenced'] = None
			result['distance'] = 0
		return result


	def levelsListing(self):
		return self.__referenceAtlas.levelsListing()

	def _getLevelsDict_virtual(self):
		return self.__referenceAtlas.levels_dict

	def setDistance(self, distance):
		"""
		Set desired maximal distance for the reference
		"""
		if distance < 0:
			raise ValueError("Distance should not be negative. Thus '%f' is not a legal value" % distance)
		printdebug("Setting maximal distance for queries to be %d" % distance, 5)
		self.__distance = distance

	distance = property(fget=lambda self:self.__distance, fset=setDistance)


#
# Atlases from FSL support
#

class FSLAtlas(XMLBasedAtlas):
	"""Base class for FSL atlases

	"""
	source = 'FSL'

	def __init__(self, *args, **kwargs):
		"""

		:Parameters:
		  filename : string
		    Filename for the xml definition of the atlas
		  resolution : None or float
		    Some atlases link to multiple images at different
			resolutions. if None -- best resolution is selected
			using 0th dimension resolution
		"""
		XMLBasedAtlas.__init__(self, *args, **kwargs)
		self.space = 'MNI'


	def _loadImages(self):
		resolution = self._resolution
		header = self.header
		images = header.images
		# Load present images
		# XXX might be refactored to avoid duplication of
		#     effort with RumbaAtlas
		ni_image = None
		resolutions = []
		for image in images:
			imagefile = image.imagefile
			imagefilename = reuseAbsolutePath(
				self._filename, imagefile.text, force=True)

			try:
				ni_image_  = NiftiImage(imagefilename, load=False)
			except RuntimeError, e:
				raise RuntimeError, " Cannot open file " + imagefilename

			resolution_ = ni_image_.pixdim[0]
			if resolution is None:
				# select this one if the best
				if ni_image is None or \
					   resolution_ < ni_image.pixdim[0]:
					ni_image = ni_image_
					self._imagefile = imagefilename
			else:
				if resolution_ == resolution:
					ni_image = ni_image_
					self._imagefile = imagefilename
					break
				else:
					resolutions += [resolution_]
			# TODO: also make use of summaryimagefile may be?

		if ni_image is None:
			msg = "Could not find an appropriate atlas among %d atlases."
			if resolution is not None:
				msg += " Atlases had resolutions %s" % \
					  (resolutions,)
			raise RuntimeError, msg
		printdebug("Loading atlas data from %s" % self._imagefile, 4)
		self._image = ni_image
		self._resolution = ni_image.pixdim[0]
		self._origin = np.abs(ni_image.header['qoffset']) * 1.0  # XXX
		self._data   = self._image.data


	def _loadData(self):
		"""   """
		# Load levels
		self._levels_dict = {}
		# preprocess labels for different levels
		self.Nlevels = 1
		#level = Level.generateFromXML(self.data, levelType='label')
		level = LabelsLevel.generateFromXML(self.data)#, levelType='label')
		level.description = self.header.name.text
		self._levels_dict = {0: level}
		#for index, child in enumerate(self.data.getchildren()):
		#	if child.tag == 'level':
		#		level = Level.generateFromXML(child)
		#		self._levels_dict[level.description] = level
		#		try:
		#			self._levels_dict[level.index] = level
		#		except:
		#			pass
		#	else:
		#		raise XMLAtlasException("Unknown child '%s' within data" % child.tag)
		#	self.Nlevels += 1
		#pass


	@staticmethod
	def _checkVersion(version):
		return re.search('^[0-9]+\.[0-9]', version) is not None


class FSLProbabilisticAtlas(FSLAtlas):

	def __init__(self, thr=0.0, strategy='all', sort=True, *args, **kwargs):
		"""

		:Parameters:
		  thr : float
		    Value to threshold at
		  strategy : basestring
		    Possible values
			  all - all entries above thr
			  max - entry with maximal value
		  sort : bool
		    Either to sort entries for 'all' strategy according to
			probability
		"""

		FSLAtlas.__init__(self, *args, **kwargs)
		self.thr = thr
		self.strategy = strategy
		self.sort = sort

	def labelVoxel(self, c, levels=None):
		if levels is not None and not (levels in [0, [0], (0,)]):
			raise ValueError, \
				  "I guess we don't support levels other than 0 in FSL atlas"

		# check range
		c = self._checkRange(c)

		# XXX think -- may be we should better assign each map to a
		# different level
		level = 0
		resultLabels = []
		for index, area in enumerate(self._levels_dict[level]):
			prob =  int(self._data[index, c[2], c[1], c[0]])
			if prob > self.thr:
				resultLabels += [dict(index=index,
									  #id=
									  label=area.text,
									  prob=prob)]

		if self.sort or self.strategy == 'max':
			resultLabels.sort(cmp=lambda x,y: cmp(x['prob'], y['prob']),
							  reverse=True)

		if self.strategy == 'max':
			resultLabels = resultLabels[:1]
		elif self.strategy == 'all':
			pass
		else:
			raise ValueError, 'Unknown strategy %s' % self.strategy

		result = {'voxel_queried' : c,
				  # in the list since we have only single level but
				  # with multiple entries
				  'labels': [resultLabels]}

		return result


class FSLLabelsAtlas(XMLBasedAtlas):
	def __init__(self, *args, **kwargs):
		FSLAtlas.__init__(self, *args, **kwargs)
		raise NotImplementedError


def Atlas(filename=None, name=None, *args, **kwargs):
	"""
	Somewhat of a factory for the atlases
	"""
	if filename is None:
		if name is None:
			raise ValueError, "Please provide either path or name of the atlas to be used"
		atlaspath = KNOWN_ATLASES[name]
		filename = atlaspath % ( {'name': name} )
	else:
		if name is not None:
			raise ValueError, "Provide only filename or name"

	try:
		tempAtlas = XMLBasedAtlas(filename=filename, *args, **kwargs)
		version = tempAtlas.version
		atlas_source = None
		for cls in [RumbaAtlas, FSLAtlas]:
			if cls._checkVersion(version):
				atlas_source = cls.source
				break
		if atlas_source is None:
			printdebug("Unknown atlas " + filename, 2)
			return tempAtlas

		atlasTypes = {
			'RUMBA': {"Label" : LabelsAtlas,
					 "Reference": ReferencesAtlas},
			'FSL': {"Label" : FSLLabelsAtlas,
					"Probabalistic": FSLProbabilisticAtlas}
			}[atlas_source]
		atlasType = tempAtlas.header.type.text
		if atlasTypes.has_key(atlasType):
			printdebug("Creating %s Atlas" % atlasType, 3)
			return atlasTypes[atlasType](filename=filename, *args, **kwargs)
		    #return ReferencesAtlas(filename)
		else:
			printdebug("Unknown %s type '%s' of atlas in %s."\
					   " Known are %s" %
					   (atlas_source, atlasType, filename,
						atlasTypes.keys()), 2)
			return tempAtlas
	except XMLAtlasException, e:
		printdebug("File %s is not a valid XML based atlas due to %s" %
				   (filename, `e`), 0)
		raise e


if __name__ == '__main__':
	setVerbosity(10)
	for name in [
		'data/talairach_atlas.xml',
		'/usr/share/fsl/data/atlases/HarvardOxford-Cortical.xml',
		'/usr/share/fsl/data/atlases/HarvardOxford-Subcortical.xml'
		]:
		atlas = Atlas(name)
		#print isinstance(atlas.atlas, objectify.ObjectifiedElement)
		#print atlas.header.images.imagefile.get('offset')
		#print atlas.labelVoxel( (0, -7, 20) )
		#print atlas[ 0, 0, 0 ]
		print atlas[ -63, -12, 22 ]
		#print atlas[ 0, -7, 20, [1,2,3] ]
		#print atlas[ (0, -7, 20), 1:2 ]
		#print atlas[ (0, -7, 20) ]
		#print atlas[ (0, -7, 20), : ]
		#	print atlas.getLabels(0)

