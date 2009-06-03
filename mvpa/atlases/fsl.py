# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""FSL atlases interfaces

"""

from mvpa.base import warning, externals

if externals.exists('nifti', raiseException=True):
    from nifti import NiftiImage

import os, re
import numpy as N

from mvpa.misc.support import reuseAbsolutePath
from mvpa.base.dochelpers import enhancedDocString

from mvpa.atlases.base import XMLBasedAtlas, LabelsLevel

if __debug__:
	from mvpa.base import debug

#
# Atlases from FSL
#

class FSLAtlas(XMLBasedAtlas):
    """Base class for FSL atlases

    """
    source = 'FSL'


    def __init__(self, *args, **kwargs):
        """
        """
        XMLBasedAtlas.__init__(self, *args, **kwargs)
        self.space = 'MNI'


    __doc__ = enhancedDocString('FSLAtlas', locals(), XMLBasedAtlas)


    def _loadImages(self):
        resolution = self._resolution
        header = self.header
        images = header.images
        # Load present images
        # XXX might be refactored to avoid duplication of
        #     effort with PyMVPAAtlas
        ni_image = None
        resolutions = []
        if self._force_image_file is None:
            imagefile_candidates = [
                reuseAbsolutePath(self._filename, i.imagefile.text, force=True)
                for i in images]
        else:
            imagefile_candidates = [self._force_image_file]

        for imagefilename in imagefile_candidates:
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
                    self._image_file = imagefilename
            else:
                if resolution_ == resolution:
                    ni_image = ni_image_
                    self._image_file = imagefilename
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
        if __debug__:
            debug('ATL__', "Loading atlas data from %s" % self._image_file)
        self._image = ni_image
        self._resolution = ni_image.pixdim[0]
        self._origin = N.abs(ni_image.header['qoffset']) * 1.0  # XXX
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
        #   if child.tag == 'level':
        #       level = Level.generateFromXML(child)
        #       self._levels_dict[level.description] = level
        #       try:
        #           self._levels_dict[level.index] = level
        #       except:
        #           pass
        #   else:
        #       raise XMLAtlasException("Unknown child '%s' within data" % child.tag)
        #   self.Nlevels += 1
        #pass


    @staticmethod
    def _checkVersion(version):
        return re.search('^[0-9]+\.[0-9]', version) is not None


class FSLProbabilisticAtlas(FSLAtlas):
    """Probabilistic FSL atlases
    """

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

    __doc__ = enhancedDocString('FSLProbabilisticAtlas', locals(), FSLAtlas)

    def labelVoxel(self, c, levels=None):
        """Return labels for the voxel

        :Parameters:
          - c : tuple of coordinates (xyz)
          - levels : just for API consistency (heh heh). Must be 0 for FSL atlases
        """

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

    def find(self, *args, **kwargs):
        """Just a shortcut to the only level.

        See :class:`~mvpa.atlases.base.Level.find` for more info
        """
        return self.levels_dict[0].find(*args, **kwargs)

    def getMap(self, target, strategy='unique'):
        """Return a probability map

        :Parameters:
          target : int or str or re._pattern_type
            If int, map for given index is returned. Otherwise, .find is called
            with unique=True to find matching area
          strategy : str in ('unique', 'max')
            If 'unique', then if multiple areas match, exception would be raised.
            In case of 'max', each voxel would get maximal value of probabilities
            from all matching areas
        """
        if isinstance(target, int):
            return self._data[target]
        else:
            lev = self.levels_dict[0]       # we have just 1 here
            if strategy == 'unique':
                return self.getMap(lev.find(target, unique=True).index)
            else:
                maps = N.array(self.getMaps(target))
                return N.max(maps, axis=0)

    def getMaps(self, target):
        """Return a list of probability maps for the target

        :Parameters:
          target : str or re._pattern_type
            .find is called with a target and unique=False to find all matches
        """
        lev = self.levels_dict[0]       # we have just 1 here
        return [self.getMap(l.index) for l in lev.find(target, unique=False)]


class FSLLabelsAtlas(XMLBasedAtlas):
    """Not sure what this one was for"""
    def __init__(self, *args, **kwargs):
        """not implemented"""
        FSLAtlas.__init__(self, *args, **kwargs)
        raise NotImplementedError


