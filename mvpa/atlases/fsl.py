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

import os, re
import numpy as N

from mvpa.base import warning, externals
from mvpa.misc.support import reuseAbsolutePath

if externals.exists('nifti', raiseException=True):
    from nifti import NiftiImage

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
        #     effort with PyMVPAAtlas
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
        if __debug__:
            debug('ATL__', "Loading atlas data from %s" % self._imagefile)
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


