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

from mvpa2.base import externals

if externals.exists('nibabel', raise_=True):
    import nibabel as nb

import re
import os.path
import numpy as np

from mvpa2.misc.support import reuse_absolute_path
from mvpa2.base.dochelpers import enhanced_doc_string

from mvpa2.atlases.base import XMLBasedAtlas, LabelsLevel

if __debug__:
	from mvpa2.base import debug

__all__ = [ "FSLAtlas", "FSLLabelsAtlas", "FSLProbabilisticAtlas" ]

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


    __doc__ = enhanced_doc_string('FSLAtlas', locals(), XMLBasedAtlas)


    ##REF: Name was automagically refactored
    def _load_images(self):
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
                reuse_absolute_path(self._filename, i.imagefile.text, force=True)
                for i in images]
        else:
            imagefile_candidates = [self._force_image_file]

        for imagefilename in imagefile_candidates:
            try:
                if not os.path.exists(imagefilename):
                    # try with extension if filename doesn't exist
                    imagefilename += '.nii.gz'
                ni_image_  = nb.load(imagefilename)
            except RuntimeError, e:
                raise RuntimeError, " Cannot open file " + imagefilename

            resolution_ = ni_image_.get_header().get_zooms()[0]
            if resolution is None:
                # select this one if the best
                if ni_image is None or \
                       resolution_ < ni_image.get_header().get_zooms()[0]:
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
            msg = "Could not find an appropriate atlas among %d atlases." \
                  % len(imagefile_candidates)
            if resolution is not None:
                msg += " Atlases had resolutions %s" % \
                      (resolutions,)
            raise RuntimeError, msg
        if __debug__:
            debug('ATL__', "Loading atlas data from %s" % self._image_file)
        self._image = ni_image
        self._resolution = ni_image.get_header().get_zooms()[0]
        self._origin = np.abs(ni_image.get_header().get_qform()[:3,3])  # XXX

        self._data   = self._image.get_data()
        if len(self._data.shape) == 4:
            # want to have volume axis first
            self._data = np.rollaxis(self._data, -1)


    def _load_metadata(self):
        """   """
        # Load levels
        self._levels = {}
        # preprocess labels for different levels
        self.nlevels = 1
        #level = Level.from_xml(self.data, level_type='label')
        level = LabelsLevel.from_xml(self.data)#, level_type='label')
        level.description = self.header.name.text
        self._levels = {0: level}
        #for index, child in enumerate(self.data.getchildren()):
        #   if child.tag == 'level':
        #       level = Level.from_xml(child)
        #       self._levels[level.description] = level
        #       try:
        #           self._levels[level.index] = level
        #       except:
        #           pass
        #   else:
        #       raise XMLAtlasException("Unknown child '%s' within data" % child.tag)
        #   self.nlevels += 1
        #pass


    @staticmethod
    ##REF: Name was automagically refactored
    def _check_version(version):
        return re.search('^[0-9]+\.[0-9]', version) is not None


class FSLProbabilisticAtlas(FSLAtlas):
    """Probabilistic FSL atlases
    """

    def __init__(self, thr=0.0, strategy='all', sort=True, *args, **kwargs):
        """

        Parameters
        ----------
        thr : float
          Value to threshold at
        strategy : str
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

    __doc__ = enhanced_doc_string('FSLProbabilisticAtlas', locals(), FSLAtlas)

    ##REF: Name was automagically refactored
    def label_voxel(self, c, levels=None):
        """Return labels for the voxel

        Parameters
        ----------
        c : tuple of coordinates (xyz)
        - levels : just for API consistency (heh heh). Must be 0 for FSL atlases
        """

        if levels is not None and not (levels in [0, [0], (0,)]):
            raise ValueError, \
                  "I guess we don't support levels other than 0 in FSL atlas." \
                  " Got levels=%s" % (levels,)
        # check range
        c = self._check_range(c)

        # XXX think -- may be we should better assign each map to a
        # different level
        level = 0
        resultLabels = []
        for index, area in enumerate(self._levels[level]):
            prob =  int(self._data[index, c[0], c[1], c[2]])
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

        See :class:`~mvpa2.atlases.base.Level.find` for more info
        """
        return self.levels[0].find(*args, **kwargs)

    def get_map(self, target, strategy='unique', axes_order='xyz'):
        """Return a probability map as an array

        Parameters
        ----------
        target : int or str or re._pattern_type
          If int, map for given index is returned. Otherwise, .find is called
          with ``unique=True`` to find matching area
        strategy : str in ('unique', 'max')
          If 'unique', then if multiple areas match, exception would be raised.
          In case of 'max', each voxel would get maximal value of probabilities
          from all matching areas
        axes_order : str in ('xyz', 'zyx')
          In what order axes of the returned array should follow.
        """
        if isinstance(target, int):
            res = self._data[target]
            # since we no longer support pynifti all is XYZ
            if axes_order == 'xyz':
                return res
            elif axes_order == 'zyx':
                return res.T
            else:
                raise ValueError, \
                      "Unknown axes_order=%r provided" % (axes_order,)
        else:
            lev = self.levels[0]       # we have just 1 here
            if strategy == 'unique':
                return self.get_map(lev.find(target, unique=True).index,
                                    axes_order=axes_order)
            else:
                maps_dict = self.get_maps(target, axes_order=axes_order)
                maps = np.array(maps_dict.values())
                return np.max(maps, axis=0)

    def get_maps(self, target, axes_order='xyz', key_attr=None,
                 overlaps=None):
        """Return a dictionary of probability maps for the target

        Each key is a `Label` instance, and value is the probability map

        Parameters
        ----------
        target : str or re._pattern_type
          .find is called with a target and unique=False to find all matches
        axes_order : str in ('xyz', 'zyx')
          In what order axes of the returned array should follow.
        key_attr : None or str
          What to use for the keys of the dictionary.  If None,
          `Label` instance would be used as a key.  If some attribute
          provided (e.g. 'text', 'abbr', 'index'), corresponding
          attribute of the `Label` instance would be taken as a key.
        overlaps : None or {'max'}
          How to treat overlaps in maps.  If None, nothing is done and maps
          might have overlaps.  If 'max', then maps would not overlap and
          competing maps will be resolved based on maximal value (e.g. if
          maps contain probabilities).
        """
        lev = self.levels[0]       # we have just 1 here
        if key_attr is None:
            key_gen = lambda x: x
        else:
            key_gen = lambda x: getattr(x, key_attr)

        res = [[key_gen(l),
                self.get_map(l.index, axes_order=axes_order)]
               for l in lev.find(target, unique=False)]

        if overlaps == 'max':
            # not efficient since it places all maps back into a single
            # ndarray... but well
            maps = np.array([x[1] for x in res])
            maximums = np.argmax(maps, axis=0)
            overlaps = np.sum(maps != 0, axis=0)>1
            # now lets go and infiltrate maps:
            # and do silly loop since we will reassign
            # the entries possibly
            for i in xrange(len(res)):
                n, m = res[i]
                loosers = np.logical_and(overlaps, ~(maximums == i))
                if len(loosers):
                    # copy and modify
                    m_new = m.copy()
                    m_new[loosers] = 0
                    res[i][1] = m_new
        elif overlaps is None:
            pass
        else:
            raise ValueError, \
                  "Incorrect value of overlaps argument %s" % overlaps
        return dict(res)

class FSLLabelsAtlas(XMLBasedAtlas):
    """Not sure what this one was for"""
    def __init__(self, *args, **kwargs):
        """not implemented"""
        FSLAtlas.__init__(self, *args, **kwargs)
        raise NotImplementedError


